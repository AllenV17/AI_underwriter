"""AI Underwriter agent: plan -> run tools -> synthesise a cited, compliant answer.

The orchestration is an explicit state graph (no framework needed for this control flow):

    classify intent -> plan tool sequence -> execute tools (state: evidence + computed
    results) -> synthesise with the LLM -> validate citations -> compliance screen

Intent routing is deterministic (keyword rules), so tool selection is testable and does
not depend on a model supporting function calling. The LLM is used for reading
evidence and writing, never for arithmetic: computed values are passed in verbatim.
"""

import logging
import re
from dataclasses import dataclass, field

from app.agent import tools as T
from app.core.compliance import ADVICE_REFUSAL_PREFIX, COMPLIANCE_RULES, VALUATION_NOTICE, is_advice_request, screen_output
from app.rag.citations import format_evidence, resolve_citations
from app.services.llm_service import LLMError, LLMNotConfiguredError, LLMService
from app.services.retrieval_service import RetrievalService

log = logging.getLogger(__name__)

# intent -> (keywords, tool plan). A keyword matches at a word start (so "concentrat" covers "concentration");
# multi-word phrases count double. The highest score wins and ties go to the earlier entry.
INTENTS: list[tuple[str, tuple[str, ...], list[tuple[str, dict]]]] = [
    ("valuation", ("valuation", "fair value", "intrinsic value", "price range", "dcf", "multiples", "overvalued", "undervalued", "peer", "how much is it worth", "is it worth"),
     [("get_financial_data", {}), ("calculate_ratios", {}), ("find_comparable_companies", {}),
      ("calculate_comparable_valuation", {}), ("calculate_dcf", {}), ("predict_listing_price", {})]),
    ("listing", ("listing price", "listing gain", "list at", "grey market", "gmp", "listing day"),
     [("predict_listing_price", {}), ("retrieve_evidence", {"topic": "offer price basis price band", "chapter_keys": ["basis_for_offer_price", "the_offer"]})]),
    ("approvals", ("approval", "licen", "permit", "registration", "certificate"),
     [("retrieve_evidence", {"topic": "{q}", "chapter_keys": ["approvals", "regulatory_disclosures", "regulations", "litigation"]})]),
    ("risk", ("risk", "litigation", "legal", "regulatory", "concentrat", "dependen", "top customers", "key customers risk", "contingent", "related party", "related-party"),
     [("analyze_risk_factors", {"focus": "{q}"})]),
    ("management", ("promoter", "management", "director", "board", "ceo", "kmp", "shareholding", "key managerial"),
     [("retrieve_evidence", {"topic": "{q}", "chapter_keys": ["promoters", "management", "capital_structure", "group_companies"]})]),
    ("business", ("business", "what does the company do", "products", "customers", "segments", "industry", "competit", "market share", "strategy", "geograph", "employee", "manufactur", "plants"),
     [("retrieve_evidence", {"topic": "{q}", "chapter_keys": ["our_business", "industry_overview", "offer_document_summary"]})]),
    ("offer", ("objects", "use of proceeds", "net proceeds", "proceeds", "issue size", "offer size", "ofs", "offer for sale", "fresh issue", "price band", "lead manager", "book running", "registrar", "lot size"),
     [("retrieve_evidence", {"topic": "{q}", "chapter_keys": ["objects_of_offer", "the_offer", "capital_structure", "general_information", "basis_for_offer_price"]})]),
    ("financial_health", ("financial health", "financial", "ratio", "profit", "revenue", "margin", "debt", "leverage", "cash flow",
                          "ebitda", "balance sheet", "liquidity", "roe", "earnings", "growth", "net worth", "networth"),
     [("get_financial_data", {}), ("calculate_ratios", {}),
      ("retrieve_evidence", {"topic": "{q}", "chapter_keys": ["mdna", "restated_financials", "financial_summary"]})]),
]


def classify(question: str) -> str:
    q = question.lower()
    best, best_score = "general", 0
    for name, kws, _ in INTENTS:
        score = sum((2 if " " in k else 1) for k in kws if re.search(r"\b" + re.escape(k), q))
        if score > best_score:  # strict: ties keep the earlier intent
            best, best_score = name, score
    return best


def plan_for(intent: str, question: str) -> list[tuple[str, dict]]:
    for name, _, plan in INTENTS:
        if name == intent:
            return [(tool, {k: (v.replace("{q}", question) if isinstance(v, str) else v) for k, v in args.items()})
                    for tool, args in plan]
    return [("search_drhp", {"query": question})]


@dataclass
class AgentAnswer:
    answer: str
    intent: str
    steps: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    computed: list[str] = field(default_factory=list)
    llm_used: bool = False
    compliance_flagged: bool = False
    warnings: list[str] = field(default_factory=list)


ANSWER_INSTRUCTIONS = """Answer the user's question about the company using ONLY the evidence and the computed results below.
- Cite evidence inline with its label, e.g. [S2] or [S1, S3]. Cite only labels that appear in the evidence.
- Quote computed figures exactly as given; do not recalculate or estimate new numbers.
- Clearly label model-generated estimates (valuations) as estimates.
- If the evidence does not answer the question, say "This information was not found in the uploaded DRHP."
- Be concise and structured (short paragraphs or bullets). No investment recommendation."""


class UnderwriterAgent:
    def __init__(self, document_id: str, retrieval: RetrievalService | None = None, llm: LLMService | None = None) -> None:
        self.document_id = document_id
        self.retrieval = retrieval or RetrievalService()
        self.llm = llm or LLMService()

    def run(self, question: str, *, peers: list | None = None, dcf_assumptions: dict | None = None,
            shares: float | None = None, price_per_share: float | None = None) -> AgentAnswer:
        intent = classify(question)
        ctx = T.ToolContext(self.document_id, self.retrieval, self.llm, evidence=[])
        result = AgentAnswer(answer="", intent=intent)
        computed: list[str] = []

        plan = plan_for(intent, question)
        if intent != "general" and not any(t in ("search_drhp", "retrieve_evidence", "analyze_risk_factors", "find_comparable_companies") for t, _ in plan):
            plan.append(("search_drhp", {"query": question}))
        for tool, args in plan:
            if tool == "calculate_comparable_valuation":
                args = {"peers": peers, "shares": shares}
            elif tool == "calculate_dcf":
                args = {"assumptions": dcf_assumptions}
            elif tool == "predict_listing_price":
                args = {"issue_price": price_per_share}
            elif tool == "calculate_ratios":
                args = {"price_per_share": price_per_share, "shares": shares}
            try:
                out = T.TOOLS[tool](ctx, **args)
            except Exception as e:  # one failing tool must not sink the answer
                log.exception("Tool %s failed", tool)
                out = {"tool": tool, "available": False, "computed": f"{tool} failed: {e}"}
            result.steps.append({"tool": tool, "available": out.get("available", True),
                                 "evidence_refs": out.get("evidence_refs", []), "summary": str(out.get("computed", ""))[:300]})
            if out.get("computed"):
                computed.append(f"[{tool}]\n{out['computed']}")

        result.computed = computed
        result.evidence = [e.to_dict() for e in ctx.evidence]
        advice = is_advice_request(question)
        if intent in ("valuation", "listing"):
            computed.append(f"[notice]\n{VALUATION_NOTICE}")

        user_prompt = (f"QUESTION: {question}\n\nEVIDENCE:\n{format_evidence(ctx.evidence) or '(none retrieved)'}\n\n"
                       f"COMPUTED RESULTS (deterministic):\n" + ("\n\n".join(computed) or "(none)") + f"\n\n{ANSWER_INSTRUCTIONS}")
        try:
            resp = self.llm.chat(COMPLIANCE_RULES, user_prompt, max_tokens=1500)
            cited = resolve_citations(resp.text, ctx.evidence)
            if cited.invalid_refs:
                result.warnings.append(f"Removed citations to evidence that was not retrieved: {cited.invalid_refs}")
            text = cited.text
            result.citations = cited.citations
            result.llm_used = True
        except LLMNotConfiguredError:
            text = ("The LLM is not configured (set OPENROUTER_API_KEY), so no written answer was generated. "
                    "The retrieved evidence and computed results are shown below.")
            if computed:
                text += "\n\n" + "\n\n".join(computed)
        except LLMError as e:
            text = f"The LLM request failed ({e}). The retrieved evidence and computed results are shown below."
            result.warnings.append(str(e))

        if advice:
            text = ADVICE_REFUSAL_PREFIX + text
        screened = screen_output(text)
        result.answer, result.compliance_flagged = screened.text, screened.flagged
        return result

