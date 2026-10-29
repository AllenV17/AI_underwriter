"""Agent tools. Each wraps one module; numbers always come from deterministic code.

Tool outputs are plain dicts with an `evidence` list (retrieved chunks, cited by [S#]) and/or
`computed` results (quoted to the LLM verbatim, never recomputed by it).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.core.compliance import COMPLIANCE_RULES
from app.services.llm_service import LLMError, LLMNotConfiguredError, LLMService
from app.services.retrieval_service import Evidence, RetrievalService

log = logging.getLogger(__name__)


@dataclass
class ToolContext:
    document_id: str
    retrieval: RetrievalService
    llm: LLMService
    evidence: list[Evidence]

    def add_evidence(self, items: list[Evidence]) -> list[Evidence]:
        """Merge new evidence, renumbering refs S1..Sn across all tool calls."""
        known = {e.chunk.chunk_id for e in self.evidence}
        for e in items:
            if e.chunk.chunk_id not in known:
                self.evidence.append(e)
                known.add(e.chunk.chunk_id)
        for i, e in enumerate(self.evidence, 1):
            e.ref = f"S{i}"
        return [e for e in self.evidence if e.chunk.chunk_id in {x.chunk.chunk_id for x in items}]


def search_drhp(ctx: ToolContext, query: str, chapter_keys: list[str] | None = None, k: int = 5) -> dict:
    res = ctx.retrieval.retrieve(ctx.document_id, query, final_k=k, chapter_keys=chapter_keys)
    added = ctx.add_evidence(res.evidence)
    return {"tool": "search_drhp", "query": query, "evidence_refs": [e.ref for e in added], "reranked": res.reranked}


def retrieve_evidence(ctx: ToolContext, topic: str, chapter_keys: list[str]) -> dict:
    """Section-filtered retrieval, falling back to the whole document when the section is empty."""
    out = search_drhp(ctx, topic, chapter_keys)
    if not out["evidence_refs"]:
        out = search_drhp(ctx, topic)
    return out


def get_financial_data(ctx: ToolContext) -> dict:
    from app.services.financial_extraction_service import load_financials

    data = load_financials(ctx.document_id)
    if data is None or not data.values:
        return {"tool": "get_financial_data", "available": False, "computed": "No financial statements were extracted."}
    key = ["revenue_from_operations", "total_income", "ebitda", "profit_before_tax", "profit_after_tax", "eps_basic",
           "total_assets", "total_equity", "total_debt", "cash_and_equivalents", "operating_cash_flow", "free_cash_flow"]
    lines = [f"Basis: {data.basis}; unit as printed: {data.unit}; values below in INR crore (1 crore = 10 million)."]
    for f in key:
        cells = []
        for p in data.periods:
            v = data.get(f, p)
            if v and v.value_inr is not None:
                val = v.value_inr if f == "eps_basic" else v.value_inr / 1e7
                cells.append(f"{p}: {val:,.2f}" + (" (₹/share)" if f == "eps_basic" else "") + f" [p.{v.source_page}{', derived' if v.derived else ''}]")
        if cells:
            label = next(v.label for v in data.values if v.field == f)
            lines.append(f"- {label}: " + "; ".join(cells))
    checks = [f"{c.name} ({c.period}): {'pass' if c.passed else 'FAIL' if c.passed is False else 'not checked'}" for c in data.checks]
    return {"tool": "get_financial_data", "available": True, "computed": "\n".join(lines),
            "checks": checks, "warnings": data.warnings}


def calculate_ratios(ctx: ToolContext, price_per_share: float | None = None, shares: float | None = None) -> dict:
    from app.services.ratio_service import FinancialsNotAvailableError, compute_ratios

    try:
        ratios, meta = compute_ratios(ctx.document_id, price_per_share, shares)
    except FinancialsNotAvailableError as e:
        return {"tool": "calculate_ratios", "available": False, "computed": str(e)}
    # Keep the two latest full financial years (plus the latest interim period) to stay concise.
    fys = sorted({r.period for r in ratios if r.period.startswith("FY")}, key=lambda p: int(p[2:]))[-2:]
    stubs = [p for p in meta["periods"] if not p.startswith("FY")][:1]
    keep = set(fys) | set(stubs)
    lines = [f"Financial company (lender) treatment: {meta['financial_company']}. Periods shown: {', '.join(sorted(keep))}."]
    for r in ratios:
        if r.available and r.unit in ("%", "x") and r.period in keep:
            lines.append(f"- {r.name} ({r.period}): {r.value:,.2f}{r.unit}. {r.interpretation}")
    na = sorted({f"{r.name}: {r.reason}" for r in ratios if not r.available and r.category != "Valuation"})
    if na:
        lines.append("Not calculated: " + " | ".join(na[:8]))
    return {"tool": "calculate_ratios", "available": True, "computed": "\n".join(lines),
            "ratios": [r.to_dict() for r in ratios]}


RISK_PROMPT = """From the evidence below, list the most material risk factors disclosed in the offer document.
Return JSON: {"risks": [{"title": str, "category": one of ["Business","Financial","Industry","Regulatory",
"Promoter/Management","Litigation","Related-party","Concentration","Valuation","Market","Other"],
"description": str, "severity": one of ["High","Medium","Low"], "financial_impact": str,
"evidence_refs": ["S1", ...]}]}
Rules: use only risks stated in the evidence; every risk must cite at least one evidence ref; severity is your
assessment of materiality based on the disclosure and must be justified in the description; do not invent figures.
Return at most 10 risks."""


def analyze_risk_factors(ctx: ToolContext, focus: str | None = None) -> dict:
    from app.rag.citations import format_evidence

    queries = [focus] if focus else [
        "material risks to the business and operations", "dependence on key customers suppliers concentration",
        "financial risks debt cash flows losses", "regulatory and legal risks litigation", "promoter and related party risks"]
    refs: list[str] = []
    for q in queries:
        refs += retrieve_evidence(ctx, q, ["risk_factors", "litigation", "related_party"])["evidence_refs"]
    ev = [e for e in ctx.evidence if e.ref in set(refs)]
    try:
        js = ctx.llm.chat_json(COMPLIANCE_RULES, f"{RISK_PROMPT}\n\nEVIDENCE:\n{format_evidence(ev, 1500)}", max_tokens=2500)
    except (LLMNotConfiguredError, LLMError) as e:
        return {"tool": "analyze_risk_factors", "available": False, "computed": f"Risk analysis needs the LLM: {e}", "evidence_refs": refs}
    valid = {e.ref: e for e in ev}
    risks = []
    for r in js.get("risks", []):
        cited = [x for x in r.get("evidence_refs", []) if x in valid]
        if not cited:  # uncited risks are dropped: they cannot be verified
            continue
        r["evidence_refs"] = cited
        r["sources"] = [{"ref": x, "page_number": valid[x].chunk.page_number, "section": valid[x].chunk.section_label} for x in cited]
        risks.append(r)
    lines = [f"- [{r.get('severity')}] {r.get('title')}: {r.get('description')} [{', '.join(r['evidence_refs'])}]" for r in risks]
    return {"tool": "analyze_risk_factors", "available": True, "risks": risks, "computed": "\n".join(lines), "evidence_refs": refs}


def find_comparable_companies(ctx: ToolContext) -> dict:
    """Peers as disclosed in the DRHP's Basis for Offer Price section (no external data is fetched)."""
    out = retrieve_evidence(ctx, "comparison with listed industry peers P/E EPS RoNW NAV peer companies", ["basis_for_offer_price"])
    from app.services.financial_extraction_service import load_peers

    out["tool"] = "find_comparable_companies"
    saved = load_peers(ctx.document_id) or {}
    rows = [f"- {p['name']}: P/E {p['pe']}x (DRHP p.{p['source'].split('p.')[1].split(' ')[0]})" for p in saved.get("suggested", []) if p.get("pe")]
    out["computed"] = ("Listed peers are taken only from the DRHP's own peer comparison (see evidence). Peer market "
                       "multiples must be entered or confirmed by the user before a comparable valuation is run."
                       + (("\nPeer P/E as printed in the DRHP (names may need checking against the page):\n" + "\n".join(rows)) if rows else
                          ("\n" + saved["note"] if saved.get("note") else "")))
    return out


def calculate_comparable_valuation(ctx: ToolContext, peers: list | None = None, shares: float | None = None) -> dict:
    if not peers:
        return {"tool": "calculate_comparable_valuation", "available": False,
                "computed": "Comparable valuation not run: no peer data has been entered (peer figures are never generated automatically)."}
    from app.services.valuation_service import comparable_valuation

    res = comparable_valuation(ctx.document_id, peers, shares_outstanding=shares)
    rng = res.get("price_range") or res.get("equity_range")
    return {"tool": "calculate_comparable_valuation", "available": True, "result": res,
            "computed": f"Comparable valuation range (low, mid, high): {rng}. {res['notice']}"}


def calculate_dcf(ctx: ToolContext, assumptions: dict | None = None) -> dict:
    if not assumptions:
        return {"tool": "calculate_dcf", "available": False,
                "computed": "DCF not run: it needs user assumptions (growth, margin, WACC, terminal growth). Use the Valuation dashboard."}
    from app.services.valuation_service import dcf_valuation
    from app.valuation.dcf import DCFAssumptions

    res = dcf_valuation(DCFAssumptions(**assumptions))
    return {"tool": "calculate_dcf", "available": True, "result": res,
            "computed": f"DCF equity value: {res['equity_value']:,.0f} INR; price per share: {res['price_per_share']}. {res['notice']}"}


def predict_listing_price(ctx: ToolContext) -> dict:
    from app.valuation.listing_predictor import ListingPricePredictor

    pred = ListingPricePredictor().predict({})
    return {"tool": "predict_listing_price", "available": pred.status == "ok", "computed": pred.message, "result": pred.to_dict()}


TOOLS: dict[str, Callable] = {f.__name__: f for f in (
    search_drhp, retrieve_evidence, get_financial_data, calculate_ratios, analyze_risk_factors,
    find_comparable_companies, calculate_comparable_valuation, calculate_dcf, predict_listing_price)}
