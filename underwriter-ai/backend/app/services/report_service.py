"""Underwriting report: assemble sections, then render to PDF.

Structure (separates fundamental valuation, issue-price analysis and listing scenarios):
 1 Executive Summary          10 Comparable Company Valuation
 2 Company Overview           11 DCF Valuation
 3 Business Model             12 Historical IPO Prediction
 4 Industry Overview          13 Estimated Fundamental Value Range
 5 Financial Performance      14 Potential Listing-Price Scenario
 6 Financial Ratio Analysis   15 Key Assumptions
 7 Financial Health Assessment 16 Data Limitations
 8 Risk Factor Analysis       17 Regulatory and Investment Disclaimer
 9 Management and Promoters   18 Sources and Citations

Narrative sections are written by the LLM from retrieved evidence with validated
citations; tables and valuations come from deterministic code. Nothing in the report is
a buy/sell/subscribe recommendation.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from app.agent import tools as T
from app.core.compliance import COMPLIANCE_RULES, DISCLAIMER, LISTING_NOTICE, VALUATION_NOTICE, screen_output
from app.rag.citations import format_evidence, resolve_citations
from app.services.llm_service import LLMError, LLMNotConfiguredError, LLMService
from app.services.retrieval_service import RetrievalService
from app.utils.storage import load_ir, processed_dir

log = logging.getLogger(__name__)

NARRATIVE = {
    "Company Overview": (["company history incorporation headquarters overview", "summary of the business key highlights"],
                         ["offer_document_summary", "our_business", "history_corporate"]),
    "Business Model": (["business model products services revenue streams", "revenue by segment geography customers",
                        "key customers suppliers concentration", "competitive strengths strategies"],
                       ["our_business", "offer_document_summary"]),
    "Industry Overview": (["industry size growth outlook", "competitive landscape market share peers"], ["industry_overview"]),
    "Management and Promoter Analysis": (["promoters background experience shareholding", "board of directors key managerial personnel",
                                          "related party transactions", "outstanding litigation against promoters directors"],
                                         ["promoters", "management", "related_party", "litigation", "capital_structure"]),
}

SECTION_INSTRUCTIONS = """Write the "{title}" section of an IPO underwriting report from the evidence only.
Cite every factual statement with its evidence label, e.g. [S1]. 120-250 words, plain prose or short bullets.
If something important is not in the evidence, say it was not found. No investment recommendation."""


class ReportService:
    def __init__(self, document_id: str, llm: LLMService | None = None, retrieval: RetrievalService | None = None) -> None:
        self.document_id = document_id
        self.llm = llm or LLMService()
        self.retrieval = retrieval or RetrievalService()
        self.all_citations: list[dict] = []
        self.flags: list[str] = []

    # ------------------------------------------------------------------ sections
    def _narrative(self, title: str, queries: list[str], keys: list[str], extra: str = "") -> dict:
        ctx = T.ToolContext(self.document_id, self.retrieval, self.llm, evidence=[])
        for q in queries:
            T.retrieve_evidence(ctx, q, keys)
        ev = ctx.evidence[:10]
        try:
            resp = self.llm.chat(COMPLIANCE_RULES, SECTION_INSTRUCTIONS.format(title=title) + extra
                                 + f"\n\nEVIDENCE:\n{format_evidence(ev, 1600)}", max_tokens=900)
            cited = resolve_citations(resp.text, ev)
            sc = screen_output(cited.text)
            if sc.flagged:
                self.flags.append(title)
            self.all_citations += [{**c, "section_title": title} for c in cited.citations]
            return {"title": title, "text": sc.text, "citations": cited.citations}
        except (LLMNotConfiguredError, LLMError) as e:
            excerpts = "\n".join(f"- p.{e_.chunk.page_number} ({e_.chunk.section_label}): {e_.chunk.text[:300]}..." for e_ in ev[:4])
            return {"title": title, "text": f"Narrative not generated ({e}). Most relevant DRHP passages:\n{excerpts}",
                    "citations": [{"ref": x.ref, "page_number": x.chunk.page_number, "section": x.chunk.section_label} for x in ev[:4]]}

    def build(self, peers: list | None = None, shares: float | None = None, dcf_assumptions: dict | None = None,
              issue_price: float | None = None) -> dict:
        ir = load_ir(self.document_id)
        company = (ir.company_name if ir else None) or "the company"
        ctx = T.ToolContext(self.document_id, self.retrieval, self.llm, evidence=[])
        fin = T.get_financial_data(ctx)
        ratios = T.calculate_ratios(ctx, issue_price, shares)
        risks = T.analyze_risk_factors(ctx)
        self.all_citations += [{"ref": s["ref"], "page_number": s["page_number"], "section": s["section"], "section_title": "Risk Factor Analysis"}
                               for r in risks.get("risks", []) for s in r.get("sources", [])]
        comps = T.calculate_comparable_valuation(ctx, peers, shares) if peers else None
        dcf = T.calculate_dcf(ctx, dcf_assumptions) if dcf_assumptions else None
        listing = T.predict_listing_price(ctx, issue_price)

        sections: list[dict] = []
        narratives = {t: self._narrative(t, q, k) for t, (q, k) in NARRATIVE.items()}
        sections += [narratives["Company Overview"], narratives["Business Model"], narratives["Industry Overview"]]
        sections.append({"title": "Financial Performance", "text": fin["computed"], "kind": "computed",
                         "checks": fin.get("checks", []), "warnings": fin.get("warnings", [])})
        sections.append({"title": "Financial Ratio Analysis", "text": ratios["computed"], "kind": "computed",
                         "ratios": ratios.get("ratios", [])})
        sections.append(self._health(fin, ratios))
        sections.append({"title": "Risk Factor Analysis", "kind": "risks", "risks": risks.get("risks", []),
                         "text": "" if risks.get("available") else risks["computed"]})
        sections.append(narratives["Management and Promoter Analysis"])
        sections.append({"title": "Comparable Company Valuation", "kind": "valuation",
                         "text": comps["computed"] if comps else "Not performed: no peer data was entered. Peer figures are never generated automatically.",
                         "result": comps.get("result") if comps else None, "notice": VALUATION_NOTICE})
        sections.append({"title": "DCF Valuation", "kind": "valuation",
                         "text": dcf["computed"] if dcf else "Not performed: DCF assumptions were not provided.",
                         "result": dcf.get("result") if dcf else None, "notice": VALUATION_NOTICE})
        sections.append({"title": "Historical IPO Prediction", "text": listing["computed"]})
        sections.append(self._fundamental_range(comps, dcf, issue_price))
        pred = listing.get("result") or {}
        if pred.get("status") == "ok" and pred.get("price_range"):
            lo, hi = pred["price_range"]
            scenario = (f"Model scenario for the listing price: Rs. {lo:,.2f} to Rs. {hi:,.2f} "
                        f"({pred['predicted_listing_gain']:+.1%} central estimate). {listing['computed']} {LISTING_NOTICE}")
        elif pred.get("status") == "ok":
            scenario = f"{listing['computed']} Enter the issue price to convert this into a price range. {LISTING_NOTICE}"
        else:
            scenario = f"Not available: {listing['computed']} {LISTING_NOTICE}"
        sections.append({"title": "Potential Listing-Price Scenario", "text": scenario})
        sections.append({"title": "Key Assumptions", "kind": "list", "items": self._assumptions(comps, dcf, issue_price, shares)})
        sections.append({"title": "Data Limitations", "kind": "list", "items": self._limitations(ir, fin, risks)})
        sections.append({"title": "Regulatory and Investment Disclaimer", "text": DISCLAIMER})

        exec_summary = self._executive_summary(company, sections)
        sections.insert(0, exec_summary)
        sections.append({"title": "Sources and Citations", "kind": "sources", "items": self._dedupe_sources()})
        return {
            "company_name": company, "document_id": self.document_id, "doc_type": ir.doc_type if ir else None,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sections": sections, "disclaimer": DISCLAIMER, "compliance_flags": self.flags,
            "labels": {"fundamental_valuation": "Model-generated estimate",
                   "listing_price": "Model-generated estimate" if pred.get("status") == "ok" else "Not available (no validated model)"},
        }

    def _health(self, fin: dict, ratios: dict) -> dict:
        """Rule-based health signals from computed ratios (latest full year), then labelled."""
        rs = [r for r in ratios.get("ratios", []) if r["available"] and r["value"] is not None]
        fys = sorted({r["period"] for r in rs if r["period"].startswith("FY")}, key=lambda p: int(p[2:]))
        latest = fys[-1] if fys else None
        signals = []
        for r in rs:
            if r["period"] == latest and r["name"] in ("Net Profit Margin", "Return on Equity", "Debt-to-Equity",
                                                      "Interest Coverage Ratio", "Current Ratio", "Free Cash Flow", "Operating Cash Flow"):
                v = f"{r['value'] / 1e7:,.1f} crore" if r["unit"] == "INR" else f"{r['value']:,.2f}{r['unit']}"
                signals.append(f"{r['name']} {latest}: {v}. {r['interpretation']}")
        text = ("Signals from the latest full financial year (deterministic rules, not a credit rating):\n- " + "\n- ".join(signals)) \
            if signals else "Not assessed: ratios could not be computed from the extracted statements."
        return {"title": "Financial Health Assessment", "text": text, "kind": "computed"}

    def _fundamental_range(self, comps: dict | None, dcf: dict | None, issue_price: float | None) -> dict:
        parts = []
        if comps and comps.get("result"):
            r = comps["result"]
            rng = r.get("price_range") or r.get("equity_range")
            if rng:
                unit = "per share" if r.get("price_range") else "equity value (INR)"
                parts.append(f"Comparables: {rng[0]:,.2f} - {rng[2]:,.2f} {unit} (median {rng[1]:,.2f}).")
        if dcf and dcf.get("result"):
            r = dcf["result"]
            v = r.get("price_per_share") or r.get("equity_value")
            parts.append(f"DCF base case: {v:,.2f} {'per share' if r.get('price_per_share') else 'equity value (INR)'}; "
                         "see the sensitivity table for the range.")
        if not parts:
            return {"title": "Estimated Fundamental Value Range",
                    "text": "Not estimated: run the comparable and/or DCF valuation with your inputs first."}
        txt = " ".join(parts)
        if issue_price:
            txt += f" For reference, the issue price / price band entered is ₹{issue_price:,.2f}; comparing the two is an analysis, not a recommendation."
        return {"title": "Estimated Fundamental Value Range", "text": f"{txt}\n\n{VALUATION_NOTICE}", "kind": "valuation"}

    def _assumptions(self, comps, dcf, issue_price, shares) -> list[str]:
        items = ["Financial figures are from the DRHP's restated financial statements as extracted by the system; derived figures (EBITDA, net debt, free cash flow) are computed with the formulas shown.",
                 "Ratios use period-end balances; interim-period flow ratios are not annualised."]
        if comps and comps.get("result"):
            items += comps["result"].get("assumptions", [])
        if dcf and dcf.get("result"):
            a = dcf["result"]["assumptions"]
            items.append(f"DCF: growth {a['revenue_growth']}, EBITDA margin {a['ebitda_margin']:.1%}, WACC {a['wacc']:.1%}, terminal growth {a['terminal_growth']:.1%}, tax {a['tax_rate']:.1%} (user inputs unless sourced).")
        if shares:
            items.append(f"Share count used: {shares:,.0f} (user input).")
        if issue_price:
            items.append(f"Issue price / price band reference: ₹{issue_price:,.2f} (user input).")
        return items

    def _limitations(self, ir, fin: dict, risks: dict) -> list[str]:
        items = ["Extraction is automated and may misread complex tables; verify key figures against the cited DRHP pages.",
                 "The DRHP is a draft: figures and disclosures may change in the RHP and final prospectus.",
                 "Market data (peer prices, index levels) is not fetched automatically and is only as current as the user's inputs."]
        if ir:
            items += ir.warnings[:6]
        items += fin.get("warnings", [])[:6]
        if not risks.get("available"):
            items.append("Risk analysis could not be generated by the LLM; see the DRHP's Risk Factors section directly.")
        return items

    def _executive_summary(self, company: str, sections: list[dict]) -> dict:
        body = "\n\n".join(f"## {s['title']}\n{s.get('text', '')}" for s in sections
                           if s["title"] not in ("Regulatory and Investment Disclaimer",) and s.get("text"))
        risks = next((s for s in sections if s["title"] == "Risk Factor Analysis"), {})
        risk_txt = "\n".join(f"- [{r.get('severity')}] {r.get('title')}" for r in risks.get("risks", [])[:6])
        prompt = (f"Write a 150-220 word executive summary of this IPO underwriting report on {company}. Summarise the "
                  "business, financial performance (quote figures exactly as given), key risks and the status of the valuation "
                  "work. Do not add new facts or numbers. No recommendation to buy, sell or subscribe.\n\n"
                  f"REPORT SECTIONS:\n{body[:12000]}\n\nKEY RISKS:\n{risk_txt}")
        try:
            text = self.llm.chat(COMPLIANCE_RULES, prompt, max_tokens=600).text
            text = resolve_citations(text, []).text  # strips any stray [S#] (summary cites sections, not chunks)
            sc = screen_output(text)
            if sc.flagged:
                self.flags.append("Executive Summary")
            return {"title": "Executive Summary", "text": sc.text}
        except (LLMNotConfiguredError, LLMError) as e:
            return {"title": "Executive Summary", "text": f"Not generated ({e})."}

    def _dedupe_sources(self) -> list[str]:
        seen, out = set(), []
        for c in sorted(self.all_citations, key=lambda c: (c.get("page_number") or 0)):
            key = (c.get("page_number"), c.get("section"))
            if key not in seen:
                seen.add(key)
                out.append(f"DRHP page {c.get('page_number')}: {c.get('section')} (used in {c.get('section_title')})")
        return out


# ---------------------------------------------------------------------- PDF

def render_pdf(report: dict, path: Path | None = None) -> Path:
    """Render a report dict to PDF with reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    path = path or processed_dir(report["document_id"]) / f"underwriting_report_{report['generated_at'][:10]}.pdf"
    ss = getSampleStyleSheet()
    navy = colors.HexColor("#1E2761")
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=navy, fontSize=20, alignment=TA_LEFT)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=navy, spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#52514e"))
    box = ParagraphStyle("box", parent=small, backColor=colors.HexColor("#EEF2FB"), borderPadding=6)

    def esc(t: str) -> str:
        # Built-in PDF fonts have no rupee glyph; write "Rs." rather than a missing-glyph box.
        t = (t or "").replace("₹", "Rs. ").replace("Rs.  ", "Rs. ")
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def paras(text: str) -> list:
        out = []
        for block in (text or "").split("\n"):
            b = block.strip()
            if not b:
                continue
            if b.startswith(("- ", "* ")):
                out.append(Paragraph("• " + esc(b[2:]), body))
            else:
                out.append(Paragraph(esc(b.lstrip("#").strip()), body))
        return out

    story = [Paragraph(f"IPO Underwriting Report: {esc(report['company_name'])}", h1),
             Paragraph(f"Generated {report['generated_at']} by UnderwriterAI · {esc(report.get('doc_type') or 'offer document')} analysis", small),
             Spacer(1, 6), Paragraph(esc(DISCLAIMER), box), Spacer(1, 8)]
    for i, s in enumerate(report["sections"], 1):
        story.append(Paragraph(f"{i}. {esc(s['title'])}", h2))
        kind = s.get("kind")
        if kind == "risks" and s.get("risks"):
            rows = [["Severity", "Risk", "Evidence"]]
            for r in s["risks"]:
                src = ", ".join(f"p.{x['page_number']}" for x in r.get("sources", []))
                rows.append([r.get("severity", ""), Paragraph(f"<b>{esc(r.get('title', ''))}</b> ({esc(r.get('category', ''))}): "
                                                               f"{esc(r.get('description', ''))} <i>Impact:</i> {esc(r.get('financial_impact', ''))}", small),
                             src])
            t = Table(rows, colWidths=[2 * cm, 12.5 * cm, 2.5 * cm], repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                   ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D6DCEA"))]))
            story.append(t)
        elif kind in ("list", "sources"):
            story += [Paragraph("• " + esc(x), small if kind == "sources" else body) for x in s.get("items", [])]
        else:
            story += paras(s.get("text", ""))
        if s.get("notice"):
            story.append(Paragraph(esc(s["notice"]), box))
    story.append(Spacer(1, 10))
    story.append(KeepTogether([Paragraph("Disclaimer", h2), Paragraph(esc(DISCLAIMER), box)]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#52514e"))
        canvas.drawString(2 * cm, 1.2 * cm, "AI-generated analysis for informational and educational purposes only. Not investment advice.")
        canvas.drawRightString(19 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title=f"Underwriting report - {report['company_name']}").build(story, onFirstPage=footer, onLaterPages=footer)
    return path
