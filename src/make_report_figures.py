"""Generate the charts and diagrams for the Stage 2 report.

Charts are computed from the parser output in data/processed/, so they stay in step
with the data. Diagrams are drawn from fixed text. Output: report_figures/*.png

Usage:
  python src/make_report_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "report_figures"
OUT.mkdir(exist_ok=True)

# Validated categorical palette (dataviz reference, light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
NAVY = "#1E2761"

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

SHORT = {
    "BajajHousingFinance_DRHP": "Bajaj Housing Finance", "DOMSIndustries_DRHP": "DOMS Industries",
    "HonasaConsumer_DRHP": "Honasa Consumer", "HyundaiMotorIndia_DRHP": "Hyundai Motor India",
    "IREDA_DRHP": "IREDA", "JSWInfrastructure_DRHP": "JSW Infrastructure", "MankindPharma_DRHP": "Mankind Pharma",
    "PremierEnergies_RHP": "Premier Energies (RHP)", "TataTechnologies_DRHP": "Tata Technologies",
    "Unicommerce_DRHP": "Unicommerce",
}

# DRHP chapters grouped into the parts of the document later stages use.
GROUPS = [
    ("Front matter & definitions", MUTED, {"front_matter", "definitions", "presentation_conventions", "forward_looking", "offer_document_summary"}),
    ("Risk factors", ORANGE, {"risk_factors"}),
    ("Offer & valuation", AQUA, {"the_offer", "financial_summary", "general_information", "capital_structure", "objects_of_offer", "basis_for_offer_price", "tax_benefits"}),
    ("Business & management", YELLOW, {"industry_overview", "our_business", "regulations", "history_corporate", "subsidiaries", "management", "promoters", "group_companies", "dividend_policy"}),
    ("Financial information", BLUE, {"restated_financials", "other_financial_info", "statistical_info", "related_party", "mdna", "capitalisation", "indebtedness"}),
    ("Legal & offer procedure", MAGENTA, None),  # everything else
]


def group_of(key):
    for name, _, keys in GROUPS:
        if keys and key in keys:
            return name
    return GROUPS[-1][0]


def load_docs():
    docs = {}
    for d in sorted(PROCESSED.iterdir()):
        s = json.loads((d / "sections.json").read_text(encoding="utf-8"))
        pages = {g[0]: 0 for g in GROUPS}
        first = s["chapters"][0]["pdf_start"] if s["chapters"] else s["n_pages"] + 1
        pages[GROUPS[0][0]] += first - 1  # cover + contents before the first chapter
        for c in s["chapters"]:
            pages[group_of(c["key"])] += c["pdf_end"] - c["pdf_start"] + 1
        docs[d.name] = {"n_pages": s["n_pages"], "chapters": len(s["chapters"]), "pages": pages}
    return docs


def clean(ax):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------- charts ----------

def chart_pages(docs):
    rows = sorted(docs.items(), key=lambda kv: kv[1]["n_pages"])
    names = [SHORT[k] for k, _ in rows]
    vals = [v["n_pages"] for _, v in rows]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    colors = [ORANGE if "RHP)" in n else BLUE for n in names]
    bars = ax.barh(names, vals, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(v + 6, b.get_y() + b.get_height() / 2, f"{v}", va="center", fontsize=10, color=INK)
    ax.axvline(total / len(vals), color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(total / len(vals) + 4, len(vals) - 0.35, f"average {total / len(vals):.0f}", fontsize=9, color=INK2)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="y", length=0)
    clean(ax)
    ax.spines["bottom"].set_visible(False)
    fig.suptitle(f"The pilot documents run 361–664 pages, {total:,} pages in total", x=0.02, ha="left", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.02, 0.905, "Pages per offer document · blue = DRHP, orange = RHP", ha="left", fontsize=10, color=INK2)
    save(fig, "fig_pages_per_document.png")


def chart_page_share(docs):
    tot = {g[0]: sum(d["pages"][g[0]] for d in docs.values()) for g in GROUPS}
    allp = sum(tot.values())
    order = ["Financial information", "Business & management", "Legal & offer procedure", "Risk factors", "Offer & valuation", "Front matter & definitions"]
    colors = {g[0]: g[1] for g in GROUPS}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    vals = [tot[k] for k in order]
    wedges, _ = ax.pie(vals, colors=[colors[k] for k in order], startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.38, edgecolor=SURFACE, linewidth=2))
    ax.text(0, 0.08, f"{allp:,}", ha="center", va="center", fontsize=20, fontweight="bold", color=NAVY, family="Cambria")
    ax.text(0, -0.2, "pages", ha="center", va="center", fontsize=10, color=INK2)
    # legend doubling as direct labels (value + share), since 3 hues are low-contrast on white
    labels = [f"{k}  —  {tot[k]:,} pages ({tot[k] / allp:.0%})" for k in order]
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10, handlelength=1.1, labelspacing=0.9)
    ax.set_aspect("equal")
    fin = tot["Financial information"] / allp
    fig.suptitle(f"Financial information is the largest part of a DRHP: {fin:.0%} of all pages", x=0.02, ha="left", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.02, 0.905, "Share of the 10 pilot documents' pages, by part of the DRHP", ha="left", fontsize=10, color=INK2)
    save(fig, "fig_page_share_donut.png")


def chart_parts_per_doc(docs):
    rows = sorted(docs.items(), key=lambda kv: kv[1]["pages"]["Financial information"])
    names = [SHORT[k] for k, _ in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    left = [0] * len(rows)
    order = ["Financial information", "Business & management", "Offer & valuation", "Risk factors", "Legal & offer procedure", "Front matter & definitions"]
    colors = {g[0]: g[1] for g in GROUPS}
    for g in order:
        vals = [v["pages"][g] for _, v in rows]
        ax.barh(names, vals, left=left, color=colors[g], height=0.64, edgecolor=SURFACE, linewidth=1.5, label=g)
        if g == "Financial information":
            for i, v in enumerate(vals):
                ax.text(v / 2, i, f"{v}", va="center", ha="center", fontsize=9.5, color="white", fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("PDF pages")
    ax.tick_params(axis="y", length=0)
    clean(ax)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.13), frameon=False, fontsize=9.5, handlelength=1.1)
    fin = [v["pages"]["Financial information"] for _, v in rows]
    fig.suptitle(f"Financial sections range from {min(fin)} to {max(fin)} pages per document", x=0.02, ha="left", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.02, 0.905, "Pages per document by part of the DRHP · white figures = financial-information pages", ha="left", fontsize=10, color=INK2)
    save(fig, "fig_parts_per_document.png")
    return {SHORT[k]: v["pages"]["Financial information"] for k, v in rows}


def chart_doc_types():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    vals, labels, cols = [9, 1], ["DRHP (9)", "RHP (1)"], [BLUE, ORANGE]
    w, _ = ax.pie(vals, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor=SURFACE, linewidth=2))
    ax.text(0, 0, "10\ndocs", ha="center", va="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.legend(w, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.set_aspect("equal")
    save(fig, "fig_doc_types.png")


# ---------- diagrams ----------

def box(ax, x, y, w, h, text, fc="white", ec=NAVY, tc=INK, fs=10, bold=False, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", wrap=True, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3, shrinkA=0, shrinkB=0))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def diagram_pipeline():
    fig, ax = canvas(7.4, 2.5)
    stages = [("Literature\nreview", "done"), ("Pipeline\nplanning", "done"), ("Data\ncollection", "done"), ("PDF\nparsing", "done"),
              ("Table\nextraction", "next"), ("RAG\nretrieval", "plan"), ("Valuation\n+ ML", "plan")]
    style = {"done": (BLUE, "white", BLUE), "next": (ORANGE, "white", ORANGE), "plan": ("white", INK2, MUTED)}
    w, h, gap, y = 0.9, 0.85, 0.135, 0.85
    for i, (t, s) in enumerate(stages):
        x = 0.08 + i * (w + gap)
        fc, tc, ec = style[s]
        box(ax, x, y, w, h, t, fc=fc, ec=ec, tc=tc, fs=9.5, bold=True)
        if i < len(stages) - 1:
            arrow(ax, x + w + 0.01, y + h / 2, x + w + gap - 0.01, y + h / 2)
    # stage brackets
    def bracket(x1, x2, label, color):
        ax.plot([x1, x1, x2, x2], [0.62, 0.52, 0.52, 0.62], color=color, lw=1.2)
        ax.text((x1 + x2) / 2, 0.3, label, ha="center", fontsize=9.5, color=color, fontweight="bold")
    bracket(0.08, 0.08 + w + gap + w, "Stage 1", INK2)
    bracket(0.08 + 2 * (w + gap), 0.08 + 3 * (w + gap) + w, "Stage 2 (this report)", BLUE)
    bracket(0.08 + 4 * (w + gap), 0.08 + 4 * (w + gap) + w, "Stage 3", ORANGE)
    ax.text(0.08, 2.15, "Four of seven pipeline stages are complete; table extraction is next", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.08, 1.88, "blue = done   ·   orange = next   ·   outline = planned", fontsize=9.5, color=INK2)
    save(fig, "fig_pipeline_status.png")


def diagram_collection():
    fig, ax = canvas(7.4, 2.75)
    steps = ["SEBI Public Issues\nlistings", "Filing page\n(found via web search)", "Embedded\nPDF link", "Download\n+ verify"]
    w, h, gap, y = 1.55, 0.8, 0.33, 1.1
    for i, t in enumerate(steps):
        x = 0.08 + i * (w + gap)
        box(ax, x, y, w, h, t, fc="#eef4fc" if i < 3 else BLUE, ec=BLUE, tc=INK if i < 3 else "white", fs=9.5, bold=i == 3)
        if i < 3:
            arrow(ax, x + w + 0.02, y + h / 2, x + w + gap - 0.02, y + h / 2)
    xv = 0.08 + 3 * (w + gap)
    checks = "PDF header  ·  page count 300–700  ·  cover names the right company and date"
    ax.text(xv + w, 0.82, checks, ha="right", fontsize=9, color=INK2)
    ax.text(0.08, 0.42, "Outputs:  data/raw/drhp/  (10 PDFs, 110 MB)    data/manifest.csv  (source URL, date, sector, pages)", fontsize=9.5, color=INK)
    ax.text(0.08, 2.42, "Each document was traced from SEBI's listing to a verified local PDF", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.08, 2.15, "SEBI's firewall blocks scripted search, so filing pages were located by web search instead", fontsize=9.5, color=INK2)
    save(fig, "fig_collection_flow.png")


def diagram_parser():
    fig, ax = canvas(7.4, 3.0)
    steps = [("1  Read table\nof contents", "titles + printed\npage numbers"),
             ("2  Map printed\n→ PDF pages", "most common\npage offset"),
             ("3  Tag every\npage", "section, chapter,\nstandard key"),
             ("4  Self-check", "title found on its\nstart page?")]
    w, h, gap, y = 1.55, 0.78, 0.33, 1.45
    for i, (t, sub) in enumerate(steps):
        x = 0.08 + i * (w + gap)
        box(ax, x, y, w, h, t, fc="#eef4fc", ec=BLUE, fs=9.5, bold=True)
        ax.text(x + w / 2, y - 0.3, sub, ha="center", va="center", fontsize=9, color=INK2, linespacing=1.2)
        if i < 3:
            arrow(ax, x + w + 0.02, y + h / 2, x + w + gap - 0.02, y + h / 2)
    # outputs
    bx = 0.08 + 2 * (w + gap)
    box(ax, bx, 0.12, 1.55, 0.5, "pages.jsonl", fc=BLUE, ec=BLUE, tc="white", fs=9.5, bold=True)
    box(ax, bx + w + gap, 0.12, 1.55, 0.5, "sections.json", fc=BLUE, ec=BLUE, tc="white", fs=9.5, bold=True)
    ax.text(bx - 0.12, 0.37, "Output per document:", ha="right", va="center", fontsize=9.5, color=INK)
    ax.text(0.08, 2.72, "The parser turns each contents page into a page-by-page section map", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.08, 2.45, "PDF bookmarks were missing in 7 of 10 files, so the printed table of contents is the source", fontsize=9.5, color=INK2)
    save(fig, "fig_parser_flow.png")


def diagram_ml():
    fig, ax = canvas(7.4, 3.5)
    feats = ["Financial ratios", "Offer details", "Issue P/E vs peers", "Market conditions", "Risk-factor text (opt.)"]
    fy0 = 0.35
    for i, f in enumerate(feats):
        y = fy0 + (len(feats) - 1 - i) * 0.5
        box(ax, 0.08, y, 1.75, 0.38, f, fc="#eef4fc", ec=BLUE, fs=9)
        arrow(ax, 1.85, y + 0.19, 2.55, 1.55, color=GRID if False else "#b9c6d8")
    box(ax, 2.6, 1.15, 1.7, 0.8, "Gradient boosting\n(XGBoost / LightGBM)", fc=BLUE, ec=BLUE, tc="white", fs=9.5, bold=True)
    ax.text(3.45, 0.88, "must beat ridge baseline", ha="center", fontsize=8.5, color=INK2)
    arrow(ax, 4.32, 1.55, 4.75, 1.55)
    box(ax, 4.8, 1.15, 1.15, 0.8, "Listing-day\nreturn", fc="white", ec=NAVY, fs=9.5, bold=True)
    arrow(ax, 5.97, 1.55, 6.3, 1.55)
    box(ax, 6.33, 1.15, 1.0, 0.8, "Price\nrange", fc=ORANGE, ec=ORANGE, tc="white", fs=9.5, bold=True)
    ax.text(5.38, 0.88, "explained per IPO with SHAP values", ha="center", fontsize=8.5, color=INK2)
    ax.text(0.08, 3.25, "Pre-listing features feed one model that predicts listing-day return", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.08, 2.98, "the LLM retrieves and cites evidence; it never produces the price", fontsize=9.5, color=INK2)
    save(fig, "fig_ml_design.png")


def diagram_split():
    fig, ax = canvas(7.4, 1.9)
    segs = [("Train: older IPOs", 4.3, BLUE), ("Validate", 1.3, MUTED), ("Test: 15–20 newest", 1.6, ORANGE)]
    x = 0.08
    for t, w, c in segs:
        ax.add_patch(FancyBboxPatch((x, 0.75), w - 0.04, 0.5, boxstyle="round,pad=0,rounding_size=0.06", fc=c, ec=c))
        ax.text(x + w / 2, 1.0, t, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        x += w
    arrow(ax, 0.08, 0.5, 7.25, 0.5)
    ax.text(0.08, 0.2, "older listings", fontsize=9, color=INK2)
    ax.text(7.25, 0.2, "listing date  →  most recent", fontsize=9, color=INK2, ha="right")
    ax.text(0.08, 1.62, "IPOs are split by listing date, so the model never learns from the future", fontsize=12.5, fontweight="bold", color=NAVY)
    save(fig, "fig_time_split.png")


def diagram_stage3():
    fig, ax = canvas(7.4, 2.75)
    steps = ["Locate statement\npages (chapter map)", "Extract tables\n(pdfplumber)", "Map to schema\n(units, FY, page)", "Cross-check\nidentities"]
    w, h, gap, y = 1.55, 0.8, 0.33, 1.15
    for i, t in enumerate(steps):
        x = 0.08 + i * (w + gap)
        box(ax, x, y, w, h, t, fc="#fdf0ea", ec=ORANGE, fs=9.5)
        if i < 3:
            arrow(ax, x + w + 0.02, y + h / 2, x + w + gap - 0.02, y + h / 2)
    xc = 0.08 + 3 * (w + gap)
    ax.annotate("", xy=(0.08 + 2 * (w + gap) + w / 2, y), xytext=(xc + w / 2, y),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2, connectionstyle="arc3,rad=-0.45", mutation_scale=12))
    ax.text(xc - 0.1, 0.42, "failed check → re-map or try another tool", ha="center", fontsize=9, color=ORANGE)
    ax.text(0.08, 2.42, "Stage 3 extracts the financial tables and checks them automatically", fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.08, 2.15, "assets = equity + liabilities; P&L revenue matches the summary table", fontsize=9.5, color=INK2)
    save(fig, "fig_stage3_flow.png")


if __name__ == "__main__":
    docs = load_docs()
    chart_pages(docs)
    chart_page_share(docs)
    fin = chart_parts_per_doc(docs)
    print("financial pages per doc:", fin)
    diagram_pipeline()
    diagram_collection()
    diagram_parser()
    diagram_ml()
    diagram_split()
    diagram_stage3()
