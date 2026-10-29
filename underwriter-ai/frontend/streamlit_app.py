"""UnderwriterAI dashboard (Streamlit). Run: streamlit run frontend/streamlit_app.py"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import API_URL, APIError, get, get_bytes, post

st.set_page_config(page_title="UnderwriterAI", page_icon="📑", layout="wide")

NAVY, BLUE, ORANGE, AQUA, GREY = "#1E2761", "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
st.markdown(f"""
<style>
h1, h2, h3 {{ color: {NAVY}; }}
.disclaimer {{ background:#EEF2FB; border-left: 0; border-radius:8px; padding:10px 14px; font-size:0.82rem; color:#33384d; }}
.notice {{ background:#fdf0ea; border-radius:8px; padding:8px 12px; font-size:0.85rem; color:#5a2e17; }}
.metric-src {{ font-size:0.75rem; color:#6b6f80; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def compliance() -> dict:
    try:
        return get("/api/compliance/disclaimer")
    except APIError:
        return {"disclaimer": "Informational and educational use only. Not investment advice.", "valuation_notice": ""}


def disclaimer_box() -> None:
    st.markdown(f"<div class='disclaimer'><b>Disclaimer.</b> {compliance()['disclaimer']}</div>", unsafe_allow_html=True)


def notice_box(text: str | None = None) -> None:
    st.markdown(f"<div class='notice'>{text or compliance()['valuation_notice']}</div>", unsafe_allow_html=True)


def crore(v: float | None) -> str:
    return "—" if v is None else f"₹{v / 1e7:,.1f} cr"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("UnderwriterAI")
st.sidebar.caption("AI-assisted DRHP analysis and IPO valuation")
try:
    health = get("/api/health", timeout=10)
    st.sidebar.success(f"Backend OK · LLM {'configured' if health['llm_configured'] else 'not configured'}")
except APIError as e:
    health = None
    st.sidebar.error(str(e))

docs = []
if health:
    try:
        docs = get("/api/documents")
    except APIError as e:
        st.sidebar.error(str(e))
labels = {d["id"]: f"{d['company_name'] or d['filename']} · {d['status']}" for d in docs}
doc_id = st.sidebar.selectbox("Document", list(labels), format_func=lambda i: labels[i]) if docs else None
page = st.sidebar.radio("Go to", ["Home", "Document Analysis", "Chat", "Financial Dashboard", "Valuation", "Final Report"])
st.sidebar.markdown("---")
st.sidebar.caption("Not a SEBI-registered investment adviser or research analyst. Outputs are analysis, not recommendations.")

doc = next((d for d in docs if d["id"] == doc_id), None)


def require_ready() -> bool:
    if not doc:
        st.info("Upload a DRHP on the Home page first.")
        return False
    if doc["status"] != "ready":
        st.warning(f"Document status: {doc['status']} ({doc['stage']}). Wait for processing to finish on the Home page.")
        return False
    return True


# ------------------------------------------------------------------ pages
if page == "Home":
    st.title("UnderwriterAI")
    st.write("Upload a Draft Red Herring Prospectus (DRHP). The system maps its sections, extracts the restated financial "
             "statements, indexes every page for cited question answering, and supports ratio analysis, valuation and a "
             "full underwriting report.")
    disclaimer_box()
    st.subheader("Upload a DRHP")
    up = st.file_uploader("DRHP / RHP PDF", type=["pdf"])
    if up and st.button("Upload and process", type="primary"):
        try:
            d = post("/api/documents/upload", files={"file": (up.name, up.getvalue(), "application/pdf")})
            st.session_state["watch"] = d["id"]
            st.success(f"Uploaded {d['filename']} ({d['n_pages']} pages). Processing started.")
        except APIError as e:
            st.error(str(e))
    watch = st.session_state.get("watch") or (doc["id"] if doc and doc["status"] == "processing" else None)
    if watch:
        st.subheader("Processing status")
        bar, msg = st.progress(0.0), st.empty()
        for _ in range(3600):
            d = get(f"/api/documents/{watch}")
            bar.progress(min(1.0, d["progress"]))
            msg.write(f"**{d['stage']}** · {d['progress']:.0%}")
            if d["status"] in ("ready", "failed"):
                break
            time.sleep(2)
        if d["status"] == "ready":
            st.success(f"Ready: {d['company_name'] or d['filename']}")
            for w in d.get("warnings") or []:
                st.caption(f"⚠ {w}")
            st.session_state.pop("watch", None)
        else:
            st.error(f"Processing failed: {d['error']}")
    if docs:
        st.subheader("Documents")
        st.dataframe(pd.DataFrame([{k: d[k] for k in ("company_name", "doc_type", "n_pages", "status", "stage", "indexed",
                                                     "financials_extracted", "filename")} for d in docs]),
                     hide_index=True, use_container_width=True)

elif page == "Document Analysis":
    st.title("Document Analysis")
    if require_ready():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Company", doc["company_name"] or "—")
        c2.metric("Document", doc["doc_type"] or "—")
        c3.metric("Pages", doc["n_pages"])
        c4.metric("Indexed for search", "Yes" if doc["indexed"] else "No")
        secs = get(f"/api/documents/{doc_id}/sections")
        st.subheader("Sections (from the table of contents)")
        df = pd.DataFrame(secs)
        if not df.empty:
            df["pages"] = df["pdf_start"].astype(str) + "–" + df["pdf_end"].astype(str)
            st.dataframe(df[["section", "title", "key", "pages", "title_verified"]], hide_index=True, use_container_width=True)
        if doc.get("warnings"):
            with st.expander(f"Processing warnings ({len(doc['warnings'])})"):
                for w in doc["warnings"]:
                    st.write(f"- {w}")
        try:
            fin = get(f"/api/documents/{doc_id}/financials")
            st.subheader(f"Extracted financial data ({fin['basis']}, as printed in ₹ {fin['unit']})")
            vals = pd.DataFrame(fin["values"])
            pivot = vals.pivot_table(index="label", columns="period", values="value_original", aggfunc="first")
            st.dataframe(pivot[[p for p in fin["periods"] if p in pivot.columns]].style.format("{:,.2f}", na_rep="—"), use_container_width=True)
            pages = vals.groupby("label")["source_page"].first()
            st.caption("Source pages: " + "; ".join(f"{k}: p.{int(v)}" for k, v in pages.dropna().items()))
            st.subheader("Validation checks")
            st.dataframe(pd.DataFrame(fin["checks"]), hide_index=True, use_container_width=True)
        except APIError as e:
            st.warning(f"Financial data not available: {e}")

elif page == "Chat":
    st.title("Ask the DRHP")
    if require_ready():
        st.caption("Answers cite DRHP pages. Figures come from deterministic calculations; the assistant does not give buy/sell/subscribe advice.")
        hist = st.session_state.setdefault(f"chat-{doc_id}", [])
        for m in hist:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("evidence"):
                    with st.expander(f"Retrieved evidence ({len(m['evidence'])})"):
                        for e in m["evidence"]:
                            st.markdown(f"**[{e['ref']}] {e['section']} · page {e['page_number']}** "
                                        f"<span class='metric-src'>(dense #{e['dense_rank']}, BM25 #{e['sparse_rank']}, rerank {e['rerank_score']})</span>",
                                        unsafe_allow_html=True)
                            st.text(e["text"][:900])
        q = st.chat_input("e.g. How concentrated is revenue among top customers?")
        if q:
            hist.append({"role": "user", "content": q})
            with st.spinner("Retrieving evidence and analysing..."):
                try:
                    r = post(f"/api/documents/{doc_id}/chat", json={"question": q})
                    content = r["answer"]
                    if r.get("disclaimer"):
                        content += f"\n\n> {r['disclaimer']}"
                    if r["warnings"]:
                        content += "\n\n" + "\n".join(f"⚠ {w}" for w in r["warnings"])
                    content += f"\n\n<span class='metric-src'>Intent: {r['intent']} · tools: {', '.join(s['tool'] for s in r['steps'])}</span>"
                    hist.append({"role": "assistant", "content": content, "evidence": r["evidence"]})
                except APIError as e:
                    hist.append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()

elif page == "Financial Dashboard":
    st.title("Financial Dashboard")
    if require_ready():
        try:
            fin = get(f"/api/documents/{doc_id}/financials")
            rat = get(f"/api/documents/{doc_id}/ratios")
        except APIError as e:
            st.warning(str(e))
            st.stop()
        vals = pd.DataFrame(fin["values"])
        periods = [p for p in fin["periods"]][::-1]  # oldest -> newest
        fy = [p for p in periods if p.startswith("FY")]

        def series(field):
            d = vals[vals.field == field].set_index("period")["value_inr"]
            return [d.get(p) for p in fy]

        st.caption(f"Basis: {fin['basis']} · interim periods excluded from trends (not annualised) · values in ₹ crore")
        latest = fy[-1] if fy else None
        if latest:
            c = st.columns(4)
            for col, (f, name) in zip(c, [("revenue_from_operations", "Revenue"), ("ebitda", "EBITDA (derived)"),
                                          ("profit_after_tax", "PAT"), ("free_cash_flow", "Free cash flow")]):
                s = series(f)
                col.metric(f"{name} {latest}", crore(s[-1]),
                           None if len(s) < 2 or not s[-2] or s[-1] is None else f"{(s[-1] / s[-2] - 1):+.1%} YoY")
        fig = go.Figure()
        for f, name, color in [("revenue_from_operations", "Revenue", BLUE), ("ebitda", "EBITDA", AQUA), ("profit_after_tax", "PAT", ORANGE)]:
            s = series(f)
            fig.add_bar(x=fy, y=[None if v is None else v / 1e7 for v in s], name=name, marker_color=color,
                        hovertemplate="%{x}: ₹%{y:,.1f} cr<extra>" + name + "</extra>")
        fig.update_layout(barmode="group", title="Revenue, EBITDA and PAT (₹ crore)", height=380, plot_bgcolor="white",
                          legend=dict(orientation="h", y=1.1), margin=dict(t=60, b=30))
        fig.update_yaxes(gridcolor="#e4e3df")
        st.plotly_chart(fig, use_container_width=True)

        cf = go.Figure()
        for f, name, color in [("operating_cash_flow", "Operating cash flow", BLUE), ("capex", "Capex (purchase of PPE)", GREY),
                               ("free_cash_flow", "Free cash flow", AQUA)]:
            s = series(f)
            cf.add_bar(x=fy, y=[None if v is None else v / 1e7 for v in s], name=name, marker_color=color)
        cf.update_layout(barmode="group", title="Cash flow analysis (₹ crore)", height=340, plot_bgcolor="white",
                         legend=dict(orientation="h", y=1.12), margin=dict(t=60, b=30))
        st.plotly_chart(cf, use_container_width=True)

        st.subheader("Financial ratios")
        if rat["meta"]["financial_company"]:
            st.info("Detected a lender (finance costs are a major share of revenue): EBITDA, coverage and current-ratio metrics are marked not applicable.")
        r = pd.DataFrame(rat["ratios"])
        r = r[r.unit.isin(["%", "x"])]
        cat = st.selectbox("Category", ["Profitability", "Liquidity", "Leverage", "Efficiency", "Cash flow", "Valuation"])
        sub = r[r.category == cat]
        table = sub.pivot_table(index="name", columns="period", values="value", aggfunc="first")
        st.dataframe(table[[p for p in fin["periods"] if p in table.columns]].style.format("{:,.2f}", na_rep="n/a"), use_container_width=True)
        with st.expander("Formulas, inputs and interpretation"):
            for _, row in sub.iterrows():
                if row["available"]:
                    st.markdown(f"**{row['name']} ({row['period']})** = {row['value']:,.2f}{row['unit']} · `{row['formula']}` · {row['interpretation']}")
                else:
                    st.markdown(f"**{row['name']} ({row['period']})**: {row['reason']}")

elif page == "Valuation":
    st.title("Valuation")
    if require_ready():
        notice_box()
        st.subheader("Comparable company valuation")
        st.caption("Enter listed peers and their multiples from a reliable, dated source (e.g. the DRHP's Basis for Offer Price "
                   "peer table, or exchange data). Peer figures are never generated automatically.")
        default = pd.DataFrame([{"name": "", "pe": None, "ev_ebitda": None, "ps": None, "source": ""}] * 3)
        if st.button("Load peer table from the DRHP"):
            try:
                found = get(f"/api/documents/{doc_id}/peers")
                rows = [{"name": p["name"], "pe": p.get("pe"), "ev_ebitda": None, "ps": None, "source": p["source"]}
                        for p in found.get("suggested", []) if p.get("pe")]
                if rows:
                    st.session_state["peers_df"] = pd.DataFrame(rows)
                    st.warning("Loaded from the DRHP. Company names can be mis-read when they wrap over lines: check each row against the cited page before running.")
                else:
                    st.info(found.get("note") or "No peer P/E figures were found in the DRHP.")
            except APIError as e:
                st.error(str(e))
        peers_df = st.data_editor(st.session_state.get("peers_df", default), num_rows="dynamic", use_container_width=True,
                                  column_config={"pe": st.column_config.NumberColumn("P/E (x)"),
                                                 "ev_ebitda": st.column_config.NumberColumn("EV/EBITDA (x)"),
                                                 "ps": st.column_config.NumberColumn("P/S (x)"),
                                                 "source": st.column_config.TextColumn("Source and date")})
        st.session_state["peers_df"] = peers_df
        shares = st.number_input("Post-issue shares outstanding (from the DRHP capital structure)", min_value=0.0, value=0.0, step=1e6, format="%.0f")
        if st.button("Run comparable valuation", type="primary"):
            peers = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in peers_df.to_dict("records") if row.get("name")]
            try:
                res = post(f"/api/documents/{doc_id}/valuation/comparable",
                           json={"peers": peers, "shares_outstanding": shares or None})
                st.session_state["comps"] = res
            except APIError as e:
                st.error(str(e))
        res = st.session_state.get("comps")
        if res:
            st.write(f"Target period: **{res['target']['period']}** · inputs: {res['target']['sources']}")
            rows = [{"Multiple": r["multiple"], "Low": r["low"], "Median": r["median"], "High": r["high"],
                     "Implied price (mid)": r["implied_price_mid"], "Implied equity (mid, ₹ cr)": None if r["implied_equity_mid"] is None else r["implied_equity_mid"] / 1e7,
                     "Note": r["note"]} for r in res["results"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if res.get("price_range"):
                lo, mid, hi = res["price_range"]
                st.metric("Estimated value per share (model estimate)", f"₹{mid:,.0f}", f"range ₹{lo:,.0f} – ₹{hi:,.0f}", delta_color="off")
            for w in res["warnings"]:
                st.caption(f"⚠ {w}")
            with st.expander("Assumptions"):
                for a in res["assumptions"]:
                    st.write(f"- {a}")
            notice_box(res["notice"])

        st.subheader("DCF valuation")
        try:
            dflt = get(f"/api/documents/{doc_id}/valuation/dcf/defaults")
        except APIError:
            dflt = {}
        st.caption(f"Base year {dflt.get('base_period', '—')} from the DRHP: revenue {crore(dflt.get('base_revenue'))}, "
                   f"net debt {crore(dflt.get('net_debt'))}. All other inputs below are your assumptions.")
        c1, c2, c3 = st.columns(3)
        growth = c1.text_input("Revenue growth by year (%, comma-separated)", "15, 13, 11, 10, 9")
        margin = c1.number_input("EBITDA margin (%)", value=round(100 * dflt.get("ebitda_margin_historical", 0.15), 1))
        tax = c1.number_input("Tax rate (%)", value=25.17)
        da = c2.number_input("D&A (% of revenue)", value=round(100 * dflt.get("da_pct_revenue_historical", 0.03), 2))
        capex = c2.number_input("Capex (% of revenue)", value=round(100 * dflt.get("capex_pct_revenue_historical", 0.04), 2))
        nwc = c2.number_input("Working capital (% of incremental revenue)", value=10.0)
        wacc = c3.number_input("WACC (%)", value=12.0)
        tg = c3.number_input("Terminal growth (%)", value=5.0)
        dshares = c3.number_input("Shares outstanding", min_value=0.0, value=shares, step=1e6, format="%.0f")
        if st.button("Run DCF") and dflt.get("base_revenue"):
            a = {"base_revenue": dflt["base_revenue"], "base_period": dflt.get("base_period", ""),
                 "revenue_growth": [float(x) / 100 for x in growth.split(",") if x.strip()],
                 "ebitda_margin": margin / 100, "tax_rate": tax / 100, "da_pct_revenue": da / 100, "capex_pct_revenue": capex / 100,
                 "nwc_pct_incremental_revenue": nwc / 100, "wacc": wacc / 100, "terminal_growth": tg / 100,
                 "net_debt": dflt.get("net_debt") or 0.0, "shares_outstanding": dshares or None,
                 "input_sources": {**dflt.get("input_sources", {}), "growth/margin/wacc/terminal": "user input"}}
            try:
                st.session_state["dcf"] = post(f"/api/documents/{doc_id}/valuation/dcf", json={"assumptions": a})
                st.session_state["dcf_assumptions"] = a
            except APIError as e:
                st.error(str(e))
        d = st.session_state.get("dcf")
        if d:
            c1, c2, c3 = st.columns(3)
            c1.metric("Enterprise value", crore(d["enterprise_value"]))
            c2.metric("Equity value", crore(d["equity_value"]))
            c3.metric("Value per share (model estimate)", "—" if d["price_per_share"] is None else f"₹{d['price_per_share']:,.0f}")
            st.caption(f"Terminal value is {d['terminal_share_of_ev']:.0%} of EV.")
            yrs = pd.DataFrame(d["years"])
            st.dataframe((yrs.set_index("year")[["revenue", "ebitda", "nopat", "capex", "delta_nwc", "fcff", "pv_fcff"]] / 1e7)
                         .style.format("{:,.1f}"), use_container_width=True)
            s = d["sensitivity_wacc_g"]
            grid = pd.DataFrame(s["values"], index=[f"WACC {w:.1%}" for w in s["wacc"]], columns=[f"g {g:.1%}" for g in s["terminal_growth"]])
            if s["metric"] == "equity_value":
                grid = grid / 1e7
            st.write(f"Sensitivity: {'price per share (₹)' if s['metric'] == 'price_per_share' else 'equity value (₹ crore)'}")
            st.dataframe(grid.style.format("{:,.0f}", na_rep="—"), use_container_width=True)
            for w in d["warnings"]:
                st.caption(f"⚠ {w}")
            notice_box(d["notice"])

elif page == "Final Report":
    st.title("Underwriting Report")
    if require_ready():
        disclaimer_box()
        st.write("The report combines cited narrative sections, the extracted financials and ratios, the risk analysis and any "
                 "valuation you ran on the Valuation page.")
        issue_price = st.number_input("Issue price / upper price band for reference (optional, ₹)", min_value=0.0, value=0.0)
        agree = st.checkbox("I understand this report is AI-generated analysis for informational purposes, not investment advice.")
        if st.button("Generate report", type="primary", disabled=not agree):
            peers_df = st.session_state.get("peers_df")
            peers = [] if peers_df is None else [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in peers_df.to_dict("records") if r.get("name")]
            body = {"peers": peers or None, "issue_price": issue_price or None,
                    "dcf_assumptions": st.session_state.get("dcf_assumptions"),
                    "shares_outstanding": (st.session_state.get("dcf_assumptions") or {}).get("shares_outstanding")}
            with st.spinner("Generating the report (retrieval + LLM per section)..."):
                try:
                    st.session_state["report"] = post(f"/api/documents/{doc_id}/report", json=body)
                except APIError as e:
                    st.error(str(e))
        rep = st.session_state.get("report")
        if rep:
            st.download_button("Download PDF", get_bytes(rep["pdf_url"]), file_name=f"underwriting_report_{doc['company_name'] or 'drhp'}.pdf",
                               mime="application/pdf")
            for i, s in enumerate(rep["report"]["sections"], 1):
                st.subheader(f"{i}. {s['title']}")
                if s.get("kind") == "risks" and s.get("risks"):
                    st.dataframe(pd.DataFrame([{"Severity": r.get("severity"), "Risk": r.get("title"), "Category": r.get("category"),
                                                "Description": r.get("description"),
                                                "Pages": ", ".join(str(x["page_number"]) for x in r.get("sources", []))} for r in s["risks"]]),
                                 hide_index=True, use_container_width=True)
                elif s.get("kind") in ("list", "sources"):
                    for x in s.get("items", []):
                        st.write(f"- {x}")
                else:
                    st.markdown(s.get("text", ""))
                if s.get("notice"):
                    notice_box(s["notice"])

st.sidebar.caption(f"API: {API_URL}")
