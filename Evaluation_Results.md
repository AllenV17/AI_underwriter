# Evaluation results

Reproduce with `python src/eval_pipeline.py extraction` and `python src/eval_pipeline.py retrieval` (needs the pilot PDFs in `data/raw/drhp/`; see `data/manifest.csv` for URLs).

## Financial extraction (hand-checked answer set)

`data/eval/extraction_answers.csv` holds 28 figures from 3 DRHPs (DOMS, Honasa, Tata Technologies), each read by hand from the raw text of the cited page. A figure counts as correct only if the value matches to 0.01 in the printed unit and the reported source page matches.

| Result | Value |
|---|---|
| Value accuracy | 28 / 28 (100%) |
| Value + page accuracy | 28 / 28 (100%) |

Caveats: the set is small, and the parser was tuned while fixing Honasa (header and EPS handling), so 100% on this set overstates accuracy on unseen layouts. Wider coverage on all 10 pilots: every one extracts revenue, PAT, total assets, total equity and operating cash flow for all periods with no failed accounting check; borrowings are not found for Honasa, Tata Technologies and Unicommerce, and capex for Unicommerce (the statements may not print them; not checked by hand).

## Retrieval (BM25 only, chapter level)

`data/eval/retrieval_questions.csv`: 10 questions, each tied to the chapter that should answer it, asked against every pilot document that has that chapter (100 pairs). A hit means a top-k chunk comes from the expected chapter.

| hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|
| 56.0% | 82.0% | 90.0% | 0.70 |

Weakest chapters: promoters (hit@5 70%), MD&A (50%). Dense retrieval (BGE-M3), rank fusion and the reranker are not evaluated here because they need the embedding models. This is a chapter-level check, not an answer-correctness benchmark.

## Peer table extraction

On the 10 pilots the extractor returns P/E figures for DOMS, Mankind, Bajaj Housing, JSW Infrastructure and Tata Technologies. Honasa and Unicommerce state there are no comparable listed peers. Hyundai, IREDA and Premier Energies yield none (the table is blank, left as [●], or uses a layout the parser does not read). Wrapped company names are sometimes mis-assembled, so results are shown as suggestions to verify against the cited page.

## Historical IPO table and listing-return model

`data/ipo_history.csv` has **129 mainboard IPOs** listed 2021-09-24 to 2025-12-30 (`data/ipo_outcomes.csv` has the sourced outcome columns). Rows come from Trendlyne's screener (25 latest listings per year for 2021 to 2025; the site's other pages are not reachable without scripting its JavaScript) plus the pilot rows in `data/manifest.csv`. Each row has issue price, first-trade (listing) price and total subscription. Every row that carries a source gain figure matches listing/issue − 1 to 0.2 points (125 of 125), which rules out copy errors in the prices; subscription is not independently checked. The sample is the end of each year, not a random draw, so it is not representative of all IPOs.

Pre-listing features (revenue growth, margins, ROE, leverage, peer P/E) need each company's DRHP and exist only for the 8 pilots, so **the pre-bidding model cannot be trained yet** (the trainer refuses: no feature has half coverage). A **post-bidding model** that uses only (log) subscription can be trained (`python src/train_listing_model.py --use-subscription`):

| Evaluation | Model | MAE of listing gain | Direction accuracy |
|---|---|---|---|
| Walk-forward, 5 blocks, 65 IPOs | Ridge on log subscription | 22.4% | 75% |
| | Gradient boosting | 24.9% | 72% |
| | Predict average gain | 28.1% | 71% |
| | Predict zero gain | 26.8% | 29% |
| Newest 19 IPOs held out | Ridge | 22.8% | 74% |
| | Predict average gain | 19.8% | 63% |
| | Predict zero gain | 13.5% | 37% |

Subscription carries real signal across the whole period (it beats both baselines in the walk-forward blocks, mostly on direction), but on the newest 19 listings (late 2025, mostly muted) it loses on error to both naive baselines. The code therefore marks the model **not validated** and the predictor returns no number. The stored interval (±22.4 points) is too wide to be useful. Next steps: extract features for the other ~120 companies from their DRHPs, add a market-conditions feature, and re-run.
