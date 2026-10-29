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

## Listing-return model

Not trained, and no accuracy is claimed. `data/ipo_history.csv` has 10 pilot IPOs, 7 with a usable listing price, and the trainer refuses to run below 40. The pipeline (time-ordered split, ridge vs gradient boosting vs two naive baselines, MAE, direction accuracy, range hit rate) is tested on synthetic data only. Building the 100–150 row history is still open.
