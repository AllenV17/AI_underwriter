"""Regression tests for the findings of the PR review."""

import numpy as np
import pandas as pd

from app.financial_engine.peer_extractor import PeerRow, PeerTable, _LEAK
from app.financial_engine.statement_parser import _merge_split_columns
from app.valuation.listing_features import median_peer_pe
from app.valuation.listing_model import train
from app.valuation.listing_predictor import ListingPricePredictor
from tests.test_listing_model import _csv, _history


def test_median_even_count():
    assert median_peer_pe([{"pe": v} for v in (14.27, 24.99, 34.05, 37.01, 48.46, 58.48)]) == (34.05 + 37.01) / 2
    assert median_peer_pe([{"pe": None}]) is None


def test_missing_features_use_training_median_not_zero(tmp_path):
    train(_csv(tmp_path, _history()), tmp_path / "m")
    p = ListingPricePredictor(tmp_path / "m" / "listing_model.joblib")
    full = {"revenue_growth": 30, "pat_margin": 10, "roe": 15, "debt_to_equity": 1, "peer_median_pe": 35, "sector": "a"}
    part = {k: v for k, v in full.items() if k != "roe"}
    a, b = p.predict(full, 100), p.predict(part, 100)
    assert b.status == "ok" and abs(a.predicted_listing_gain - b.predicted_listing_gain) < 0.05  # roe=0 would shift it more
    assert p.predict({"roe": 10}, 100).status == "insufficient_inputs"  # fewer than half the features


def test_leak_does_not_match_real_peer_names():
    for name in ("Aptus Value Housing Finance", "The Tata Power Company Limited", "Godrej Consumer Products Company"):
        assert not _LEAK.search(name)
    for leak in ("income Value EPS (₹) EPS (%)", "P/ E (₹) from operations per equity shar", "company Value from for EPS"):
        assert _LEAK.search(leak)


def test_merge_only_sparse_neighbour():
    # Line items at x=270 (40 rows) and indented subtotals at x=305 (10 rows, never on the same row): merge.
    rows = [[270.0] for _ in range(40)] + [[305.0] for _ in range(10)]
    cl = [[270.0] * 40, [305.0] * 10]
    assert len(_merge_split_columns(cl, rows)) == 1
    # Two genuine period columns 70pt apart with equal row counts that never share a row: keep both.
    rows = [[270.0] for _ in range(20)] + [[340.0] for _ in range(20)]
    cl = [[270.0] * 20, [340.0] * 20]
    assert len(_merge_split_columns(cl, rows)) == 2
    # Sparse neighbour that does share rows with the big column: keep both.
    rows = [[270.0, 305.0] for _ in range(10)] + [[270.0] for _ in range(30)]
    assert len(_merge_split_columns([[270.0] * 40, [305.0] * 10], rows)) == 2


def test_peer_dict_carries_page_and_flag():
    t = PeerTable(peers=[PeerRow(name="X Limited", page=7, values={"pe": 20.0}), PeerRow(name="", page=7, values={"pe": 10.0}, name_reliable=False)])
    out = t.as_peer_companies()
    assert out[0]["page"] == 7 and out[1]["name_verified"] is False and "p.7" in out[1]["name"]
