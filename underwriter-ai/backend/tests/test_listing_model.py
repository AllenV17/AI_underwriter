"""Listing-return model: training guards, time-ordered evaluation, prediction (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from app.valuation.listing_model import InsufficientDataError, load_history, time_split, train
from app.valuation.listing_predictor import ListingPricePredictor


def _history(n=80, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gain = rng.normal(0.1, 0.1, n)
    df = pd.DataFrame({
        "company": [f"Co{i}" for i in range(n)],
        "listing_date": pd.date_range("2022-01-01", periods=n, freq="9D"),
        "issue_price": 100.0,
        "revenue_growth": rng.normal(30, 10, n),
        "pat_margin": rng.normal(10, 4, n),
        "roe": rng.normal(15, 5, n),
        "debt_to_equity": rng.uniform(0, 2, n),
        "peer_median_pe": rng.uniform(15, 60, n),
        "sector": rng.choice(["a", "b"], n),
    })
    df["listing_price"] = 100 * (1 + 0.01 * (df["revenue_growth"] - 30) + 0.02 * (df["pat_margin"] - 10) + gain * 0.2)
    return df


def _csv(tmp_path, df):
    p = tmp_path / "h.csv"
    df.to_csv(p, index=False)
    return load_history(p)


def test_refuses_small_dataset(tmp_path):
    with pytest.raises(InsufficientDataError):
        train(_csv(tmp_path, _history(10)), tmp_path / "m")


def test_split_is_chronological(tmp_path):
    tr, va, te = time_split(_csv(tmp_path, _history()))
    assert tr["listing_date"].max() < va["listing_date"].min() <= va["listing_date"].max() < te["listing_date"].min()


def test_train_beats_baselines_and_predicts(tmp_path):
    rep = train(_csv(tmp_path, _history()), tmp_path / "m")
    assert "subscription_total_x" not in rep.features and not rep.uses_subscription
    assert rep.metrics[rep.chosen]["mae"] < rep.metrics["baseline_zero_gain"]["mae"]
    pred = ListingPricePredictor(tmp_path / "m" / "listing_model.joblib").predict(
        {"revenue_growth": 60, "pat_margin": 20, "roe": 15, "debt_to_equity": 1, "peer_median_pe": 30, "sector": "a"}, issue_price=100)
    assert pred.status == "ok" and pred.price_range[0] < pred.price_range[1]
    assert pred.predicted_listing_gain > 0.1  # strong growth and margin push the estimate up


def test_leakage_rejected(tmp_path):
    train(_csv(tmp_path, _history()), tmp_path / "m")
    with pytest.raises(ValueError):
        ListingPricePredictor(tmp_path / "m" / "listing_model.joblib").predict({"listing_day_return": 0.3})


def test_untrained_returns_no_number(tmp_path):
    pred = ListingPricePredictor(tmp_path / "none.joblib").predict({"roe": 10})
    assert pred.status == "not_trained" and pred.predicted_listing_gain is None
