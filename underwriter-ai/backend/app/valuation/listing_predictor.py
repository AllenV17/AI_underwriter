"""Historical IPO listing-price prediction (Phase 6).

The model is trained on 100-150 historical mainboard IPOs with a time-aware split
(train on older listings, test on the most recent) and only pre-listing features
(ratios, issue size, fresh/OFS mix, issue P/E vs peers, market conditions; subscription
data only if the model is used after bidding closes). Training lives in `listing_model.py`
(`src/train_listing_model.py`). Until a trained, validated model artefact exists,
`predict` returns status "not_trained" and no number: a prediction is never fabricated.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.compliance import LISTING_NOTICE
from app.core.config import get_settings

ALLOWED_FEATURES = [
    "revenue_growth", "pat_margin", "roe", "debt_to_equity", "issue_size_inr", "fresh_issue_share",
    "issue_pe", "peer_median_pe", "nifty_return_30d", "sector", "subscription_total_x",
]
FORBIDDEN_FEATURES = ["listing_price", "listing_day_return", "post_listing_volume", "post_listing_news"]


@dataclass
class ListingPrediction:
    status: str  # "not_trained" | "ok" | "insufficient_inputs"
    message: str
    predicted_listing_gain: float | None = None
    price_range: tuple[float, float] | None = None
    model_version: str | None = None
    features_used: dict = field(default_factory=dict)
    notice: str = LISTING_NOTICE

    def to_dict(self) -> dict:
        return asdict(self)


class ListingPricePredictor:
    """Loads a trained model from data/models/listing_model.joblib when one exists."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or get_settings().data_dir / "models" / "listing_model.joblib"

    @property
    def trained(self) -> bool:
        return self.model_path.exists()

    def predict(self, features: dict, issue_price: float | None = None) -> ListingPrediction:
        bad = [f for f in features if f in FORBIDDEN_FEATURES]
        if bad:
            raise ValueError(f"Post-listing features are not allowed (data leakage): {bad}")
        if not self.trained:
            return ListingPrediction(
                status="not_trained",
                message="Listing-price prediction is not available: no model has been trained and validated yet "
                        "(planned: gradient boosting on 100-150 historical IPOs with a time-aware test set).",
                features_used=features)
        import json

        import joblib
        import pandas as pd

        from app.valuation.listing_model import _features

        art = joblib.load(self.model_path)
        meta_path = self.model_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        needed = [c for c in art["columns"] if not c.startswith("sector_")]
        have = [c for c in needed if features.get(c) is not None]
        if len(have) < max(1, (len(needed) + 1) // 2):
            return ListingPrediction(
                status="insufficient_inputs", features_used=features,
                message=f"Only {len(have)} of the model's {len(needed)} numeric features are available "
                        f"(missing: {', '.join(c for c in needed if c not in have)}); at least half are required.")
        x = _features(pd.DataFrame([features]), art["columns"])
        gain = float(art["model"].predict(x)[0])
        half = float(meta.get("interval_halfwidth", 0.0))
        rng = (issue_price * (1 + gain - half), issue_price * (1 + gain + half)) if issue_price else None
        test = meta.get("metrics", {}).get(meta.get("chosen", ""), {})
        msg = (f"Model estimate of the listing-day return: {gain:+.1%} (+/- {half:.1%}, the validation mean absolute error). "
               f"Held-out test: MAE {test.get('mae', float('nan')):.1%}, direction accuracy {test.get('direction_accuracy', float('nan')):.0%}.")
        return ListingPrediction(status="ok", message=msg, predicted_listing_gain=gain, price_range=rng,
                                 model_version=f"{meta.get('chosen')}@{meta.get('trained_on_through')}", features_used=features)
