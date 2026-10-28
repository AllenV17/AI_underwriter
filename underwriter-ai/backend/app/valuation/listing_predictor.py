"""Historical IPO listing-price prediction: interface only (Phase 6).

The model will be trained on 100-150 historical mainboard IPOs with a time-aware split
(train on older listings, test on the most recent) and only pre-listing features
(ratios, issue size, fresh/OFS mix, issue P/E vs peers, market conditions; subscription
data only if the model is used after bidding closes). Until a trained, validated model
artefact exists, `predict` returns status "not_trained" and no number: a prediction is
never fabricated.
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
    """Loads a trained model from data/models/listing_model.json when one exists."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or get_settings().data_dir / "models" / "listing_model.json"

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
        raise NotImplementedError("Model loading is implemented in Phase 6 once a validated model exists.")
