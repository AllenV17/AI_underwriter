"""Training and evaluation of the listing-day return model.

Target: listing_gain = (listing price - issue price) / issue price. Rows are sorted by
listing date and split in time (oldest train, next validation, newest test) so the model
never learns from IPOs that listed after the ones it is tested on. Gradient boosting is
compared with a ridge baseline and two naive baselines on the same test rows; the
artefact stored is the one with the lowest validation error, so the simpler model wins
if boosting does not beat it. Training refuses to run below MIN_ROWS: with a handful of
IPOs any score is noise.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.valuation.listing_predictor import ALLOWED_FEATURES, FORBIDDEN_FEATURES

MIN_ROWS = 40
NUMERIC = [f for f in ALLOWED_FEATURES if f != "sector"]
REQUIRED_COLUMNS = ["company", "listing_date", "issue_price", "listing_price"]


class InsufficientDataError(ValueError):
    pass


@dataclass
class ModelReport:
    n_rows: int
    n_train: int
    n_val: int
    n_test: int
    features: list[str]
    chosen: str
    metrics: dict = field(default_factory=dict)  # model -> {mae, direction_accuracy, range_hit_rate}
    interval_halfwidth: float = 0.0  # +/- listing gain used to build the price range (validation MAE)
    trained_on_through: str = ""
    uses_subscription: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")
    leak = [c for c in df.columns if c in FORBIDDEN_FEATURES]
    if leak:  # outcome columns are allowed only as the target source, never as inputs
        df = df.drop(columns=[c for c in leak if c != "listing_price"], errors="ignore")
    df["listing_date"] = pd.to_datetime(df["listing_date"])
    df = df.dropna(subset=["issue_price", "listing_price", "listing_date"]).sort_values("listing_date").reset_index(drop=True)
    df["listing_gain"] = df["listing_price"] / df["issue_price"] - 1.0
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.15, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = max(int(round(n * test_frac)), 1)
    n_val = max(int(round(n * val_frac)), 1)
    return df.iloc[: n - n_test - n_val], df.iloc[n - n_test - n_val: n - n_test], df.iloc[n - n_test:]


def _features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    x = df[[c for c in NUMERIC if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    if "sector" in df.columns:
        x = pd.concat([x, pd.get_dummies(df["sector"].fillna("unknown"), prefix="sector", dtype=float)], axis=1)
    if columns is not None:
        x = x.reindex(columns=columns, fill_value=0.0)
    return x


def _metrics(y: np.ndarray, pred: np.ndarray, half: float) -> dict:
    return {"mae": float(np.mean(np.abs(y - pred))),
            "direction_accuracy": float(np.mean((y > 0) == (pred > 0))),
            "range_hit_rate": float(np.mean(np.abs(y - pred) <= half))}


def _models():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "gradient_boosting": make_pipeline(SimpleImputer(strategy="median"),
                                           GradientBoostingRegressor(max_depth=2, n_estimators=150, learning_rate=0.05,
                                                                     subsample=0.8, random_state=0)),
    }


def train(df: pd.DataFrame, out_dir: Path, min_rows: int = MIN_ROWS, use_subscription: bool = False) -> ModelReport:
    """Fit, compare on a time-ordered split, refit the winner on train+val and save it.

    Subscription levels are only known once bidding closes, so they are excluded unless the
    model is meant to be used after bidding (`use_subscription=True`).
    """
    import joblib

    if len(df) < min_rows:
        raise InsufficientDataError(f"{len(df)} IPOs with prices; at least {min_rows} are needed (plan: 100-150). "
                                    "Add rows to data/ipo_history.csv rather than training on too little data.")
    tr, va, te = time_split(df)
    cols = [c for c in _features(df).columns if use_subscription or c != "subscription_total_x"]
    xtr, xva, xte = (_features(d, cols) for d in (tr, va, te))
    ytr, yva, yte = (d["listing_gain"].to_numpy() for d in (tr, va, te))
    fitted, val_mae = {}, {}
    for name, model in _models().items():
        model.fit(xtr, ytr)
        fitted[name] = model
        val_mae[name] = float(np.mean(np.abs(yva - model.predict(xva))))
    chosen = min(val_mae, key=val_mae.get)
    half = val_mae[chosen]

    metrics = {name: _metrics(yte, m.predict(xte), val_mae[name]) for name, m in fitted.items()}
    metrics["baseline_train_mean"] = _metrics(yte, np.full_like(yte, ytr.mean()), val_mae[chosen])
    metrics["baseline_zero_gain"] = _metrics(yte, np.zeros_like(yte), val_mae[chosen])

    final = _models()[chosen]
    trva = pd.concat([tr, va])
    final.fit(_features(trva, cols), trva["listing_gain"].to_numpy())
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ModelReport(len(df), len(tr), len(va), len(te), cols, chosen, metrics, half,
                         str(df["listing_date"].max().date()))
    report.uses_subscription = use_subscription
    joblib.dump({"model": final, "columns": cols}, out_dir / "listing_model.joblib")
    (out_dir / "listing_model.json").write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
    return report
