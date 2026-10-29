"""Train the listing-day return model from data/ipo_history.csv.

Usage:  python src/train_listing_model.py [--history data/ipo_history.csv] [--out underwriter-ai/data/models] [--use-subscription]

Refuses to train below 40 priced IPOs. Prints MAE, direction accuracy and range hit rate
on the newest IPOs (held out by listing date) for ridge, gradient boosting and two naive baselines.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "underwriter-ai" / "backend"))

from app.valuation.listing_model import InsufficientDataError, load_history, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=str(ROOT / "data" / "ipo_history.csv"))
    ap.add_argument("--out", default=str(ROOT / "underwriter-ai" / "data" / "models"))
    ap.add_argument("--use-subscription", action="store_true", help="include subscription levels (model used after bidding closes)")
    a = ap.parse_args()
    try:
        report = train(load_history(Path(a.history)), Path(a.out), use_subscription=a.use_subscription)
    except InsufficientDataError as e:
        sys.exit(f"Not trained: {e}")
    print(json.dumps(report.to_dict(), indent=1))


if __name__ == "__main__":
    main()
