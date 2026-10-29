"""Build data/ipo_outcomes.csv (and validate it) from the raw screener rows, then data/ipo_history.csv.

Usage (repo root):  python src/build_ipo_history.py [--features]

Outcome columns (issue price, listing price = first-trade price on the exchange, total subscription) come from
data/sources/trendlyne_mainboard_screener_raw.txt plus the pilot rows of data/manifest.csv. Every row is
checked: the gain printed by the source must equal listing price / issue price - 1 (to 0.2 points), which
catches transcription errors. With --features, pre-listing features are added for companies whose DRHP is in
data/raw/drhp (the pilots); other rows keep those columns empty rather than guessed.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "sources" / "trendlyne_mainboard_screener_raw.txt"
SUFFIX = re.compile(r"\s+(ltd\.?|limited)$", re.I)


def parse_raw() -> pd.DataFrame:
    rows = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, date, issue, listing, gain, sub = line.rsplit(",", 5)
        d = pd.to_datetime(re.sub(r"\s*[`']", " 20", date), format="%d %b %Y")
        rows.append({"company": SUFFIX.sub("", name.strip()), "listing_date": d.date().isoformat(), "issue_price": float(issue),
                     "listing_price": float(listing), "reported_gain_pct": float(gain.strip("%+")),
                     "subscription_total_x": float(sub.rstrip("x")), "source": "Trendlyne mainboard screener"})
    return pd.DataFrame(rows)


def add_pilots(df: pd.DataFrame) -> pd.DataFrame:
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    extra = []
    for _, r in man.iterrows():
        if r["company"] in set(df["company"]) or pd.isna(r["listing_open"]) or pd.isna(r["issue_price"]):
            continue
        extra.append({"company": r["company"], "listing_date": r["listing_date"], "issue_price": float(r["issue_price"]),
                      "listing_price": float(r["listing_open"]), "reported_gain_pct": None,
                      "subscription_total_x": r["subscription_x"], "source": "data/manifest.csv (web sources, see price_source_note)"})
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df["computed_gain_pct"] = (df["listing_price"] / df["issue_price"] - 1) * 100
    bad = df[(df["reported_gain_pct"].notna()) & ((df["computed_gain_pct"] - df["reported_gain_pct"]).abs() > 0.2)]
    if len(bad):
        print("ROWS WHOSE REPORTED GAIN DOES NOT MATCH listing/issue - 1:")
        print(bad[["company", "issue_price", "listing_price", "reported_gain_pct", "computed_gain_pct"]].to_string(index=False))
    else:
        print(f"consistency check: all {df['reported_gain_pct'].notna().sum()} rows with a reported gain match listing/issue - 1")
    dup = df[df.duplicated("company", keep=False)]
    if len(dup):
        print("DUPLICATE COMPANIES:", sorted(set(dup["company"])))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true", help="extract pre-listing features from DRHPs present in data/raw/drhp")
    a = ap.parse_args()
    df = validate(add_pilots(parse_raw())).sort_values("listing_date").reset_index(drop=True)
    df.drop(columns=["computed_gain_pct"]).to_csv(ROOT / "data" / "ipo_outcomes.csv", index=False)
    out = df[["company", "listing_date", "issue_price", "listing_price", "subscription_total_x"]].copy()
    for c in ("sector", "revenue_growth", "pat_margin", "roe", "debt_to_equity", "peer_median_pe", "issue_size_inr", "fresh_issue_share", "nifty_return_30d"):
        out[c] = None
    if a.features:
        sys.path.insert(0, str(ROOT / "src"))
        from build_ipo_features import features_for

        man = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("company")
        for company in out["company"]:
            if company in man.index and (ROOT / "data" / man.at[company, "file"]).exists():
                i = out.index[out["company"] == company][0]
                out.at[i, "sector"] = man.at[company, "sector"]
                for k, v in features_for(ROOT / "data" / man.at[company, "file"], company).items():
                    if k in out.columns:
                        out.at[i, k] = v
                print("features", company, flush=True)
    out.to_csv(ROOT / "data" / "ipo_history.csv", index=False)
    print(f"{len(out)} IPOs, {out['listing_date'].min()} to {out['listing_date'].max()}")


if __name__ == "__main__":
    main()
