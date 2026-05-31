from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "march 2025"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def parse_price_series(s: pd.Series) -> pd.Series:
    """Convert Inside Airbnb price strings ('$1,234.00') to float."""
    return (
        s.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(",", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
        .astype(float)
    )


_BATH_NUM = re.compile(r"([\d.]+)")


def parse_bathrooms(text: str) -> tuple[float, str]:
    """Return (count, kind) from 'bathrooms_text'. Kind in {private, shared, regular}."""
    if not isinstance(text, str):
        return (np.nan, "unknown")
    s = text.strip().lower()
    if "half" in s and not any(c.isdigit() for c in s):
        n = 0.5
    else:
        m = _BATH_NUM.search(s)
        n = float(m.group(1)) if m else np.nan
    if "private" in s:
        kind = "private"
    elif "shared" in s:
        kind = "shared"
    else:
        kind = "regular"
    return n, kind


def calendar_median_price(calendar_path: Path) -> pd.DataFrame:
    """Compute median nightly price per listing from the calendar.

    The calendar has 365 nights per listing. Listing-level price in listings.csv is
    a snapshot and often missing — the calendar median is far more reliable.
    """
    print(f"  reading calendar from {calendar_path.name} ...")
    cal = pd.read_csv(calendar_path, usecols=["listing_id", "price", "available"])
    cal["price"] = parse_price_series(cal["price"])
    cal = cal.dropna(subset=["price"])
    # use all rows (available or not) — booked-night price is still a price signal
    out = (
        cal.groupby("listing_id")["price"]
        .median()
        .rename("price")
        .reset_index()
        .rename(columns={"listing_id": "id"})
    )
    print(f"  calendar median price computed for {len(out):,} listings")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    listings_path = RAW_DIR / "listings-main.csv"
    calendar_path = RAW_DIR / "calendar.csv"
    geojson_path = RAW_DIR / "neighbourhoods.geojson"

    if not listings_path.exists():
        raise FileNotFoundError(f"missing {listings_path}")

    print("loading listings ...")
    df = pd.read_csv(listings_path, low_memory=False)
    n0 = len(df)
    print(f"  raw rows: {n0:,}, cols: {df.shape[1]}")

    # keep a minimal column set; everything else is dropped
    keep = [
        "id",
        "latitude", "longitude",
        "neighbourhood_cleansed",
        "property_type", "room_type",
        "accommodates", "bedrooms", "beds", "bathrooms_text",
        "amenities",
        "minimum_nights", "maximum_nights",
        "availability_365", "estimated_occupancy_l365d",
        "instant_bookable",
        "number_of_reviews", "review_scores_rating",
        "price",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # price: prefer calendar median; fall back to listings price
    df["listing_price"] = parse_price_series(df["price"])
    cal_price = calendar_median_price(calendar_path)
    df = df.merge(cal_price, on="id", how="left", suffixes=("_listing", ""))
    df["price"] = df["price"].fillna(df["listing_price"])
    df = df.drop(columns=["listing_price"])
    n_with_price = df["price"].notna().sum()
    print(f"  rows with price: {n_with_price:,} ({n_with_price / n0:.1%})")

    # drop rows with no usable price — they cannot be analyzed
    df = df.dropna(subset=["price"])

    # bathrooms_text → numeric + kind
    bath = df["bathrooms_text"].map(parse_bathrooms)
    df["bathrooms_n"] = bath.map(lambda x: x[0])
    df["bathrooms_kind"] = bath.map(lambda x: x[1])

    # drop rows missing structural fields we treat as required
    before = len(df)
    df = df.dropna(subset=["bedrooms", "bathrooms_n", "latitude", "longitude"])
    print(f"  dropped {before - len(df):,} rows missing bedrooms/bath/coord")

    # impute beds via IterativeImputer with RandomForest using only structural cols
    # (this matches the imputer comparison in the original notebook — RF was the best)
    impute_feats = ["accommodates", "bedrooms", "bathrooms_n"]
    needs_imp = df["beds"].isna().sum()
    if needs_imp > 0:
        print(f"  imputing {needs_imp:,} missing 'beds' values (IterativeImputer + RF) ...")
        sub = df[impute_feats + ["beds"]].copy()
        imp = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50, max_depth=5, random_state=1),
            max_iter=20,
            random_state=1,
        )
        sub_imputed = pd.DataFrame(imp.fit_transform(sub), columns=sub.columns, index=sub.index)
        df["beds"] = np.rint(sub_imputed["beds"]).clip(lower=1).astype("Int64")

    # amenities — count is enough for clustering; we drop the TF-IDF baroque tower
    def amenity_count(s):
        if not isinstance(s, str):
            return 0
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            # python-list-literal form
            try:
                lst = json.loads(s.replace("'", '"'))
                return len(lst) if isinstance(lst, list) else 0
            except Exception:
                pass
        return len([x for x in s.split(",") if x.strip()])

    df["amenities_count"] = df["amenities"].map(amenity_count)

    # instant_bookable t/f → 1/0
    df["instant_bookable"] = (
        df["instant_bookable"].astype(str).str.strip().str.lower()
        .map({"t": 1, "f": 0}).astype("Int8")
    )

    # min/max nights — cap at 365 (the rest are placeholders per Inside Airbnb)
    df["minimum_nights"] = df["minimum_nights"].clip(upper=365)
    df["maximum_nights"] = df["maximum_nights"].clip(upper=365)

    # price sanity — drop the 1% extreme tails to stop them dominating later
    p_lo, p_hi = df["price"].quantile([0.005, 0.995])
    before = len(df)
    df = df[(df["price"] >= p_lo) & (df["price"] <= p_hi)].copy()
    print(f"  trimmed {before - len(df):,} rows with extreme price (<{p_lo:.0f} or >{p_hi:.0f})")

    # final columns — keep what we need for modeling + interpretation
    final_cols = [
        "id",
        "latitude", "longitude", "neighbourhood_cleansed",
        "property_type", "room_type",
        "accommodates", "bedrooms", "beds",
        "bathrooms_n", "bathrooms_kind",
        "amenities_count",
        "minimum_nights", "maximum_nights",
        "availability_365", "estimated_occupancy_l365d",
        "instant_bookable",
        "number_of_reviews", "review_scores_rating",
        "price",
    ]
    df = df[[c for c in final_cols if c in df.columns]].reset_index(drop=True)

    print(f"final rows: {len(df):,}, cols: {df.shape[1]}")

    out_path = PROC_DIR / "listings_clean.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")

    # copy geojson for downstream use
    if geojson_path.exists():
        shutil.copy(geojson_path, PROC_DIR / "neighbourhoods.geojson")

    summary = {
        "raw_rows": int(n0),
        "rows_after_clean": int(len(df)),
        "columns": list(df.columns),
        "median_price": float(df["price"].median()),
        "neighbourhoods": int(df["neighbourhood_cleansed"].nunique()),
        "room_types": df["room_type"].value_counts().to_dict(),
    }
    with open(PROC_DIR / "data_prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
