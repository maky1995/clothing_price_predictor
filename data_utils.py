"""
Shared data-loading and cleaning logic for the Clothing Price Predictor.

Used by both train_model.py (offline training) and app.py (Streamlit UI),
so the exact same cleaning rules are guaranteed to apply in both places.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

CSV_CANDIDATES = [
    BASE_DIR / "Clothing_dataset.csv",
    BASE_DIR / "clothing_dataset.csv",
]
EXCEL_CANDIDATES = [
    BASE_DIR / "Clothing_dataset.xlsx",
    BASE_DIR / "Overfitting clothing_dataset.xlsx",
]

FEATURES = ["category", "gender", "brand", "fabric", "pattern", "occasion", "length"]

DROP_COLUMNS = [
    "product_id", "product_title", "fit", "neck", "closure",
    "sleeve_length", "currency", "description", "material_and_care",
    "discount_pct", "final_price", "all_image_urls", "rating",
    "ratings_count", "primary_image_url", "Unnamed: 23",
]

# Rows near the end of the source file are misaligned (columns shifted,
# apparently from a different product export mixed into this file): their
# "gender"/"occasion"/"length" cells contain full sentences or raw JSON
# instead of short categorical tokens. These three columns are reliable
# corruption signals (unlike "brand"/"category", which can legitimately run
# longer, e.g. "AMERICAN EAGLE OUTFITTERS"), so we use them to drop the
# misaligned tail without hard-coding a row number.
CORRUPTION_CHECK_COLUMNS = ["gender", "occasion", "length"]
MAX_VALID_TOKEN_LEN = 20


def clean_num(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )


def load_dataset():
    """Load the raw dataset file, tolerating a mislabeled extension.

    The dataset is sometimes exported as an Excel file but saved with a
    .csv extension (or vice versa), so we sniff by trying both readers
    rather than trusting the file name.
    """
    for path in CSV_CANDIDATES:
        if path.exists():
            try:
                return pd.read_csv(path)
            except (UnicodeDecodeError, pd.errors.ParserError):
                return pd.read_excel(path)

    for path in EXCEL_CANDIDATES:
        if path.exists():
            return pd.read_excel(path)

    raise FileNotFoundError(
        "Dataset not found. Put Clothing_dataset.csv (or .xlsx) in this folder."
    )


def prepare_data(df):
    """Clean the raw dataframe into a training-ready frame."""
    df = df.copy()

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Drop rows near the end of the export whose columns are shifted /
    # misaligned (see MAX_VALID_TOKEN_LEN comment above), detected by
    # checking that every feature value looks like a short categorical token.
    check_cols = [c for c in CORRUPTION_CHECK_COLUMNS if c in df.columns]
    valid_mask = pd.Series(True, index=df.index)
    for col in check_cols:
        valid_mask &= df[col].astype(str).str.len().fillna(0) <= MAX_VALID_TOKEN_LEN
    # A row with every feature column blank isn't a real product record
    # (seen at the very end of the export); drop it rather than let it
    # through just because "nan" happens to be a short string.
    valid_mask &= df[FEATURES].notna().any(axis=1)
    df = df[valid_mask].copy()

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required columns: {missing_features}")

    for col in FEATURES:
        mode = df[col].mode()
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df[col] = df[col].fillna(fill_value).astype(str)

    df["initial_price"] = clean_num(df["initial_price"])
    df["initial_price"] = df["initial_price"].fillna(df["initial_price"].median())
    df = df.dropna(subset=["initial_price"])

    return df.reset_index(drop=True)
