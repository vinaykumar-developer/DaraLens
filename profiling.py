"""
profiling.py
------------
Responsible for looking at a raw pandas DataFrame and producing a structured
"profile" dictionary describing every column: its dtype, how much data is
missing, how many unique values it has, and a simplified "type" classification
(numeric / categorical / datetime / constant / text).

This profile is the single source of truth that every insight rule in
insight_rules.py reads from — rules never need to re-inspect the raw
DataFrame's dtypes themselves.
"""

import pandas as pd
import numpy as np


def _classify_column(series: pd.Series) -> str:
    """
    Decide which broad 'type' a column belongs to.

    Order of checks matters:
    1. Constant columns (only 1 unique non-null value) are flagged first,
       regardless of dtype, because a constant numeric column is not
       useful to treat as "numeric" for skew/outlier analysis.
    2. Datetime columns (either already datetime64, or object columns that
       pandas can confidently parse as dates).
    3. Numeric columns (int/float dtypes).
    4. Categorical vs free text — object/category columns are treated as
       'categorical' if they have relatively few unique values compared to
       the number of rows, otherwise as 'text' (e.g. free-form comments,
       IDs, names) since those aren't useful for grouping.
    """
    non_null = series.dropna()

    # 1. Constant / near-constant (only one distinct value present)
    if non_null.nunique(dropna=True) <= 1:
        return "constant"

    # 2. Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.dtype == object:
        # Try a cautious parse — only classify as datetime if the vast
        # majority of non-null values parse successfully. This avoids
        # accidentally classifying normal text columns as dates.
        try:
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            success_ratio = parsed.notna().mean() if len(non_null) > 0 else 0
            if success_ratio > 0.9:
                return "datetime"
        except Exception:
            pass

    # 3. Numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # 4. Categorical vs text — based on cardinality ratio
    n_rows = len(series)
    n_unique = non_null.nunique()
    if n_rows == 0:
        return "text"
    unique_ratio = n_unique / n_rows

    # Heuristic: if there are relatively few distinct values (either an
    # absolute cap or a low ratio of unique-to-total), treat as categorical.
    if n_unique <= 50 or unique_ratio < 0.5:
        return "categorical"

    return "text"


def profile_columns(df: pd.DataFrame) -> dict:
    """
    Build a profile dictionary for every column in df.

    Returns:
        {
            column_name: {
                "dtype": str,              # pandas dtype as string
                "missing_pct": float,      # % of rows that are NaN/None
                "missing_count": int,
                "unique_count": int,
                "type": str,               # numeric / categorical / datetime / constant / text
                "skew": float | None,      # only for numeric columns
                "std": float | None,       # only for numeric columns
            },
            ...
        }
    """
    profile = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        col_type = _classify_column(series)

        skew = None
        std = None
        if col_type == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric_series) > 2:
                # skew() needs a handful of points to be meaningful
                try:
                    skew = round(float(numeric_series.skew()), 3)
                except Exception:
                    skew = None
                try:
                    std = round(float(numeric_series.std()), 3)
                except Exception:
                    std = None

        profile[col] = {
            "dtype": str(series.dtype),
            "missing_pct": missing_pct,
            "missing_count": missing_count,
            "unique_count": unique_count,
            "type": col_type,
            "skew": skew,
            "std": std,
        }

    return profile
