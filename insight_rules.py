"""
insight_rules.py
-----------------
Every "insight rule" lives here. A rule is a plain Python function with the
signature:

    rule_function(df: pd.DataFrame, profile: dict) -> list[dict]

...where each returned dict is a "finding":
    {
        "column": str,          # column name, or "-" / "multiple" if not tied to one column
        "message": str,         # human-readable explanation, built with f-strings
        "severity": str,        # "critical" | "warning" | "info"
    }

All messages are generated with plain f-string templates filled in with real
computed numbers — there is no ML/NLP text generation involved anywhere.

To add a NEW rule:
    1. Write a function following the signature above.
    2. Add it to the INSIGHT_RULES list at the bottom with a "name" key.
That's it — engine.py automatically picks up and runs every rule in the list.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Rule 1: Missing values
# ---------------------------------------------------------------------------
def check_missing_values(df: pd.DataFrame, profile: dict) -> list:
    """
    Flags columns with a high percentage of missing values.

    Thresholds:
        > 50%  -> critical (column may be unusable)
        20-50% -> warning   (imputation strategy needed)
        5-20%  -> info      (worth being aware of)
    """
    findings = []
    for col, info in profile.items():
        pct = info["missing_pct"]
        if pct > 50:
            findings.append({
                "column": col,
                "message": (
                    f"Column '{col}' has {pct}% missing values "
                    f"({info['missing_count']} rows). This column may be unreliable "
                    f"for analysis unless the missingness itself is meaningful."
                ),
                "severity": "critical",
            })
        elif pct > 20:
            findings.append({
                "column": col,
                "message": (
                    f"Column '{col}' has {pct}% missing values. "
                    f"Consider an imputation strategy (mean/median/mode) or dropping rows."
                ),
                "severity": "warning",
            })
        elif pct > 5:
            findings.append({
                "column": col,
                "message": f"Column '{col}' has {pct}% missing values — worth reviewing.",
                "severity": "info",
            })
    return findings


# ---------------------------------------------------------------------------
# Rule 2: Skewness
# ---------------------------------------------------------------------------
def check_skewness(df: pd.DataFrame, profile: dict) -> list:
    """
    Flags numeric columns with |skew| > 1 (moderately-to-highly skewed),
    and suggests an appropriate transform based on the direction of skew.
    """
    findings = []
    for col, info in profile.items():
        if info["type"] != "numeric" or info["skew"] is None:
            continue
        skew = info["skew"]
        if abs(skew) > 1:
            direction = "right (positively)" if skew > 0 else "left (negatively)"
            transform = "log or square-root transform" if skew > 0 else "square or cube transform"
            severity = "warning" if abs(skew) <= 2 else "critical"
            findings.append({
                "column": col,
                "message": (
                    f"Column '{col}' is skewed {direction} (skew = {skew}). "
                    f"A {transform} may help normalize its distribution before modeling."
                ),
                "severity": severity,
            })
    return findings


# ---------------------------------------------------------------------------
# Rule 3: High correlation pairs
# ---------------------------------------------------------------------------
def check_high_correlation(df: pd.DataFrame, profile: dict) -> list:
    """
    Flags pairs of numeric columns with an absolute Pearson correlation > 0.85.
    High correlation can indicate redundant features (multicollinearity).
    """
    findings = []
    numeric_cols = [c for c, info in profile.items() if info["type"] == "numeric"]

    if len(numeric_cols) < 2:
        return findings

    corr_matrix = df[numeric_cols].corr(numeric_only=True)

    seen_pairs = set()
    for col_a in corr_matrix.columns:
        for col_b in corr_matrix.columns:
            if col_a == col_b:
                continue
            pair_key = tuple(sorted([col_a, col_b]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            corr_value = corr_matrix.loc[col_a, col_b]
            if pd.isna(corr_value):
                continue
            if abs(corr_value) > 0.85:
                findings.append({
                    "column": f"{col_a} & {col_b}",
                    "message": (
                        f"Columns '{col_a}' and '{col_b}' are highly correlated "
                        f"(r = {round(corr_value, 3)}). One of them may be redundant."
                    ),
                    "severity": "warning",
                })
    return findings


# ---------------------------------------------------------------------------
# Rule 4: Outlier detection (IQR method)
# ---------------------------------------------------------------------------
def check_outliers_iqr(df: pd.DataFrame, profile: dict) -> list:
    """
    Uses the IQR (interquartile range) method to detect outliers in numeric
    columns: any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier.

    Flags columns where outliers make up more than 1% of non-null values.
    """
    findings = []
    for col, info in profile.items():
        if info["type"] != "numeric":
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 5:
            continue  # not enough data points for a meaningful IQR check

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue  # avoid flagging when the middle 50% of data is constant

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(series)) * 100, 2)

        if outlier_pct > 1:
            severity = "warning" if outlier_pct <= 10 else "critical"
            findings.append({
                "column": col,
                "message": (
                    f"Column '{col}' has {outlier_count} outlier value(s) "
                    f"({outlier_pct}% of data) outside the range "
                    f"[{round(lower_bound, 2)}, {round(upper_bound, 2)}] (IQR method)."
                ),
                "severity": severity,
            })
    return findings


# ---------------------------------------------------------------------------
# Rule 5: Constant / near-constant columns
# ---------------------------------------------------------------------------
def check_constant_columns(df: pd.DataFrame, profile: dict) -> list:
    """
    Flags columns that have only one distinct value — they carry no
    information for analysis or modeling and can usually be dropped.
    """
    findings = []
    for col, info in profile.items():
        if info["type"] == "constant":
            findings.append({
                "column": col,
                "message": (
                    f"Column '{col}' contains only a single unique value "
                    f"across all rows. It carries no useful information and "
                    f"can likely be dropped."
                ),
                "severity": "warning",
            })
    return findings


# ---------------------------------------------------------------------------
# Rule 6: High cardinality categorical columns
# ---------------------------------------------------------------------------
def check_high_cardinality(df: pd.DataFrame, profile: dict) -> list:
    """
    Flags categorical (or text-classified) columns whose number of unique
    values is too high relative to the dataset to be useful for grouping
    or one-hot encoding.
    """
    findings = []
    n_rows = len(df)
    if n_rows == 0:
        return findings

    for col, info in profile.items():
        if info["type"] not in ("categorical", "text"):
            continue
        unique_count = info["unique_count"]
        unique_ratio = unique_count / n_rows

        if unique_count > 50 and unique_ratio > 0.5:
            findings.append({
                "column": col,
                "message": (
                    f"Column '{col}' has {unique_count} unique values "
                    f"({round(unique_ratio * 100, 1)}% of rows are distinct). "
                    f"It likely behaves like an identifier and is not useful "
                    f"for grouping or one-hot encoding as-is."
                ),
                "severity": "info",
            })
    return findings


# ---------------------------------------------------------------------------
# Rule 7: Possible class imbalance
# ---------------------------------------------------------------------------
def check_class_imbalance(df: pd.DataFrame, profile: dict, target_col: str = None) -> list:
    """
    Attempts to detect a likely target/label column and flags class
    imbalance if one class makes up more than 80% of the rows.

    Target detection heuristic (used only if target_col isn't explicitly given):
        - Prefer a low-cardinality categorical column (2-20 unique values)
        - Prefer columns near the END of the DataFrame (common convention:
          the label is usually the last column)
    """
    findings = []

    candidate = target_col

    if candidate is None:
        # Search columns from last to first for a plausible label column
        categorical_like = [
            col for col, info in profile.items()
            if info["type"] == "categorical" and 2 <= info["unique_count"] <= 20
        ]
        if categorical_like:
            # Preserve DataFrame column order, then take the last matching one
            ordered = [c for c in df.columns if c in categorical_like]
            if ordered:
                candidate = ordered[-1]

    if candidate is None or candidate not in df.columns:
        return findings

    value_counts = df[candidate].value_counts(normalize=True, dropna=True)
    if value_counts.empty:
        return findings

    top_class = value_counts.index[0]
    top_share = value_counts.iloc[0]

    if top_share > 0.8:
        findings.append({
            "column": candidate,
            "message": (
                f"Possible class imbalance detected in likely target column "
                f"'{candidate}': class '{top_class}' makes up "
                f"{round(top_share * 100, 1)}% of rows. Consider resampling, "
                f"class weights, or stratified sampling when modeling."
            ),
            "severity": "warning",
        })
    return findings


# ---------------------------------------------------------------------------
# Rule registry — engine.py loops over this list. Add new rules here.
# ---------------------------------------------------------------------------
INSIGHT_RULES = [
    {"name": "Missing Values", "function": check_missing_values},
    {"name": "Skewness", "function": check_skewness},
    {"name": "High Correlation", "function": check_high_correlation},
    {"name": "Outliers (IQR)", "function": check_outliers_iqr},
    {"name": "Constant Columns", "function": check_constant_columns},
    {"name": "High Cardinality", "function": check_high_cardinality},
    {"name": "Class Imbalance", "function": check_class_imbalance},
]
