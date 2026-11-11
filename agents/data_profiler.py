"""
Agent 1 — Data Profiler
Profiles an uploaded DataFrame and determines task type (classification/regression).
"""
import numpy as np
import pandas as pd
from scipy import stats


def profile(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """
    Returns a comprehensive profile dict of the DataFrame.
    If target_col is None, attempts to auto-detect the target.
    """
    result = {}

    # ── Basic shape ──────────────────────────────────────────────────────────
    result["n_rows"]       = int(df.shape[0])
    result["n_cols"]       = int(df.shape[1])
    result["n_duplicates"] = int(df.duplicated().sum())
    result["columns"]      = list(df.columns)

    # ── Column types ─────────────────────────────────────────────────────────
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    result["numeric_cols"]     = numeric_cols
    result["categorical_cols"] = categorical_cols
    result["n_numeric"]        = len(numeric_cols)
    result["n_categorical"]    = len(categorical_cols)

    # ── Missing values ───────────────────────────────────────────────────────
    missing         = df.isnull().sum()
    missing_pct     = (missing / len(df) * 100).round(2)
    result["missing_per_col"]  = missing.to_dict()
    result["missing_pct"]      = missing_pct.to_dict()
    result["total_missing"]    = int(missing.sum())
    result["cols_with_missing"] = missing[missing > 0].index.tolist()

    # ── Numeric statistics ───────────────────────────────────────────────────
    num_stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        num_stats[col] = {
            "mean":     float(s.mean()),
            "median":   float(s.median()),
            "std":      float(s.std()),
            "min":      float(s.min()),
            "max":      float(s.max()),
            "skewness": float(stats.skew(s)),
            "kurtosis": float(stats.kurtosis(s)),
            "outliers": outliers,
        }
    result["numeric_stats"] = num_stats

    # ── Categorical statistics ───────────────────────────────────────────────
    cat_stats = {}
    for col in categorical_cols:
        s = df[col].dropna()
        cat_stats[col] = {
            "n_unique":    int(s.nunique()),
            "top_values":  s.value_counts().head(5).to_dict(),
        }
    result["categorical_stats"] = cat_stats

    # ── Correlation ──────────────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        # Top 10 highest absolute correlations (excluding self)
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                corr_pairs.append({
                    "col_a": corr.columns[i],
                    "col_b": corr.columns[j],
                    "r":     round(float(corr.iloc[i, j]), 4),
                })
        corr_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
        result["top_correlations"] = corr_pairs[:10]
    else:
        result["top_correlations"] = []

    # ── Target / task type detection ─────────────────────────────────────────
    if target_col is None:
        target_col = _guess_target(df)
    result["target_col"] = target_col

    if target_col and target_col in df.columns:
        y = df[target_col].dropna()
        n_unique = y.nunique()
        if y.dtype == object or n_unique <= 20:
            result["task_type"] = "classification"
            result["n_classes"] = int(n_unique)
            result["class_counts"] = y.value_counts().to_dict()
        else:
            result["task_type"] = "regression"
            result["target_stats"] = {
                "mean": float(y.mean()),
                "std":  float(y.std()),
                "min":  float(y.min()),
                "max":  float(y.max()),
            }
    else:
        result["task_type"] = "unknown"

    return result


def _guess_target(df: pd.DataFrame) -> str | None:
    """Heuristically guess the target column (last column is common convention)."""
    candidates = ["target", "label", "class", "y", "output", "result",
                  "price", "salary", "churn", "survived", "species"]
    lower_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_cols:
            return lower_cols[c]
    # Fallback: last column
    if len(df.columns) > 1:
        return df.columns[-1]
    return None


def summary_text(profile: dict) -> str:
    """Returns a short human-readable summary of the profile."""
    lines = [
        f"**Shape**: {profile['n_rows']:,} rows × {profile['n_cols']} columns",
        f"**Task**: {profile.get('task_type', 'unknown').title()}  |  **Target**: `{profile.get('target_col', '—')}`",
        f"**Numeric**: {profile['n_numeric']}  |  **Categorical**: {profile['n_categorical']}",
        f"**Missing cells**: {profile['total_missing']:,}  |  **Duplicate rows**: {profile['n_duplicates']:,}",
    ]
    if profile.get("task_type") == "classification":
        lines.append(f"**Classes**: {profile.get('n_classes')} unique values")
    return "\n\n".join(lines)
