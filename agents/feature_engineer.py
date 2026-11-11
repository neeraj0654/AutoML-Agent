"""
Agent 2 — Feature Engineer
Prepares features using LLM guidance + sklearn pipelines.
Returns X_train, X_test, y_train, y_test plus a plan dict.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

from utils import llm_client


def engineer(
    df: pd.DataFrame,
    profile: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Returns:
        {
          "X_train": ..., "X_test": ..., "y_train": ..., "y_test": ...,
          "preprocessor": sklearn Pipeline,
          "feature_names": [...],
          "steps_applied": [...],
          "llm_suggestions": "...",
          "label_encoder": LabelEncoder | None,
        }
    """
    target_col  = profile["target_col"]
    task_type   = profile["task_type"]
    steps_applied = []

    # ── Drop target ──────────────────────────────────────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # ── Drop columns that are entirely missing ───────────────────────────────
    all_missing = X.columns[X.isnull().all()].tolist()
    if all_missing:
        X = X.drop(columns=all_missing)
        steps_applied.append(f"Dropped {len(all_missing)} fully-missing column(s): {all_missing}")

    # ── Drop high-cardinality identifier columns ─────────────────────────────
    id_cols = [c for c in X.select_dtypes(exclude=[np.number]).columns
               if X[c].nunique() / len(X) > 0.95]
    if id_cols:
        X = X.drop(columns=id_cols)
        steps_applied.append(f"Dropped {len(id_cols)} high-cardinality ID column(s): {id_cols}")

    # ── Encode target ────────────────────────────────────────────────────────
    label_encoder = None
    if task_type == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        label_encoder = le
        steps_applied.append(f"Label-encoded target `{target_col}` → integers")

    # ── Identify feature types ────────────────────────────────────────────────
    numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── Build sklearn ColumnTransformer ──────────────────────────────────────
    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numeric_features))
        steps_applied.append(f"Numeric ({len(numeric_features)} cols): median imputation + StandardScaler")

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))
        steps_applied.append(f"Categorical ({len(categorical_features)} cols): mode imputation + OneHotEncoder")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # ── Train/test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=(y if task_type == "classification" and y.nunique() > 1 else None),
    )
    steps_applied.append(f"Train/test split: {1-test_size:.0%} / {test_size:.0%} (random_state={random_state})")

    # ── Fit preprocessor ─────────────────────────────────────────────────────
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # ── Derive feature names after encoding ──────────────────────────────────
    feature_names = _get_feature_names(preprocessor, numeric_features, categorical_features)

    # ── LLM suggestions ──────────────────────────────────────────────────────
    llm_prompt = _build_llm_prompt(profile, numeric_features, categorical_features, steps_applied)
    llm_suggestions = llm_client.ask(llm_prompt)

    return {
        "X_train":         X_train_t,
        "X_test":          X_test_t,
        "y_train":         y_train.values,
        "y_test":          y_test.values,
        "preprocessor":    preprocessor,
        "feature_names":   feature_names,
        "steps_applied":   steps_applied,
        "llm_suggestions": llm_suggestions,
        "label_encoder":   label_encoder,
        "numeric_features":     numeric_features,
        "categorical_features": categorical_features,
    }


def _get_feature_names(preprocessor, numeric_features, categorical_features) -> list:
    names = list(numeric_features)
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            try:
                enc = transformer.named_steps["encoder"]
                ohe_names = enc.get_feature_names_out(cols).tolist()
                names += ohe_names
            except Exception:
                names += list(cols)
    return names


def _build_llm_prompt(profile, num_cols, cat_cols, steps) -> str:
    return f"""You are a senior data scientist performing feature engineering for a machine learning task.

Dataset profile:
- Task type: {profile.get('task_type', 'unknown')}
- Target column: {profile.get('target_col')}
- Rows: {profile.get('n_rows')}, Columns: {profile.get('n_cols')}
- Numeric columns: {num_cols}
- Categorical columns: {cat_cols}
- Columns with missing values: {profile.get('cols_with_missing', [])}
- Top correlations: {profile.get('top_correlations', [])[:5]}

Steps already applied by the pipeline:
{chr(10).join(f'  - {s}' for s in steps)}

Please provide:
1. Brief assessment of the data quality
2. Any domain-specific feature engineering ideas that might improve model performance
3. Any potential data issues to watch out for

Keep your response concise (under 300 words) and practical.
"""
