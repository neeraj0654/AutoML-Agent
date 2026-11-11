"""
Agent 4 — Report Generator
Generates an LLM-written narrative report from the pipeline results.
"""
from utils import llm_client


def generate(
    profile: dict,
    feature_plan: dict,
    trainer_result: dict,
    importance_dict: dict | None,
    dataset_name: str = "Dataset",
) -> str:
    """
    Returns a markdown-formatted report string.
    """
    prompt = _build_prompt(profile, feature_plan, trainer_result, importance_dict, dataset_name)
    report = llm_client.ask(prompt, temperature=0.5)
    return report


def _build_prompt(profile, feature_plan, trainer_result, importance_dict, dataset_name) -> str:
    best_name  = trainer_result["best_model_name"]
    best_score = trainer_result["best_score"]
    metric     = trainer_result["metric_name"]
    task       = profile.get("task_type", "unknown")
    target     = profile.get("target_col", "unknown")
    leaderboard = trainer_result["leaderboard"][:5]
    test_metrics = trainer_result.get("test_metrics", {})

    top_features = ""
    if importance_dict:
        sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        top_features = "\n".join(f"  - {f}: {v:.4f}" for f, v in sorted_feats)

    lb_text = "\n".join(
        f"  {i+1}. {r['model']}: {r['cv_score']:.4f} ± {r['cv_std']:.4f}"
        for i, r in enumerate(leaderboard)
    )

    test_text = ""
    if task == "regression":
        test_text = (
            f"  R²={test_metrics.get('r2', 'N/A'):.4f}, "
            f"MAE={test_metrics.get('mae', 'N/A'):.4f}, "
            f"RMSE={test_metrics.get('rmse', 'N/A'):.4f}"
        )
    elif task == "classification":
        cr = test_metrics.get("classification_report", {})
        wf1 = cr.get("weighted avg", {}).get("f1-score", "N/A")
        test_text = f"  Weighted F1 on test set: {wf1:.4f}" if isinstance(wf1, float) else "  See classification report."

    return f"""You are an expert data scientist writing a professional AutoML report for a client.
Write in clear, insightful, non-technical language (but technically accurate).
Structure the report with markdown headings.

## Context
- Dataset: {dataset_name}
- Task: {task.title()} — predicting `{target}`
- Rows: {profile.get('n_rows'):,}, Columns: {profile.get('n_cols')}
- Missing values: {profile.get('total_missing'):,} cells across {len(profile.get('cols_with_missing', []))} columns
- Duplicate rows: {profile.get('n_duplicates'):,}

## Feature Engineering Applied
{chr(10).join(f'- {s}' for s in feature_plan.get('steps_applied', []))}

## Model Leaderboard (5-fold CV, metric: {metric})
{lb_text}

## Best Model: {best_name}
- CV {metric}: {best_score:.4f}
- Test set performance:
{test_text}

## Top 10 Important Features
{top_features if top_features else 'Feature importances not available for this model.'}

---

Write a professional report (400-600 words) covering:
1. **Executive Summary** — what the dataset is about and what was predicted
2. **Data Quality Assessment** — key findings from profiling (missing values, outliers, correlations)
3. **Model Selection** — why the winning model outperformed others, what this model's strengths are
4. **Key Insights** — what the top features reveal about the problem domain
5. **Recommendations** — next steps to further improve the model

Do NOT repeat the raw numbers verbatim — interpret them. Use markdown formatting.
"""


def fallback_report(profile: dict, trainer_result: dict) -> str:
    """Returns a rule-based report when LLM is unavailable."""
    best   = trainer_result["best_model_name"]
    score  = trainer_result["best_score"]
    metric = trainer_result["metric_name"]
    task   = profile.get("task_type", "unknown")
    target = profile.get("target_col", "—")

    return f"""## AutoML Report

### Executive Summary
This AutoML pipeline analyzed your dataset to predict **`{target}`** — a **{task}** task.

### Data Quality
- **{profile['n_rows']:,}** rows and **{profile['n_cols']}** columns were loaded.
- **{profile['total_missing']:,}** missing values were detected and handled via imputation.
- **{profile['n_duplicates']:,}** duplicate rows were found.

### Model Selection
After training **{len(trainer_result['leaderboard'])}** models with 5-fold cross-validation, the best-performing model was:

> **{best}** — {metric}: **{score:.4f}**

### Leaderboard Summary
| Rank | Model | CV Score |
|------|-------|----------|
{''.join(f"| {i+1} | {r['model']} | {r['cv_score']:.4f} |" + chr(10) for i, r in enumerate(trainer_result['leaderboard'][:5]))}

### Recommendations
- Add a Gemini API key to `.env` for a detailed AI-generated analysis.
- Consider hyperparameter tuning on the best model.
- Gather more data to improve model robustness.
"""
