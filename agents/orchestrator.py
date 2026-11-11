"""
Orchestrator — coordinates the full AutoML pipeline across all agents.
"""
import pandas as pd
from agents import data_profiler, feature_engineer, model_trainer, report_generator
from agents.model_trainer import build_importance_dict


def run_pipeline(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    target_col: str | None = None,
    cv_folds: int = 5,
    test_size: float = 0.2,
    progress_callback=None,   # callback(stage: str, pct: int, message: str)
) -> dict:
    """
    Runs the full AutoML pipeline and returns a results dict containing
    all intermediate and final outputs for the Streamlit UI.

    Returns:
        {
          "profile":         dict,
          "feature_plan":    dict,
          "trainer_result":  dict,
          "importance_dict": dict | None,
          "report_text":     str,
          "dataset_name":    str,
          "error":           str | None,
        }
    """
    result = {
        "profile":         None,
        "feature_plan":    None,
        "trainer_result":  None,
        "importance_dict": None,
        "report_text":     None,
        "dataset_name":    dataset_name,
        "error":           None,
    }

    def _progress(stage, pct, msg):
        if progress_callback:
            progress_callback(stage, pct, msg)

    try:
        # ── Stage 1: Data Profiling ───────────────────────────────────────────
        _progress("profiling", 5, "🔍 Profiling dataset…")
        profile = data_profiler.profile(df, target_col=target_col)

        if profile.get("task_type") == "unknown":
            result["error"] = (
                "Could not determine the task type. "
                "Please select a target column manually."
            )
            result["profile"] = profile
            return result

        result["profile"] = profile
        _progress("profiling", 20, f"✅ Profiling done — {profile['task_type'].title()} task detected")

        # ── Stage 2: Feature Engineering ─────────────────────────────────────
        _progress("engineering", 25, "🛠 Engineering features…")
        feat_plan = feature_engineer.engineer(df, profile, test_size=test_size)
        result["feature_plan"] = feat_plan
        _progress("engineering", 40, "✅ Features ready")

        # ── Stage 3: Model Training ───────────────────────────────────────────
        _progress("training", 45, "🏋 Training models with 5-fold CV…")

        def model_progress(idx, total, name):
            pct = 45 + int((idx / total) * 40)
            _progress("training", pct, f"🏋 Training [{idx+1}/{total}]: {name}")

        trainer_result = model_trainer.train(
            feat_plan["X_train"], feat_plan["y_train"],
            feat_plan["X_test"],  feat_plan["y_test"],
            task_type=profile["task_type"],
            cv_folds=cv_folds,
            progress_callback=model_progress,
        )
        result["trainer_result"] = trainer_result

        importance_dict = build_importance_dict(
            trainer_result["feature_importances_raw"],
            feat_plan["feature_names"],
        )
        result["importance_dict"] = importance_dict
        _progress("training", 85, f"✅ Best model: {trainer_result['best_model_name']} ({trainer_result['metric_name']}: {trainer_result['best_score']:.4f})")

        # ── Stage 4: Report Generation ────────────────────────────────────────
        _progress("reporting", 88, "📝 Generating AI report…")
        report_text = report_generator.generate(
            profile, feat_plan, trainer_result, importance_dict, dataset_name
        )
        result["report_text"] = report_text
        _progress("reporting", 100, "🎉 Pipeline complete!")

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"

    return result
