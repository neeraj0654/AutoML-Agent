"""
Agent 3 — Model Trainer
Trains multiple sklearn + XGBoost models with 5-fold CV and returns a leaderboard.
"""
import time
import numpy as np
from sklearn.linear_model  import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble      import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble      import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm           import SVC, SVR
from sklearn.neighbors     import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
)

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

RANDOM_STATE = 42


def _classification_models():
    models = {
        "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":          RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting":      GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM":                    SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=7),
    }
    if _XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, eval_metric="logloss",
            use_label_encoder=False, random_state=RANDOM_STATE, verbosity=0,
        )
    return models



def _regression_models():
    models = {
        "Linear Regression":      LinearRegression(),
        "Ridge Regression":       Ridge(alpha=1.0),
        "Random Forest":          RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting":      GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "SVR":                    SVR(kernel="rbf"),
        "K-Nearest Neighbors":    KNeighborsRegressor(n_neighbors=7),
    }
    if _XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=200, random_state=RANDOM_STATE, verbosity=0,
        )
    return models



def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    task_type: str,
    cv_folds: int = 5,
    progress_callback=None,
) -> dict:
    """
    Trains all models for the task type, returns:
        {
          "leaderboard": [{"model", "cv_score", "cv_std", "fit_time"}, ...],
          "best_model_name": str,
          "best_model": fitted sklearn estimator,
          "best_score": float,
          "metric_name": str,
          "feature_importances": dict | None,
          "test_metrics": dict,
        }
    """
    is_clf = task_type == "classification"
    models = _classification_models() if is_clf else _regression_models()
    scoring = "f1_weighted" if is_clf else "r2"
    metric_name = "F1 (weighted)" if is_clf else "R²"

    leaderboard = []
    total = len(models)

    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx, total, name)
        try:
            t0 = time.time()
            scores = cross_val_score(model, X_train, y_train,
                                     cv=cv_folds, scoring=scoring,
                                     n_jobs=-1, error_score="raise")
            fit_time = time.time() - t0
            leaderboard.append({
                "model":    name,
                "cv_score": float(scores.mean()),
                "cv_std":   float(scores.std()),
                "fit_time": fit_time,
            })
        except Exception as e:
            leaderboard.append({
                "model":    name,
                "cv_score": -999.0,
                "cv_std":   0.0,
                "fit_time": 0.0,
                "error":    str(e),
            })

    # ── Sort leaderboard ──────────────────────────────────────────────────────
    leaderboard.sort(key=lambda x: x["cv_score"], reverse=True)

    # ── Fit best model on full training set ──────────────────────────────────
    best_name  = leaderboard[0]["model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    # ── Test-set evaluation ───────────────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    if is_clf:
        test_metrics = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        }
    else:
        test_metrics = {
            "r2":   float(r2_score(y_test, y_pred)),
            "mae":  float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }

    # ── Feature importances ───────────────────────────────────────────────────
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        coef = best_model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        importances = np.abs(coef)

    return {
        "leaderboard":        leaderboard,
        "best_model_name":    best_name,
        "best_model":         best_model,
        "best_score":         leaderboard[0]["cv_score"],
        "metric_name":        metric_name,
        "feature_importances_raw": importances,
        "test_metrics":       test_metrics,
    }


def build_importance_dict(importances_raw, feature_names: list) -> dict | None:
    if importances_raw is None:
        return None
    n = min(len(importances_raw), len(feature_names))
    return {feature_names[i]: float(importances_raw[i]) for i in range(n)}
