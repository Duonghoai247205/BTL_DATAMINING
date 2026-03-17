import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


MODELS_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _evaluate(name: str, y_pred_log, y_true_log, y_true_raw) -> dict:
    """Tính các metric: R², MAE, RMSE, MAPE."""
    y_pred_raw = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    r2   = r2_score(y_true_log, y_pred_log)
    mape = np.mean(np.abs((y_true_raw - y_pred_raw) / (y_true_raw + 1))) * 100
    print(f"  {name:<32}: R²={r2:.4f} | MAE={mae:9.0f} | RMSE={rmse:9.0f} | MAPE={mape:.2f}%")
    return {"model": name, "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape,
            "fitted_model": None}


def train_baseline_models(X_train, y_train, X_test, y_test, y_raw_test) -> list:
    """
    Huấn luyện 3 mô hình tuyến tính cơ bản: Linear, Ridge, Lasso.

    Returns
    -------
    list[dict]: mỗi dict chứa metrics và 'fitted_model'
    """
    print("\n[trainer] === BASELINE MODELS ===")
    results = []
    for name, model in [
        ("Linear Regression",  LinearRegression()),
        ("Ridge (α=10)",        Ridge(alpha=10)),
        ("Lasso (α=0.01)",      Lasso(alpha=0.01, max_iter=5000)),
    ]:
        model.fit(X_train, y_train)
        res = _evaluate(name, model.predict(X_test), y_test, y_raw_test)
        res["fitted_model"] = model
        results.append(res)
    return results


def train_ensemble_models(X_train, y_train, X_test, y_test, y_raw_test) -> list:
    """
    Huấn luyện Random Forest và Gradient Boosting.

    Returns
    -------
    list[dict]: mỗi dict chứa metrics và 'fitted_model'
    """
    print("\n[trainer] === ENSEMBLE MODELS ===")
    results = []
    for name, model in [
        ("Random Forest (200)",    RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1)),
        ("Gradient Boosting",      GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        res = _evaluate(name, model.predict(X_test), y_test, y_raw_test)
        res["fitted_model"] = model
        results.append(res)
    return results


def train_xgboost_tuned(
    X_train, y_train, X_test, y_test, y_raw_test,
    n_iter: int = 20, cv: int = 5,
) -> dict:
    """
    Huấn luyện XGBoost với RandomizedSearchCV (hyperparameter tuning).

    Parameters
    ----------
    n_iter : số lần thử ngẫu nhiên
    cv     : số fold cross-validation

    Returns
    -------
    dict chứa: metrics, 'fitted_model', 'best_params', 'cv_scores'
    """
    print("\n[trainer] === XGBOOST TUNED ===")
    param_dist = {
        "n_estimators":     [200, 300, 500, 700],
        "max_depth":        [4, 5, 6, 8],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [1.0, 2.0, 5.0],
        "min_child_weight": [1, 3, 5],
    }
    base = xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist")
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=cv,
        scoring="r2", random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    cv_scores = cross_val_score(best, X_train, y_train, cv=cv, scoring="r2")
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    res = _evaluate("XGBoost (Tuned)", best.predict(X_test), y_test, y_raw_test)
    res["fitted_model"] = best
    res["best_params"]  = search.best_params_
    res["cv_scores"]    = cv_scores.tolist()

    # Lưu model
    save_path = MODELS_DIR / "xgb_best_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(best, f)
    print(f"  Model saved → {save_path}")
    return res


def train_all_models(
    X_train, y_train, X_test, y_test, y_raw_test,
    xgb_n_iter: int = 20,
) -> pd.DataFrame:
    """
    Chạy toàn bộ 6 mô hình và trả về bảng so sánh.

    Returns
    -------
    DataFrame sắp xếp theo R² giảm dần
    """
    all_results = []
    all_results += train_baseline_models(X_train, y_train, X_test, y_test, y_raw_test)
    all_results += train_ensemble_models(X_train, y_train, X_test, y_test, y_raw_test)
    all_results.append(train_xgboost_tuned(X_train, y_train, X_test, y_test, y_raw_test, n_iter=xgb_n_iter))

    df_cmp = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("fitted_model", "best_params", "cv_scores")}
        for r in all_results
    ]).sort_values("R2", ascending=False).reset_index(drop=True)

    print("\n[trainer] === KẾT QUẢ SO SÁNH ===")
    print(df_cmp[["model", "R2", "MAE", "RMSE", "MAPE"]].to_string(index=False))
    return df_cmp, all_results
