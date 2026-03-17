import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RPTS_DIR = Path(__file__).parent.parent.parent / "outputs" / "reports"
FIGS_DIR = Path(__file__).parent.parent.parent / "outputs" / "figures"
RPTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_by_crop(
    y_pred_log, y_true_log, y_true_raw,
    item_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Đánh giá MAE và R² theo từng loại cây trồng.

    Parameters
    ----------
    y_pred_log  : dự đoán (log scale)
    y_true_log  : thực tế (log scale)
    y_true_raw  : thực tế (hg/ha)
    item_labels : mảng tên loại cây tương ứng từng mẫu test

    Returns
    -------
    DataFrame: [Item, n, MAE, RMSE, R2]
    """
    y_pred_raw = np.expm1(y_pred_log)
    rows = []
    for crop in np.unique(item_labels):
        mask = item_labels == crop
        if mask.sum() < 5:
            continue
        mae  = mean_absolute_error(y_true_raw[mask], y_pred_raw[mask])
        rmse = np.sqrt(mean_squared_error(y_true_raw[mask], y_pred_raw[mask]))
        r2   = r2_score(y_true_log[mask], y_pred_log[mask])
        rows.append({"Item": crop, "n": mask.sum(), "MAE": mae, "RMSE": rmse, "R2": r2})
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    print("[evaluator] Kết quả theo loại cây:")
    print(df[["Item", "n", "MAE", "R2"]].to_string(index=False))
    return df


def evaluate_by_country(
    y_pred_log, y_true_log, y_true_raw,
    area_labels: np.ndarray, top_n: int = 15,
) -> pd.DataFrame:
    """
    Đánh giá MAE và R² theo quốc gia (top_n quốc gia tốt nhất và tệ nhất).
    """
    y_pred_raw = np.expm1(y_pred_log)
    rows = []
    for area in np.unique(area_labels):
        mask = area_labels == area
        if mask.sum() < 3:
            continue
        mae = mean_absolute_error(y_true_raw[mask], y_pred_raw[mask])
        r2  = r2_score(y_true_log[mask], y_pred_log[mask])
        rows.append({"Area": area, "n": mask.sum(), "MAE": mae, "R2": r2})
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    print(f"[evaluator] Top {top_n} quốc gia R² cao nhất:")
    print(df.head(top_n)[["Area", "n", "MAE", "R2"]].to_string(index=False))
    return df


def error_bands(y_pred_raw, y_true_raw, bands: list = None) -> dict:
    """
    Tính tỉ lệ dự báo nằm trong từng ngưỡng sai số tương đối.

    Parameters
    ----------
    bands : danh sách ngưỡng phần trăm (vd: [5, 10, 15, 20, 25, 30])

    Returns
    -------
    dict: {band_pct: within_pct}
    """
    if bands is None:
        bands = [5, 10, 15, 20, 25, 30]
    result = {}
    residuals = np.abs(y_true_raw - y_pred_raw) / (y_true_raw + 1)
    print("[evaluator] Tỉ lệ dự báo trong sai số x%:")
    for b in bands:
        within = np.mean(residuals < b / 100) * 100
        result[b] = float(within)
        print(f"  ±{b:2d}% : {within:.1f}%")
    return result


def timeseries_trend(df_raw: pd.DataFrame) -> dict:
    """
    Phân tích xu hướng chuỗi thời gian năng suất 1990–2013.

    Parameters
    ----------
    df_raw : DataFrame với cột Item, Year, hg/ha_yield

    Returns
    -------
    dict: {'overall_slope', 'overall_r2', 'by_crop': {crop: {slope, r2}}}
    """
    from sklearn.metrics import r2_score as _r2

    yearly = df_raw.groupby("Year")["hg/ha_yield"].mean()
    coef = np.polyfit(yearly.index, yearly.values, 1)
    overall_r2 = _r2(yearly.values, np.polyval(coef, yearly.index))

    by_crop = {}
    for crop in df_raw.Item.unique():
        sub = df_raw[df_raw.Item == crop].groupby("Year")["hg/ha_yield"].mean()
        if len(sub) < 5:
            continue
        c = np.polyfit(sub.index, sub.values, 1)
        r2c = _r2(sub.values, np.polyval(c, sub.index))
        by_crop[crop] = {"slope": float(c[0]), "r2": float(r2c)}

    print(f"[evaluator] Xu hướng tổng thể: {coef[0]:+.0f} hg/ha/năm (R²={overall_r2:.4f})")
    for crop, v in sorted(by_crop.items(), key=lambda x: -x[1]["slope"]):
        print(f"  {crop:<30}: {v['slope']:+7.0f} hg/ha/yr")

    return {
        "overall_slope": float(coef[0]),
        "overall_intercept": float(coef[1]),
        "overall_r2": float(overall_r2),
        "by_crop": by_crop,
    }


def generate_final_report(
    metrics_best: dict,
    cluster_result: dict,
    mining_result: dict,
    crop_eval: pd.DataFrame,
    timeseries: dict,
    dataset_info: dict,
    save: bool = True,
) -> dict:
    """
    Tạo báo cáo tổng kết pipeline dưới dạng JSON.

    Parameters
    ----------
    metrics_best   : dict {'R2', 'MAE', 'RMSE', 'MAPE'} của mô hình tốt nhất
    cluster_result : kết quả từ run_kmeans()
    mining_result  : kết quả từ run_apriori()
    crop_eval      : DataFrame từ evaluate_by_crop()
    timeseries     : dict từ timeseries_trend()
    dataset_info   : dict mô tả dataset (rows, features, years, ...)
    save           : có lưu file JSON không

    Returns
    -------
    dict báo cáo đầy đủ
    """
    report = {
        "dataset": dataset_info,
        "best_model": {
            "name":   "XGBoost (Tuned)",
            "R2":     round(metrics_best.get("R2", 0), 4),
            "MAE":    round(metrics_best.get("MAE", 0), 0),
            "RMSE":   round(metrics_best.get("RMSE", 0), 0),
            "MAPE":   round(metrics_best.get("MAPE", 0), 2),
        },
        "clustering": {
            "k":          cluster_result["k"],
            "silhouette": round(cluster_result["silhouette"], 4),
            "profiles":   cluster_result["profiles"],
        },
        "association_mining": {
            "total_rules":       len(mining_result["rules"]),
            "rules_high_yield":  len(mining_result["rules_high"]),
            "rules_low_yield":   len(mining_result["rules_low"]),
        },
        "timeseries": timeseries,
        "crop_evaluation": crop_eval.to_dict("records") if crop_eval is not None else [],
    }

    if save:
        path = RPTS_DIR / "final_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[evaluator] Báo cáo lưu tại: {path}")

    # In tóm tắt
    print("\n" + "="*55)
    print("  PIPELINE SUMMARY — Đề tài 7, Nhóm 3")
    print("="*55)
    print(f"  Dataset      : {dataset_info.get('rows',0):,} dòng × {dataset_info.get('features',0)} features")
    print(f"  Best Model   : XGBoost (Tuned)")
    print(f"  R²           : {report['best_model']['R2']}")
    print(f"  MAE          : {report['best_model']['MAE']:,.0f} hg/ha")
    print(f"  K (clusters) : {cluster_result['k']}  (Silhouette={cluster_result['silhouette']:.4f})")
    print(f"  Apriori rules: {len(mining_result['rules'])}")
    print(f"  Time trend   : {timeseries['overall_slope']:+.0f} hg/ha/năm")
    print("="*55)
    return report
