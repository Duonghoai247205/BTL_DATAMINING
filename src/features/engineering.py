import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Điều kiện khí hậu tối ưu cho từng loại cây (nguồn: FAO)
CROP_OPT = {
    "Maize":               {"rain": 700,  "temp": 22},
    "Wheat":               {"rain": 450,  "temp": 15},
    "Rice, paddy":         {"rain": 1500, "temp": 27},
    "Soybeans":            {"rain": 600,  "temp": 24},
    "Potatoes":            {"rain": 550,  "temp": 17},
    "Cassava":             {"rain": 1200, "temp": 28},
    "Sorghum":             {"rain": 500,  "temp": 28},
    "Sweet potatoes":      {"rain": 900,  "temp": 25},
    "Plantains and others":{"rain": 1800, "temp": 26},
    "Yams":                {"rain": 1200, "temp": 26},
}


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng tương tác giữa biến khí hậu."""
    df = df.copy()
    df["rain_x_temp"]     = df["average_rain_fall_mm_per_year"] * df["avg_temp"]
    df["pest_x_rain"]     = np.log1p(df["pesticides_tonnes"]) * np.log1p(df["average_rain_fall_mm_per_year"])
    df["pest_x_temp"]     = np.log1p(df["pesticides_tonnes"]) * df["avg_temp"]
    df["rain_per_temp"]   = df["average_rain_fall_mm_per_year"] / (df["avg_temp"].abs() + 1)
    df["temp_rain_ratio"] = df["avg_temp"] / (np.log1p(df["average_rain_fall_mm_per_year"]) + 1)
    return df


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng đa thức (bình phương, căn bậc hai)."""
    df = df.copy()
    df["rain_sq"]  = df["average_rain_fall_mm_per_year"] ** 2
    df["temp_sq"]  = df["avg_temp"] ** 2
    df["pest_sq"]  = np.log1p(df["pesticides_tonnes"]) ** 2
    df["rain_sqrt"]= np.sqrt(df["average_rain_fall_mm_per_year"].clip(0))
    df["temp_abs"] = (df["avg_temp"] - df["avg_temp"].mean()).abs()
    return df


def add_aggregated_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê tổng hợp theo loại cây và quốc gia."""
    df = df.copy()

    for col in ["average_rain_fall_mm_per_year", "avg_temp", "pesticides_tonnes"]:
        short = col[:4]
        stats = df.groupby("Item")[col].agg(["mean", "std"]).rename(
            columns={"mean": f"{short}_crop_mean", "std": f"{short}_crop_std"}
        )
        df = df.merge(stats, on="Item", how="left")

    for col in ["average_rain_fall_mm_per_year", "avg_temp"]:
        short = col[:4]
        stats = df.groupby("Area")[col].agg(["mean", "std"]).rename(
            columns={"mean": f"{short}_area_mean", "std": f"{short}_area_std"}
        )
        df = df.merge(stats, on="Area", how="left")

    df["yield_vs_crop_mean"] = df.groupby("Item")["hg/ha_yield"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1)
    )
    return df


def add_grow_score(df: pd.DataFrame, crop_opt: dict = None) -> pd.DataFrame:
    """
    Tính điểm sinh trưởng tối ưu dựa trên khoảng cách đến điều kiện lý tưởng.
    grow_score ∈ (0, 1]: càng gần 1 → điều kiện canh tác càng tốt.
    """
    if crop_opt is None:
        crop_opt = CROP_OPT
    df = df.copy()

    def _grow(row):
        opt = crop_opt.get(row["Item"], {"rain": 800, "temp": 22})
        d_r = abs(row["average_rain_fall_mm_per_year"] - opt["rain"])
        d_t = abs(row["avg_temp"] - opt["temp"])
        return np.exp(-d_r / 500) * np.exp(-d_t / 8)

    df["grow_score"]        = df.apply(_grow, axis=1)
    df["rain_dev_from_opt"] = df.apply(
        lambda r: abs(r["average_rain_fall_mm_per_year"] - crop_opt.get(r["Item"], {"rain": 800})["rain"]), axis=1
    )
    df["temp_dev_from_opt"] = df.apply(
        lambda r: abs(r["avg_temp"] - crop_opt.get(r["Item"], {"temp": 22})["temp"]), axis=1
    )
    return df


def add_stress_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các chỉ số stress nông nghiệp (nhiệt, hạn, lũ, sương giá)."""
    df = df.copy()
    df["heat_stress"]    = (df["avg_temp"] > 32).astype(int)
    df["drought_stress"] = (df["average_rain_fall_mm_per_year"] < 250).astype(int)
    df["flood_risk"]     = (df["average_rain_fall_mm_per_year"] > 2500).astype(int)
    df["frost_risk"]     = (df["avg_temp"] < 5).astype(int)
    df["stress_count"]   = (
        df["heat_stress"] + df["drought_stress"] + df["flood_risk"] + df["frost_risk"]
    )
    df["is_high_yield_crop"] = df["Item"].isin(
        ["Potatoes", "Cassava", "Sweet potatoes", "Yams", "Plantains and others"]
    ).astype(int)
    return df


def add_time_features(df: pd.DataFrame, base_year: int = 1990) -> pd.DataFrame:
    """Thêm đặc trưng thời gian (year_norm, decade, is_modern, ...)."""
    df = df.copy()
    df["years_since_base"] = df["Year"] - base_year
    df["year_norm"]        = (df["Year"] - base_year) / (2013 - base_year)
    df["decade"]           = (df["Year"] // 10) * 10
    df["is_modern"]        = (df["Year"] >= 2005).astype(int)
    df["decade_2000s"]     = (df["decade"] == 2000).astype(int)
    df["decade_2010s"]     = (df["decade"] == 2010).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame, scale: bool = True):
    """
    Chạy toàn bộ pipeline feature engineering và trả về (X, y, feature_names, scaler).

    Parameters
    ----------
    df    : DataFrame đã qua preprocessing (crop_yield_processed.csv)
    scale : có chuẩn hóa StandardScaler không

    Returns
    -------
    X            : np.ndarray (n_samples, n_features)
    y            : np.ndarray log-transformed yield
    feature_cols : list tên đặc trưng
    scaler       : StandardScaler đã fit (hoặc None nếu scale=False)
    """
    df = add_interaction_features(df)
    df = add_polynomial_features(df)
    df = add_aggregated_stats(df)
    df = add_grow_score(df)
    df = add_stress_indicators(df)
    df = add_time_features(df)

    # Log transforms
    df["log_pesticides"] = np.log1p(df["pesticides_tonnes"])
    df["log_rain"]       = np.log1p(df["average_rain_fall_mm_per_year"])

    exclude = {
        "Area", "Item", "hg/ha_yield", "log_yield",
        "yield_cat", "rain_cat", "temp_cat", "pest_cat",
        "Area_enc", "Item_enc",
    }
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]

    # Fill NaN
    for c in feature_cols:
        df[c] = df[c].fillna(df[c].median())

    X = df[feature_cols].values
    y = df["log_yield"].values if "log_yield" in df.columns else np.log1p(df["hg/ha_yield"].values)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, feature_cols, scaler
