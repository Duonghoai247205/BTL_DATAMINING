import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR  = Path(__file__).parent.parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def load_raw_files(raw_dir: Path = RAW_DIR) -> dict:
    """
    Đọc cả 5 file CSV thô từ Kaggle.

    Returns
    -------
    dict với keys: 'yield_df', 'yield', 'pesticides', 'rainfall', 'temperature'
    """
    print("[loader] Đọc 5 file Kaggle...")

    yield_df = pd.read_csv(raw_dir / "yield_df.csv", index_col=0)
    yield_raw = pd.read_csv(raw_dir / "yield.csv")
    pesticides = pd.read_csv(raw_dir / "pesticides.csv")

    rainfall = pd.read_csv(raw_dir / "rainfall.csv")
    rainfall.columns = rainfall.columns.str.strip()
    rainfall["average_rain_fall_mm_per_year"] = pd.to_numeric(
        rainfall["average_rain_fall_mm_per_year"], errors="coerce"
    )

    temperature = pd.read_csv(raw_dir / "temperature.csv")
    temperature["avg_temp"] = pd.to_numeric(temperature["avg_temp"], errors="coerce")

    print(f"  yield_df    : {yield_df.shape[0]:,} rows")
    print(f"  yield (raw) : {yield_raw.shape[0]:,} rows")
    print(f"  pesticides  : {pesticides.shape[0]:,} rows")
    print(f"  rainfall    : {rainfall.shape[0]:,} rows")
    print(f"  temperature : {temperature.shape[0]:,} rows")

    return {
        "yield_df":    yield_df,
        "yield":       yield_raw,
        "pesticides":  pesticides,
        "rainfall":    rainfall,
        "temperature": temperature,
    }


def merge_datasets(raw: dict, year_min: int = 1990, year_max: int = 2013) -> pd.DataFrame:
    """
    Ghép 4 bảng thô thành 1 dataset đầy đủ (thay thế yield_df.csv nếu cần).

    Parameters
    ----------
    raw       : kết quả từ load_raw_files()
    year_min  : năm bắt đầu lọc
    year_max  : năm kết thúc lọc

    Returns
    -------
    DataFrame với các cột: Area, Item, Year, hg/ha_yield,
                            average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp
    """
    print(f"[loader] Ghép dataset ({year_min}–{year_max})...")

    # 1. Yield
    y = raw["yield"][["Area", "Item", "Year", "Value"]].copy()
    y.columns = ["Area", "Item", "Year", "hg/ha_yield"]

    # 2. Pesticides
    p = raw["pesticides"][["Area", "Year", "Value"]].copy()
    p.columns = ["Area", "Year", "pesticides_tonnes"]
    p = p.groupby(["Area", "Year"])["pesticides_tonnes"].mean().reset_index()

    # 3. Rainfall
    r = raw["rainfall"][["Area", "Year", "average_rain_fall_mm_per_year"]].copy()

    # 4. Temperature — trung bình năm
    t = raw["temperature"].copy()
    t.columns = ["Year", "Area", "avg_temp"]
    t_avg = t[t.Year.between(year_min, year_max)].groupby(
        ["Area", "Year"])["avg_temp"].mean().reset_index()

    # Ghép lần lượt
    df = y.merge(p, on=["Area", "Year"], how="left")
    df = df.merge(r, on=["Area", "Year"], how="left")
    df = df.merge(t_avg, on=["Area", "Year"], how="left")
    df = df[df.Year.between(year_min, year_max)].copy()
    df = df.dropna(subset=["hg/ha_yield"]).reset_index(drop=True)

    print(f"  Dataset sau ghép: {df.shape[0]:,} dòng × {df.shape[1]} cột")
    return df


def load_processed(filename: str, proc_dir: Path = PROC_DIR) -> pd.DataFrame:
    """
    Tải nhanh một file CSV đã xử lý từ data/processed/.

    Parameters
    ----------
    filename : tên file (vd: 'crop_yield_features.csv')
    """
    path = proc_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy chạy pipeline từ bước 01 trước."
        )
    df = pd.read_csv(path)
    print(f"[loader] Loaded '{filename}': {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df
