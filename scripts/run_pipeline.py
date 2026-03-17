import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks"

STEPS = {
    "01": "01_data_preparation.ipynb",
    "02": "02_eda.ipynb",
    "03": "03_preprocessing.ipynb",
    "04": "04_feature_engineering.ipynb",
    "05": "05_mining.ipynb",
    "06": "06_modeling_evaluation.ipynb",
}

STEP_NAMES = {
    "01": "Chuẩn bị & Ghép Dữ liệu",
    "02": "Phân tích Khám phá (EDA)",
    "03": "Tiền xử lý Dữ liệu",
    "04": "Kỹ thuật Đặc trưng",
    "05": "Khai phá Dữ liệu (Apriori + K-Means)",
    "06": "Mô hình hóa & Đánh giá",
}

def run_notebook(nb_path: Path, step_id: str):
    print(f"\n{'='*60}")
    print(f"  Bước {step_id}: {STEP_NAMES[step_id]}")
    print(f"  File: {nb_path.name}")
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            "--ExecutePreprocessor.kernel_name=python3",
            "--inplace",
            str(nb_path),
        ],
        capture_output=True, text=True, cwd=str(ROOT)
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓ Hoàn thành trong {elapsed:.1f}s")
    else:
        print(f"  ✗ LỖI sau {elapsed:.1f}s")
        print(result.stderr[-2000:] if result.stderr else "No error output")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Crop Yield Pipeline Runner")
    parser.add_argument("--step", default="all",
        help="Bước cần chạy: 01, 02, 03, 04, 05, 06, hoặc 'all'")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  ĐỀ TÀI 7 — DỰ BÁO NĂNG SUẤT CÂY TRỒNG")
    print("  Nhóm 3 | Kaggle Crop Yield Prediction Dataset")
    print("="*60)

    # Kiểm tra dữ liệu đầu vào
    raw_dir = ROOT / "data" / "raw"
    required_files = [
        "yield_df.csv", "yield.csv", "pesticides.csv",
        "rainfall.csv", "temperature.csv"
    ]
    missing = [f for f in required_files if not (raw_dir / f).exists()]
    if missing:
        print(f"\n✗ Thiếu file dữ liệu trong data/raw/:")
        for f in missing: print(f"    - {f}")
        print("\nHãy tải các file từ Kaggle và đặt vào thư mục data/raw/")
        sys.exit(1)
    print(f"\n✓ Tìm thấy {len(required_files)} file dữ liệu trong data/raw/")

    # Xác định các bước cần chạy
    if args.step == "all":
        steps_to_run = list(STEPS.keys())
    elif args.step in STEPS:
        steps_to_run = [args.step]
    else:
        print(f"✗ Bước '{args.step}' không hợp lệ. Chọn: all, 01, 02, 03, 04, 05, 06")
        sys.exit(1)

    print(f"\nSẽ chạy: {len(steps_to_run)} bước → {', '.join(steps_to_run)}")

    total_start = time.time()
    for step_id in steps_to_run:
        nb_path = NB_DIR / STEPS[step_id]
        if not nb_path.exists():
            print(f"✗ Không tìm thấy notebook: {nb_path}")
            sys.exit(1)
        run_notebook(nb_path, step_id)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ✓ PIPELINE HOÀN TẤT — {len(steps_to_run)} bước trong {total_time:.1f}s")
    print(f"  Kết quả lưu tại:")
    print(f"    data/processed/  — dữ liệu đã xử lý")
    print(f"    outputs/figures/ — biểu đồ PNG")
    print(f"    outputs/models/  — mô hình đã train")
    print(f"    outputs/reports/ — báo cáo JSON")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
