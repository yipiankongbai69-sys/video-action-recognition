from pathlib import Path
import os
from collections import Counter
import csv

# ==========================================
# 1. 路径配置
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = Path(os.environ.get("UCF101_DIR", "data/UCF-101"))

SPLIT_FILE = PROJECT_DIR / "splits" / "testlist01.txt"

OUTPUT_FILE = PROJECT_DIR / "splits" / "test_manifest.csv"

# 与训练时相同的四种类别
SELECTED_CLASSES = {
    "PushUps",
    "JumpRope",
    "Biking",
    "PlayingGuitar"
}


# ==========================================
# 2. 读取并筛选官方测试列表
# ==========================================

def main():

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到原始数据集文件夹：{DATASET_DIR}"
        )

    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到官方测试列表：{SPLIT_FILE}"
        )

    records = []
    missing_files = []

    with SPLIT_FILE.open("r", encoding="utf-8-sig") as f:

        for line in f:

            relative_path = line.strip()

            if not relative_path:
                continue

            # 官方路径格式：PushUps/v_PushUps_g01_c01.avi
            parts = Path(relative_path).parts

            if len(parts) != 2:
                continue

            class_name = parts[0]

            if class_name not in SELECTED_CLASSES:
                continue

            video_path = DATASET_DIR / Path(relative_path)

            if video_path.is_file():

                records.append({
                    "video_path": relative_path.replace("\\", "/"),
                    "true_label": class_name
                })

            else:
                missing_files.append(str(video_path))

    # ======================================
    # 3. 统计结果
    # ======================================

    counts = Counter(
        row["true_label"] for row in records
    )

    print("\n===== UCF101 独立测试集检查 =====")

    for class_name in sorted(SELECTED_CLASSES):
        print(
            f"{class_name:15s}: "
            f"{counts[class_name]} 个视频"
        )

    print("--------------------------------")

    print("成功找到:", len(records))
    print("缺失视频:", len(missing_files))

    # ======================================
    # 4. 保存 CSV 测试清单
    # ======================================

    if missing_files:

        print("\n以下是缺失视频（最多显示前 10 个）：")

        for path in missing_files[:10]:
            print(path)

        print("\n测试集不完整，暂不生成 CSV。")
        return

    if len(records) != 149:
        raise ValueError(
            f"预期找到 149 个视频，实际找到 {len(records)} 个。"
            "请检查官方测试列表和数据集目录。"
        )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "true_label"]
        )

        writer.writeheader()
        writer.writerows(records)

    print("\n测试清单生成成功！")
    print("保存位置:", OUTPUT_FILE)


if __name__ == "__main__":
    main()