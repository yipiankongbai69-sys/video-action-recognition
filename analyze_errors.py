from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent

RESULT_FILE = PROJECT_DIR / "evaluation_results.csv"

with RESULT_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    results = list(csv.DictReader(f))

errors = [
    row for row in results
    if row["correct"] == "0"
]

print("========== 错误样本分析 ==========")
print("错误样本数量:", len(errors))

for i, row in enumerate(errors, start=1):
    print(f"\n错误样本 {i}")
    print("视频路径:", row["video_path"])
    print("真实类别:", row["true_label"])
    print("预测类别:", row["predicted_label"])
    print("最高 Softmax 分数:", f"{float(row['confidence']):.2%}")