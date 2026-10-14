from pathlib import Path
import os
from collections import Counter
import csv
import time

import numpy as np
import torch

from model import VideoClassifier
from predict import load_video


# ==========================================
# 1. 项目配置
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = (
    PROJECT_DIR / "splits" / "test_manifest.csv"
)

CHECKPOINT_PATH = (
    PROJECT_DIR / "checkpoints" / "best_8frames.pth"
)

RESULT_PATH = (
    PROJECT_DIR / "results" / "evaluation_results.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. 加载模型
# ==========================================

def load_model():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    class_to_idx = checkpoint["class_to_idx"]

    model = VideoClassifier(
        num_classes=len(class_to_idx)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    num_frames = checkpoint.get("num_frames", 8)

    return model, class_to_idx, num_frames


# ==========================================
# 3. 批量评估
# ==========================================

def main():

    model, class_to_idx, num_frames = load_model()

    idx_to_class = {
        idx: name
        for name, idx in class_to_idx.items()
    }

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        test_records = list(csv.DictReader(f))

    print("计算设备:", DEVICE)
    print("测试视频数量:", len(test_records))
    print("类别映射:", class_to_idx)

    if len(test_records) != 149:
        raise ValueError(
            f"预期测试集为149个视频，实际为{len(test_records)}个"
        )

    results = []
    total = len(test_records)

    start_time = time.perf_counter()

    for i, record in enumerate(test_records, start=1):

        video_path = str(Path(os.environ.get("UCF101_DIR", "data/UCF-101")) / record["video_path"])
        true_label = record["true_label"]

        # 视频预处理：均匀提取8帧
        video_tensor, _, _ = load_video(
            video_path,
            num_frames=num_frames
        )

        video_tensor = video_tensor.to(DEVICE)

        # GPU推理
        with torch.inference_mode():

            logits = model(video_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            predicted_idx = int(
                probabilities.argmax().item()
            )

        predicted_label = idx_to_class[predicted_idx]

        confidence = float(
            probabilities[predicted_idx].item()
        )

        is_correct = predicted_label == true_label

        results.append({
            "video_path": video_path,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": int(is_correct)
        })

        status = "正确" if is_correct else "错误"

        print(
            f"[{i:3d}/{total}] "
            f"{true_label:15s} → "
            f"{predicted_label:15s} "
            f"{status}"
        )

    elapsed = time.perf_counter() - start_time

    # ======================================
    # 4. 计算总体准确率
    # ======================================

    correct_count = sum(
        row["correct"] for row in results
    )

    accuracy = correct_count / total

    print("\n========== 测试集评估结果 ==========")

    print(f"测试视频总数: {total}")
    print(f"正确识别数量: {correct_count}")
    print(f"错误识别数量: {total - correct_count}")
    print(f"测试准确率: {accuracy:.2%}")
    print(f"总评估时间: {elapsed:.2f} 秒")

    # ======================================
    # 5. 计算每个类别的召回率
    # ======================================

    print("\n========== 各类别召回率 ==========")

    class_names = [
        idx_to_class[i]
        for i in range(len(idx_to_class))
    ]

    for class_name in class_names:

        class_results = [
            row for row in results
            if row["true_label"] == class_name
        ]

        class_correct = sum(
            row["correct"] for row in class_results
        )

        class_total = len(class_results)

        recall = (
            class_correct / class_total
            if class_total > 0 else 0
        )

        print(
            f"{class_name:15s}: "
            f"{class_correct}/{class_total} "
            f"= {recall:.2%}"
        )

    # ======================================
    # 6. 计算混淆矩阵
    # ======================================

    confusion_matrix = np.zeros(
        (len(class_names), len(class_names)),
        dtype=int
    )

    for row in results:

        true_idx = class_to_idx[row["true_label"]]

        pred_idx = class_to_idx[
            row["predicted_label"]
        ]

        confusion_matrix[true_idx, pred_idx] += 1

    print("\n========== 混淆矩阵 ==========")
    print("行 = 真实类别；列 = 预测类别")
    print("类别顺序:", class_names)

    print(confusion_matrix)

    # ======================================
    # 7. 保存所有预测结果
    # ======================================

    with RESULT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "true_label",
                "predicted_label",
                "confidence",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n详细结果已保存:")
    print(RESULT_PATH)


if __name__ == "__main__":
    main()