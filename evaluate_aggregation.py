from pathlib import Path
import os
from collections import Counter
import csv

import torch

from model import VideoClassifier
from predict import load_video


PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    PROJECT_DIR / "checkpoints" / "best_8frames.pth"
)

MANIFEST_PATH = (
    PROJECT_DIR / "splits" / "test_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR / "results" / "aggregation_results.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    # 1. 加载模型
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {
        idx: name for name, idx in class_to_idx.items()
    }

    num_frames = checkpoint.get("num_frames", 8)

    model = VideoClassifier(
        num_classes=len(class_to_idx)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    # 2. 读取测试集
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        records = list(csv.DictReader(f))

    results = []

    print("计算设备:", DEVICE)
    print("测试视频数量:", len(records))

    # 3. 遍历全部测试视频
    for i, record in enumerate(records, start=1):

        video_path = str(Path(os.environ.get("UCF101_DIR", "data/UCF-101")) / record["video_path"])
        true_label = record["true_label"]

        video_tensor, _, _ = load_video(
            video_path,
            num_frames=num_frames
        )

        # [1, 8, 3, 224, 224] -> [8, 3, 224, 224]
        frames = video_tensor.squeeze(0).to(DEVICE)

        with torch.inference_mode():

            features = model.backbone(frames)

            frame_logits = model.classifier(features)

            frame_probs = torch.softmax(
                frame_logits,
                dim=1
            )

            # A. 平均 Logits
            mean_logits = frame_logits.mean(dim=0)
            pred_a = int(mean_logits.argmax().item())

            # B. 平均 Softmax
            mean_probs = frame_probs.mean(dim=0)
            pred_b = int(mean_probs.argmax().item())

            # C. 多数投票
            votes = frame_logits.argmax(dim=1).tolist()
            counts = Counter(votes)
            max_votes = max(counts.values())

            # 平票时，用平均 Logits 作为决胜规则
            tied_classes = [
                cls for cls, count in counts.items()
                if count == max_votes
            ]

            if len(tied_classes) == 1:
                pred_c = tied_classes[0]
            else:
                pred_c = max(
                    tied_classes,
                    key=lambda cls: mean_logits[cls].item()
                )

        result = {
            "video_path": video_path,
            "true_label": true_label,
            "mean_logits": idx_to_class[pred_a],
            "mean_softmax": idx_to_class[pred_b],
            "majority_vote": idx_to_class[pred_c]
        }

        results.append(result)

        print(
            f"[{i:3d}/{len(records)}] "
            f"A={idx_to_class[pred_a]:15s} "
            f"B={idx_to_class[pred_b]:15s} "
            f"C={idx_to_class[pred_c]:15s}"
        )

    # 4. 统计三种方法的准确率
    print("\n========== 聚合策略对比 ==========")

    for method in [
        "mean_logits",
        "mean_softmax",
        "majority_vote"
    ]:

        correct = sum(
            row[method] == row["true_label"]
            for row in results
        )

        accuracy = correct / len(results)

        print(
            f"{method:15s}: "
            f"{correct}/{len(results)} "
            f"= {accuracy:.2%}"
        )

    # 5. 找出策略之间预测不同的视频
    differences = [
        row for row in results
        if len({
            row["mean_logits"],
            row["mean_softmax"],
            row["majority_vote"]
        }) > 1
    ]

    print("\n策略预测不同的视频数:", len(differences))

    for row in differences:
        print(
            Path(row["video_path"]).name,
            "| 真实:", row["true_label"],
            "| A:", row["mean_logits"],
            "| B:", row["mean_softmax"],
            "| C:", row["majority_vote"]
        )

    # 6. 保存结果
    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "true_label",
                "mean_logits",
                "mean_softmax",
                "majority_vote"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n结果已保存:", OUTPUT_PATH)


if __name__ == "__main__":
    main()