from pathlib import Path
import os

import torch
import torch.nn as nn

from model import VideoClassifier
from predict import load_video


PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = Path(os.environ.get("ERROR_VIDEO_PATH", "data/UCF-101/PushUps/v_PushUps_g01_c02.avi"))

CHECKPOINT_PATH = (
    PROJECT_DIR / "checkpoints" / "best_8frames.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

model = VideoClassifier(
    num_classes=len(class_to_idx)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# 2. 提取与原始预测完全相同的8帧
video_tensor, indices, total = load_video(
    VIDEO_PATH,
    num_frames=8
)

# [1, 8, 3, 224, 224] -> [8, 3, 224, 224]
frames = video_tensor.squeeze(0).to(device)


# 3. 分别计算每一帧的分类分数
with torch.inference_mode():

    # ResNet18提取每帧512维特征
    features = model.backbone(frames)

    # 使用训练好的线性分类器
    frame_logits = model.classifier(features)

    # 各帧的Softmax分数
    frame_probs = torch.softmax(
        frame_logits,
        dim=1
    )

    # 平均Logits，复现视频分类过程
    mean_logits = frame_logits.mean(dim=0)

    video_probs = torch.softmax(
        mean_logits,
        dim=0
    )


# 4. 打印每帧的结果
class_names = [
    idx_to_class[i]
    for i in range(len(idx_to_class))
]

print("类别顺序:", class_names)

print("\n========== 每帧分类结果 ==========")

for i in range(8):

    probs = frame_probs[i]

    predicted_idx = int(probs.argmax().item())

    predicted_class = idx_to_class[predicted_idx]

    pushups_score = probs[class_to_idx["PushUps"]].item()
    rope_score = probs[class_to_idx["JumpRope"]].item()

    print(
        f"Frame {i+1} "
        f"(Index {indices[i]:2d}) | "
        f"预测: {predicted_class:15s} | "
        f"PushUps: {pushups_score:.2%} | "
        f"JumpRope: {rope_score:.2%}"
    )


print("\n========== 视频整体预测 ==========")

for i, name in enumerate(class_names):
    print(
        f"{name:15s}: "
        f"{video_probs[i].item():.2%}"
    )

# ======================================
# 比较三种视频分类聚合方法
# ======================================

from collections import Counter

with torch.inference_mode():

    # 方法 A：平均 Logits（原模型）
    method_a = frame_logits.mean(dim=0).argmax().item()

    # 方法 B：平均每帧 Softmax 分数
    method_b = frame_probs.mean(dim=0).argmax().item()

    # 方法 C：每帧单独预测，然后多数投票
    frame_predictions = frame_logits.argmax(dim=1).tolist()

    vote_counts = Counter(frame_predictions)

    method_c = vote_counts.most_common(1)[0][0]


print("\n========== 聚合方法对比 ==========")

print(
    "A. 平均 Logits:",
    idx_to_class[method_a]
)

print(
    "B. 平均 Softmax:",
    idx_to_class[method_b]
)

print(
    "C. 多数投票:",
    idx_to_class[method_c]
)

print("真实类别: PushUps")