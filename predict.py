from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights
from model import VideoClassifier


NUM_FRAMES = 8

VIDEO_PATH = (
    Path(__file__).resolve().parent
    / "videos"
    / "test.avi"
)


def load_video(video_path, num_frames=8):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frames = []

    try:
        # 顺序解码，避免部分视频随机跳帧不准确
        while True:
            success, frame = cap.read()

            if not success:
                break

            # OpenCV 默认 BGR，转换成 RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError("视频中没有可读取的图像帧")

    # 从整个视频均匀选择 8 帧
    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames,
        dtype=int
    )

    selected_frames = [frames[i] for i in indices]

    # 使用 ResNet18 对应的 ImageNet 图像预处理
    preprocess = ResNet18_Weights.DEFAULT.transforms()

    processed_frames = []

    for frame in selected_frames:
        # [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(
            frame.copy()
        ).permute(2, 0, 1)

        tensor = preprocess(tensor)

        processed_frames.append(tensor)

    # [8, 3, 224, 224]
    video_tensor = torch.stack(processed_frames)

    # 添加 Batch 维度：[1, 8, 3, 224, 224]
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor, indices, len(frames)


if __name__ == "__main__":

    video_tensor, indices, total_frames = load_video(
        VIDEO_PATH,
        NUM_FRAMES
    )

    print("视频路径:", VIDEO_PATH)
    print("视频总帧数:", total_frames)
    print("采样帧索引:", indices)
    print("模型输入形状:", video_tensor.shape)
    print("数据类型:", video_tensor.dtype)
    print("视频预处理成功！")

    # 选择计算设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 读取训练好的模型检查点
    checkpoint_path = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "best_8frames.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    # 获取真实的类别映射
    class_to_idx = checkpoint["class_to_idx"]

    # 将类别映射反转：索引 -> 类别名称
    idx_to_class = {
        idx: name
        for name, idx in class_to_idx.items()
    }

    # 创建分类模型
    model = VideoClassifier(
        num_classes=len(class_to_idx)
    )

    # 加载训练得到的参数
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # 将视频张量移动到 GPU
    video_tensor = video_tensor.to(device)

    # 模型推理
    with torch.inference_mode():

        # 原始分类分数：[1, 4]
        logits = model(video_tensor)

        # Softmax：将分类分数转换成概率形式
        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    # 取概率最大的类别
    predicted_idx = probabilities.argmax().item()
    predicted_class = idx_to_class[predicted_idx]

    print("\n========== 视频动作识别结果 ==========")
    print("计算设备:", device)
    print("预测动作:", predicted_class)

    print("\n各类别的预测概率：")

    # 按预测概率从高到低排列
    sorted_indices = torch.argsort(
        probabilities,
        descending=True
    )

    for idx in sorted_indices.tolist():
        name = idx_to_class[idx]
        probability = probabilities[idx].item()

        print(f"{name:20s}: {probability:.2%}")