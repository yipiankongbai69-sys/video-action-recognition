from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18


class VideoClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # 1. ResNet18：提取每一帧的图像特征
        self.backbone = resnet18(weights=None)

        # 删除原来的 1000 类图像分类层
        # 保留 ResNet18 输出的 512 维特征
        self.backbone.fc = nn.Identity()

        # 2. 我们自己训练的四分类层
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x 的形状：[B, T, C, H, W]
        B, T, C, H, W = x.shape

        # 将视频帧展开，分别交给 ResNet18
        x = x.reshape(B * T, C, H, W)

        features = self.backbone(x)

        # 还原视频维度：[B, T, 512]
        features = features.reshape(B, T, 512)

        # 时间平均池化：将 8 帧特征合并
        video_features = features.mean(dim=1)

        # 输出四种动作的分类分数
        logits = self.classifier(video_features)

        return logits


if __name__ == "__main__":

    # 选择 GPU；如果没有 CUDA，则使用 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "best_8frames.pth"
    )

    # 加载本地训练得到的检查点
    # 仅加载自己信任的模型文件
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    # 从检查点读取类别映射
    class_to_idx = checkpoint["class_to_idx"]

    print("类别映射:", class_to_idx)
    print("视频帧数:", checkpoint.get("num_frames", "未记录"))

    # 创建与训练时相同的网络结构
    model = VideoClassifier(
        num_classes=len(class_to_idx)
    )

    # 加载训练得到的参数
    model.load_state_dict(checkpoint["model_state_dict"])

    # 将模型移动到 GPU，并切换到推理模式
    model = model.to(device)
    model.eval()

    print("模型权重加载成功！")
    print("运行设备:", device)

    # 生成一个模拟的 8 帧视频输入
    test_input = torch.randn(
        1, 8, 3, 224, 224,
        device=device
    )

    # 不计算梯度，节省显存
    with torch.inference_mode():
        output = model(test_input)

    print("模型输出形状:", output.shape)
    print("测试完成！")