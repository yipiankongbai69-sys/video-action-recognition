from pathlib import Path
from collections import Counter

import gradio as gr
import torch

from model import VideoClassifier
from predict import load_video


# ==========================================
# 1. 加载模型
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    PROJECT_DIR / "checkpoints" / "best_8frames.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

class_to_idx = checkpoint["class_to_idx"]

idx_to_class = {
    idx: name
    for name, idx in class_to_idx.items()
}

NUM_FRAMES = checkpoint.get("num_frames", 8)

model = VideoClassifier(
    num_classes=len(class_to_idx)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("模型加载成功")
print("运行设备:", DEVICE)
print("类别映射:", class_to_idx)


# ==========================================
# 2. 视频预测
# ==========================================

def predict_action(video_path, aggregation_method):

    if video_path is None:
        raise gr.Error("请先上传视频。")

    video_tensor, indices, total_frames = load_video(
        video_path,
        num_frames=NUM_FRAMES
    )

    # [1, 8, 3, 224, 224] → [8, 3, 224, 224]
    frames = video_tensor.squeeze(0).to(DEVICE)

    with torch.inference_mode():

        # 每一帧分别提取特征
        features = model.backbone(frames)

        # 每帧四个类别的原始分类分数
        frame_logits = model.classifier(features)

        # 每帧的 Softmax
        frame_probs = torch.softmax(
            frame_logits,
            dim=1
        )

        # A：平均 Logits
        mean_logits = frame_logits.mean(dim=0)
        logits_scores = torch.softmax(
            mean_logits,
            dim=0
        )

        # B：平均 Softmax
        softmax_scores = frame_probs.mean(dim=0)

        # C：多数投票
        frame_predictions = (
            frame_logits.argmax(dim=1).tolist()
        )

        vote_counts = Counter(frame_predictions)

        vote_scores = torch.tensor(
            [
                vote_counts.get(i, 0) / NUM_FRAMES
                for i in range(len(class_to_idx))
            ],
            dtype=torch.float32
        )

        max_votes = max(vote_counts.values())

        tied_classes = [
            index
            for index, count in vote_counts.items()
            if count == max_votes
        ]

        # 平票时使用平均 Logits 决胜
        if len(tied_classes) == 1:
            vote_winner = tied_classes[0]
        else:
            vote_winner = max(
                tied_classes,
                key=lambda idx: mean_logits[idx].item()
            )

    # ======================================
    # 3. 选择聚合策略
    # ======================================

    if aggregation_method == "Mean Logits":
        scores = logits_scores
        predicted_idx = int(scores.argmax().item())
        score_type = "Softmax 分数"

    elif aggregation_method == "Mean Softmax":
        scores = softmax_scores
        predicted_idx = int(scores.argmax().item())
        score_type = "平均 Softmax 分数"

    else:
        scores = vote_scores
        predicted_idx = vote_winner
        score_type = "投票占比"

    # Gradio Label 需要的字典
    result_scores = {
        idx_to_class[i]: float(scores[i].item())
        for i in range(len(class_to_idx))
    }

    predicted_class = idx_to_class[predicted_idx]

    # ======================================
    # 4. 每帧预测详情
    # ======================================

    frame_details = []

    for i, pred_idx in enumerate(frame_predictions):

        frame_name = idx_to_class[pred_idx]

        frame_details.append(
            {
                "帧序号": i + 1,
                "原视频帧索引": int(indices[i]),
                "预测动作": frame_name,
                "PushUps": round(
                    frame_probs[
                        i, class_to_idx["PushUps"]
                    ].item(),
                    4
                ),
                "JumpRope": round(
                    frame_probs[
                        i, class_to_idx["JumpRope"]
                    ].item(),
                    4
                ),
                "Biking": round(
                    frame_probs[
                        i, class_to_idx["Biking"]
                    ].item(),
                    4
                ),
                "PlayingGuitar": round(
                    frame_probs[
                        i, class_to_idx["PlayingGuitar"]
                    ].item(),
                    4
                )
            }
        )

    # 以 Markdown 表格显示每帧预测
    table_lines = [
        "| 帧序号 | 原帧索引 | 预测动作 | PushUps | JumpRope | Biking | PlayingGuitar |",
        "|---|---:|---|---:|---:|---:|---:|"
    ]

    for row in frame_details:
        table_lines.append(
            f"| {row['帧序号']} "
            f"| {row['原视频帧索引']} "
            f"| {row['预测动作']} "
            f"| {row['PushUps']:.2%} "
            f"| {row['JumpRope']:.2%} "
            f"| {row['Biking']:.2%} "
            f"| {row['PlayingGuitar']:.2%} |"
        )

    frame_table = "\n".join(table_lines)

    # ======================================
    # 5. 输出信息
    # ======================================

    info = (
        f"**最终预测：{predicted_class}**\n\n"
        f"聚合策略：{aggregation_method}\n\n"
        f"显示分数类型：{score_type}\n\n"
        f"视频总帧数：{total_frames}\n\n"
        f"采样帧数：{NUM_FRAMES}\n\n"
        f"运行设备：{DEVICE}"
    )

    return result_scores, info, frame_table


# ==========================================
# 6. 创建 Gradio 界面
# ==========================================

with gr.Blocks(
    title="Video Action Recognition"
) as demo:

    gr.Markdown(
        """
        # 视频动作识别与聚合策略对比系统

        **模型：** ResNet18 + Temporal Mean Pooling

        **计算框架：** PyTorch

        **支持动作：**
        PushUps、JumpRope、Biking、PlayingGuitar

        可以选择不同的聚合策略，
        比较它们对相同视频的预测结果。
        """
    )

    with gr.Row():

        video_input = gr.Video(
            label="上传视频",
            sources=["upload"],
            format=None
        )

        prediction_output = gr.Label(
            label="动作识别结果",
            num_top_classes=4
        )

    aggregation_selector = gr.Radio(
        choices=[
            "Mean Logits",
            "Mean Softmax",
            "Majority Vote"
        ],
        value="Mean Logits",
        label="聚合策略"
    )

    predict_button = gr.Button(
        "开始识别",
        variant="primary"
    )

    info_output = gr.Markdown()

    gr.Markdown("### 8 帧的单独预测结果")

    frame_output = gr.Markdown()

    predict_button.click(
        fn=predict_action,
        inputs=[
            video_input,
            aggregation_selector
        ],
        outputs=[
            prediction_output,
            info_output,
            frame_output
        ]
    )


# ==========================================
# 7. 启动应用
# ==========================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )