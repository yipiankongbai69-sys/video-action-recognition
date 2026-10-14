from pathlib import Path
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 唯一识别错误的视频
VIDEO_PATH = Path(os.environ.get("ERROR_VIDEO_PATH", "data/UCF-101/PushUps/v_PushUps_g01_c02.avi"))

NUM_FRAMES = 8


def extract_sampled_frames(video_path, num_frames=8):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    frames = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

    finally:
        cap.release()

    if not frames:
        raise RuntimeError("没有读取到任何视频帧")

    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames,
        dtype=int
    )

    selected_frames = [
        frames[i] for i in indices
    ]

    return selected_frames, indices, len(frames)


def main():

    frames, indices, total = extract_sampled_frames(
        VIDEO_PATH,
        NUM_FRAMES
    )

    print("视频总帧数：", total)
    print("采样帧索引：", indices)

    # 创建 2 行 × 4 列的可视化图
    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 8)
    )

    for i, ax in enumerate(axes.flat):

        ax.imshow(frames[i])

        ax.set_title(
            f"Frame {i + 1} | Index {indices[i]}"
        )

        ax.axis("off")

    fig.suptitle(
        "Misclassified PushUps Video - 8 Sampled Frames",
        fontsize=15
    )

    plt.tight_layout()

    # 保存到项目目录
    output_path = (
        Path(__file__).resolve().parent
        / "error_frames.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    print("可视化图片已保存：", output_path)

    plt.show()


if __name__ == "__main__":
    main()