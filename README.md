# Video Action Recognition — ResNet18 + Temporal Average Pooling

A PyTorch prototype for **four selected UCF101 classes**: PushUps, JumpRope, Biking, PlayingGuitar. Independent learning and experimental analysis project; **not a 101-class benchmark**.

## Method
Video → 8 uniformly sampled frames → pretrained ResNet18 frame encoder → temporal mean pooling → linear four-class classifier (2,052 trainable parameters). The current model does **not** explicitly model temporal order.

## Results

| Frames | Validation accuracy (81 videos) | Validation loss | Model-only latency |
|---|---:|---:|---:|
| 1 | 95.06% | 0.1989 | 4.84 ms |
| 4 | 95.06% | 0.1872 | 9.49 ms |
| 8 | 96.30% | 0.1676 | 15.60 ms |

Eight-frame Mean Logits accuracy on the **149-video four-class held-out test set**: **148/149 (99.33%)**. A post-hoc aggregation exploration gave Mean Softmax 148/149 and Majority Vote 149/149; the latter is **not an independently validated improvement**, because it was tried after examining test errors.

Model-only latency excludes video decoding and preprocessing. Frame-count experiments were not repeated across multiple random seeds.

## Installation

Python 3.10+ with compatible PyTorch and Torchvision versions.

```bash
python -m pip install -r requirements.txt
```

## Checkpoint and local demo

Place a **trusted** training checkpoint at `checkpoints/best_8frames.pth`. The public-ready project intentionally excludes the approximately 45 MB checkpoint and UCF101 videos. The checkpoint dictionary should contain `model_state_dict` and `class_to_idx`.

```bash
python app.py
```

Upload a video and choose Mean Logits, Mean Softmax, or Majority Vote.

## Evaluation

Obtain UCF101 separately. Set `UCF101_DIR` to your local UCF-101 root directory and run:

```bash
python prepare_testset.py
python evaluate.py
python evaluate_aggregation.py
```

The repository includes relative-path manifests and the official test split listing; it does not redistribute UCF101 video files. Set `ERROR_VIDEO_PATH` for the original misclassified clip when using `analyze_frame_scores.py` or `visualize_error.py`.

## Files

- `model.py` — model architecture and checkpoint loading
- `predict.py` — video preprocessing and model inference
- `app.py` — Gradio UI and aggregation strategies
- `prepare_testset.py` — four-class test manifest
- `evaluate.py` and `evaluate_aggregation.py` — model evaluation
- `analyze_frame_scores.py`, `analyze_errors.py`, `visualize_error.py` — error analysis
- `results/` — evaluation tables without local absolute paths

## Limitations

Only four UCF101 classes, limited held-out sample size, no multi-seed repeat, frame-order-invariant mean pooling, and sampling position differences across frame counts. Future work: repeated experiments, paired predictions, improved temporal modelling and end-to-end latency analysis.

## Attribution

UCF101 remains an external research dataset; obtain it from the original source and observe its terms. PyTorch, Torchvision and other packages retain their respective licenses.
