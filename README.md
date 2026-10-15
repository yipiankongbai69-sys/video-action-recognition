# Video Action Recognition — ResNet18 + Temporal Average Pooling

A PyTorch prototype for **four selected UCF101 classes**: `PushUps`, `JumpRope`, `Biking`, `PlayingGuitar`. Built as an independent learning and experimental analysis project. **This is not a 101-class UCF101 benchmark result.**

## Approach

`Video → uniform frame sampling (8) → ResNet18 (512-D feature per frame) → temporal mean pooling → linear classifier → four logits`

The backbone uses pretrained ImageNet ResNet18 weights during initial training; during local inference, the full trained checkpoint restores the backbone and classifier. The final linear layer has 2,052 trainable parameters. The mean-pooling model **does not explicitly model frame order**.

## Results

| Frame count | Selected-checkpoint validation accuracy (81 videos) | Validation loss | Model-only forward latency |
|---|---:|---:|---:|
| 1 | 95.06% | 0.1989 | 4.84 ms |
| 4 | 95.06% | 0.1872 | 9.49 ms |
| 8 | 96.30% | 0.1676 | 15.60 ms |

On a **149-video held-out four-class test set**, the eight-frame Mean Logits classifier predicted **148/149 (99.33%)** correctly. A subsequent **exploratory, test-set-informed** comparison obtained 148/149 with Mean Softmax and 149/149 with Majority Vote. Because voting was investigated *after* inspecting test errors, 149/149 is **not** an unbiased, independently confirmed improvement.

The latency figures measure **model forward execution**, not full decoding, preprocessing or user-interface time. They are environment-specific. Frame-count runs were not repeated across multiple random seeds.

## Requirements and installation

### Verified local environment (Windows / Conda)

The GitHub-cloned project successfully started Gradio and performed video inference on the original working Conda environment with:

| Component | Observed version |
|---|---|
| Python | 3.11.17 |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| Gradio | 6.30.0 |
| OpenCV | 5.0.0 |
| CUDA available | True |

This verifies local inference in **an existing environment**, not a clean installation from scratch.

### Creating a new Conda environment (not yet independently validated)

```powershell
conda env create -f environment.yml
conda activate video-action-repro

# For an NVIDIA CUDA 12.8 PyTorch wheel build:
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# Install the remaining application requirements:
python -m pip install -r requirements.txt
```

The commands above aim to match the observed local PyTorch/CUDA build, but installation depends on platform and available official wheels. For a CPU-only or different CUDA setup, consult the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) and choose compatible torch/torchvision packages instead. Run `python test_environment.py` before launching `python app.py`.

The earlier `torch>=2.2` installation error occurred in a separately created `venv`; its root cause has not been established. An already-installed compatible Conda environment was used to verify the app.


## Model checkpoint

The trained eight-frame checkpoint is available as a separate GitHub Release asset:

- **Release:** [v1.0.0 — Initial ResNet18 Model](https://github.com/yipiankongbai69-sys/video-action-recognition/releases/tag/v1.0.0)
- **Model file:** [best_8frames.pth](https://github.com/yipiankongbai69-sys/video-action-recognition/releases/download/v1.0.0/best_8frames.pth) (approximately 42.7 MiB)

Download the file and place it at `checkpoints/best_8frames.pth` (next to `checkpoints/README.md`). The app loads it from this path at startup. The checkpoint should contain `model_state_dict` and `class_to_idx` and may include `num_frames`.

The `.pth` file is intentionally **not committed to Git history**. Use only trusted checkpoints; this model file has been uploaded as a Release asset. End-to-end inference has succeeded in the original working Conda environment; fresh-install verification remains outstanding.

## Run interactive local demo

```bash
python app.py
```

Upload a video and choose **Mean Logits**, **Mean Softmax**, or **Majority Vote**. A CPU fallback is available when CUDA is unavailable.

## Reproduce evaluation with your own UCF101 dataset

The actual UCF101 video dataset is **not provided**. Set `UCF101_DIR` to your local root directory containing class folders:

**PowerShell**:
```powershell
$env:UCF101_DIR = 'D:\Datasets\UCF-101'
python prepare_testset.py
python evaluate.py
python evaluate_aggregation.py
```

**Linux / macOS**:
```bash
export UCF101_DIR='/path/to/UCF-101'
python prepare_testset.py
python evaluate.py
python evaluate_aggregation.py
```

The manifest `splits/test_manifest.csv` uses paths relative to `UCF101_DIR`. The official UCF101 test split listing is in `splits/testlist01.txt`. The commands above require the exact UCF101 dataset layout and a checkpoint for the four classes.

For the original misclassified clip, set `ERROR_VIDEO_PATH` to its absolute path before running `analyze_frame_scores.py` or `visualize_error.py`.

## Repository guide

- `model.py`: ResNet18 frame encoder, temporal averaging and classifier
- `predict.py`: RGB video decoding, sampling, preprocessing and inference
- `app.py`: Gradio interface and alternative aggregation rules
- `prepare_testset.py`: builds the selected-class evaluation manifest
- `evaluate.py`: per-video evaluation and summary metrics
- `evaluate_aggregation.py`: comparison of three video-level aggregation rules
- `analyze_frame_scores.py`, `analyze_errors.py`, `visualize_error.py`: error analysis
- `results/`: anonymised/portable experiment outputs, without machine-specific paths

## Limitations and next steps

This study uses only four of the 101 action classes, with a relatively small validation/test set. Temporal average pooling discards frame order. One-frame vs multiple-frame sampling also changes selected temporal positions. Further work includes multiple random seeds, paired per-video tests, temporal models, sampling ablations and true end-to-end latency evaluation.

## Data and attribution

UCF101 is an external research dataset: obtain it from its official source and comply with its terms. This repository excludes video data and does not claim rights to redistribute it. Torchvision / PyTorch and other dependencies retain their original licenses.
