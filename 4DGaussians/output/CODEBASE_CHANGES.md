# 4DGS Codebase Changes for Assignment 3

This document describes all modifications and additions made to the [4DGaussians](https://github.com/hustvl/4DGaussians) codebase for the temporal interpolation evaluation on the `coffee_martini` scene.

---

## 1. Modifications to Upstream Code

These changes adapt the original 4DGS training pipeline to support temporal subsampling (variable frame gap k) and correct dataset handling for the `coffee_martini` scene.

### `arguments/__init__.py`

**Change:** Added `temporal_gap` parameter (default 1) to `ModelParams`.

This parameter controls the temporal subsampling during training. When `temporal_gap=k`, only every k-th frame is used for training, enabling the temporal interpolation experiments at k ∈ {2, 4, 8}.

### `scene/__init__.py`

**Change:** Passed `args.temporal_gap` through to the dynerf scene loader.

The original code called `readdynerfInfo(args.source_path, args.white_background, args.eval)` without any temporal subsampling control. The modified version forwards `args.temporal_gap` so the dataset reader knows which frames to load.

### `scene/dataset_readers.py`

**Changes:**
- Added `temporal_gap` parameter to `readdynerfInfo()`.
- Changed the downsample factor from `1.0` to `2.0` — this sets the training resolution to 676 × 507 (4× downsampled from the native 2704 × 2028), which the original DyNeRF dataset pipeline expects.
- Passed `temporal_gap` through to `Neural3D_NDC_Dataset`.

### `scene/neural_3D_dataset_NDC.py`

**Changes (the core temporal subsampling logic):**

1. **Added `temporal_gap` parameter** to the `Neural3D_NDC_Dataset` constructor.

2. **Reduced `countss` from 300 to 50.** The original code assumed 300 frames per video and selected every 6th frame to get 50 usable frames. The `coffee_martini` MP4 videos contain exactly 300 frames, but only the first 50 are the actual sequence — the rest are duplicates/padding. Setting `countss=50` extracts all 50 sequential frames directly.

3. **Added temporal subsampling filter:** For the `"train"` split, frames where `idx % self.temporal_gap != 0` are skipped. This means for `temporal_gap=8`, only frames {0, 8, 16, 24, 32, 40, 48} are loaded for training. The `"test"` split loads all frames regardless (cam00 is the held-out test camera).

4. **Fixed timestamp normalization:** Changed `image_times.append(idx/countss)` to `image_times.append(idx/(countss - 1))`. The original normalization mapped frame 0 → 0.0 and frame 49 → 49/300 ≈ 0.163, wasting most of the [0,1] time range. The fix maps frame 0 → 0.0 and frame 49 → 1.0, using the full temporal range.

5. **Fixed `N_time` tracking:** Changed `N_time = len(images_path)` to `N_time = this_count`, so it reflects the actual number of frames loaded (respecting both `countss` and `temporal_gap`) rather than all files in the directory.

### `scripts/preprocess_dynerf.py`

**Change:** Updated downsample factor from `1.0` to `2.0` to match the training configuration. This script extracts PNG frames from the MP4 videos for training.

---

## 2. New Evaluation Scripts

All new scripts are in the repository root alongside the existing training/rendering code.

### `evaluate_midpoints.py` — Low-Resolution Midpoint Evaluation

Evaluates the trained model at the held-out midpoint timestamps at training resolution (676 × 507). Computes PSNR, SSIM, and LPIPS (VGG backbone) under two settings:
- **Setting 1:** 17 training cameras at midpoint times (pure temporal interpolation).
- **Setting 2:** Held-out cam00 at midpoint times (joint spatial-temporal generalization).

```bash
python evaluate_midpoints.py \
  --configs arguments/dynerf/coffee_martini.py \
  --model_path output/coffee_martini_k8 \
  --source_path ~/Desktop/rohan/asmt2/coffee_martini \
  --gap 8 --iteration 14000
```

### `evaluate_highres.py` — High-Resolution Evaluation (2704 × 2028)

The primary evaluation script. Renders the trained 4DGS model at **native high resolution** (2704 × 2028) by overriding the camera's `image_width` and `image_height` on a shallow copy of the camera object. Since 4DGS uses FoV-based projection, no intrinsics scaling is needed — the same trained Gaussians project correctly at any resolution.

Computes:
- Full-image PSNR/SSIM/LPIPS against high-res ground truth.
- Dynamic-region (Otsu bounding box) metrics by cropping predictions and GT to the per-camera Otsu bbox before computing metrics.
- For k=8: additionally evaluates all 50 frames (not just midpoints) for the sawtooth PSNR analysis.

```bash
python evaluate_highres.py \
  --configs arguments/dynerf/coffee_martini.py \
  --model_path output/coffee_martini_k8 \
  --source_path ~/Desktop/rohan/asmt2/coffee_martini \
  --gap 8 --iteration 14000 --no-save-images
```

**Key implementation detail:** The function `make_highres_camera(camera, timestamp)` creates a shallow copy of the camera and overrides `image_width=2704`, `image_height=2028`, and `time=timestamp`. The Gaussian renderer reads these dimensions directly for rasterization.

### `evaluate_all50_k8.py` — All-50-Frames Evaluation (Low-Res)

Evaluates the k=8 model at all 50 frames (not just the 6 midpoints) at low resolution. Used to generate the sawtooth PSNR pattern data at training resolution. Superseded by the all-50 mode in `evaluate_highres.py` for the final report.

### `extract_highres_frames.py` — High-Res Ground Truth Extraction

Extracts the first 50 frames from each of the 18 camera MP4 videos at their native 2704 × 2028 resolution. These serve as ground truth for high-resolution evaluation.

Output structure: `coffee_martini_highres/{camXX}/images/{XXXX}.png`

### `plot_psnr_per_frame_k8.py` — Sawtooth PSNR Plot (Deliverable 3)

Generates the per-frame PSNR plot for k=8 showing the characteristic sawtooth/scallop pattern. Reads the all-50-frames CSVs and plots training-camera mean PSNR and cam00 PSNR across all 50 timestamps, marking training frames and test midpoints.

### `visual_comparison.py` — Visual Comparison Figure (Deliverable 4)

Generates a 3×4 composite figure comparing GT, rendered image, error heatmap, and dynamic-region crop across k ∈ {2, 4, 8} for the held-out cam00. Frames are chosen near the pouring action peak (~t=20).

---

## 3. Dynamic Region Detection

### Evolution of approaches

| Version | Script | Method | Status |
|---------|--------|--------|--------|
| v1 | `dynamic_bbox_evaluation.py` | Basic optical flow + fixed threshold | Superseded |
| v3 | `dynamic_bbox_evaluation_v3.py` | Added forward-backward consistency | Superseded |
| v4 | `dynamic_bbox_evaluation_v4.py` | Per-frame flow, τ=0.5px, morphological outlier removal | Superseded |
| **v5** | **`dynamic_bbox_evaluation_v5_otsu.py`** | **Otsu average-flow (final)** | **Active** |

### `dynamic_bbox_evaluation_v5_otsu.py` — Otsu Average-Flow Dynamic BBox (Final)

The final dynamic region detection method. For each camera:

1. Computes **Farneback optical flow** between all 49 consecutive frame pairs at native 2704 × 2028 resolution (5 pyramid levels, window size 21, 5 iterations, poly_n=7).
2. Applies **forward-backward consistency** checking and **photometric filtering** to discard unreliable flow vectors.
3. **Averages** per-pixel flow magnitudes across all frame pairs, producing a single motion magnitude map per camera.
4. Applies **Otsu's automatic thresholding** to the averaged map — this finds the optimal threshold that separates dynamic from static pixels by minimizing intra-class variance.
5. Extracts the tight axis-aligned **bounding box** around the binary dynamic mask.

**Advantages over v4:** One stable bbox per camera (not per-frame), automatic threshold selection (no manual τ), and evaluation at high resolution.

Output: `output/dynamic_bbox_evaluation_v5_otsu/otsu_bboxes.json` containing per-camera bboxes at both high-res and low-res coordinates. Diagnostic PNGs showing the average flow map, Otsu histogram, and bbox overlay are saved per camera.

```bash
python dynamic_bbox_evaluation_v5_otsu.py --stage diagnostic  # compute bboxes
python dynamic_bbox_evaluation_v5_otsu.py --stage evaluate    # apply to predictions
```

---

## 4. Output Structure

```
output/
├── coffee_martini_k2/                  # Trained model (k=2, 14k iter)
│   └── highres_evaluation/
│       └── gap_2/
│           ├── metrics.csv             # Full-image PSNR/SSIM/LPIPS per sample
│           └── bbox_metrics.csv        # Otsu bbox + full metrics per sample
├── coffee_martini_k4/                  # Trained model (k=4)
│   └── highres_evaluation/gap_4/...
├── coffee_martini_k8/                  # Trained model (k=8)
│   └── highres_evaluation/
│       └── gap_8/
│           ├── metrics.csv
│           ├── bbox_metrics.csv
│           └── all50/
│               ├── cam00_all50.csv     # Per-frame metrics, all 50 frames
│               └── train_cameras_all50.csv
├── dynamic_bbox_evaluation_v5_otsu/
│   ├── otsu_bboxes.json               # Per-camera Otsu bboxes (HR + LR coords)
│   └── diagnostics/                    # Per-camera flow/bbox visualizations
├── figures/
│   └── visual_comparison_k2_k4_k8_highres.png
├── 4DGS_Assignment_Report.md
└── CODEBASE_CHANGES.md                 # This file
```

---

## 5. Reproducibility

### Environment

- Conda env: `/home/val_gx10/Desktop/rohan/environments/4dgs_gb10_py310_v1/`
- Python 3.10, PyTorch with CUDA
- LPIPS (VGG backbone)

### Training commands (one per k)

```bash
PYTHON=/home/val_gx10/Desktop/rohan/environments/4dgs_gb10_py310_v1/bin/python

$PYTHON train.py \
  --configs arguments/dynerf/coffee_martini.py \
  --model_path output/coffee_martini_k8 \
  --source_path ~/Desktop/rohan/asmt2/coffee_martini \
  --gap 8
```

### Evaluation commands

```bash
# High-res evaluation (repeat for k=2, k=4, k=8)
$PYTHON evaluate_highres.py \
  --configs arguments/dynerf/coffee_martini.py \
  --model_path output/coffee_martini_k8 \
  --source_path ~/Desktop/rohan/asmt2/coffee_martini \
  --gap 8 --iteration 14000 --no-save-images

# Otsu bounding box detection (run once, covers all cameras)
$PYTHON dynamic_bbox_evaluation_v5_otsu.py --stage diagnostic

# Extract high-res GT frames (run once)
$PYTHON extract_highres_frames.py
```
