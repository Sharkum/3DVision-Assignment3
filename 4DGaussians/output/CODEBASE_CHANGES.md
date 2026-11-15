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

## 2. Evaluation (used by the combined report)

All final 4DGS numbers come from `low_resolution_work_v2/`, evaluated at training resolution (676 × 507) with LPIPS-AlexNet v0.1, the same setup as TAD-GS and STGS.

### `low_resolution_work_v2/evaluate_lowres_v2.py`

Evaluates a trained model at the held-out midpoint timestamps:
- **Setting A:** 17 training cameras at midpoint times (pure temporal interpolation).
- **Setting B:** held-out cam00 at midpoint times (temporal + novel view).

Full-image PSNR/SSIM/LPIPS plus the same metrics inside the per-camera dynamic-region box. The boxes are read from the shared team masks in `low_resolution_work_v2/mask_dynamicregion/` (sequential `cam00`–`cam17` names, mapped to the 18 4DGS camera folders in sorted order). For k = 8 it also evaluates all 50 frames (`all50_cam00.csv`, `all50_train_cameras.csv`) for the per-frame plot.

```bash
for k in 2 4 8; do
  python low_resolution_work_v2/evaluate_lowres_v2.py \
    --configs arguments/dynerf/coffee_martini.py \
    --model_path output/coffee_martini_k${k} \
    --source_path <path to coffee_martini> \
    --gap $k --iteration 14000
done
```

### `low_resolution_work_v2/render_single_frame.py`

Renders one frame of cam00 at 676 × 507 for a given k and frame index, saved to `low_resolution_work_v2/renders/`. Used for the visual comparison figures (k = 2/4/8 at t = 1/2/4).

---

## 3. Output Structure

```
low_resolution_work_v2/
├── evaluate_lowres_v2.py
├── render_single_frame.py
├── mask_dynamicregion/              # shared team dynamic-region masks (one per camera)
├── renders/                         # cam00 renders used in the visual comparison
├── k2/  midpoint_metrics.csv, midpoint_bbox_metrics.csv
├── k4/  midpoint_metrics.csv, midpoint_bbox_metrics.csv
└── k8/  midpoint_metrics.csv, midpoint_bbox_metrics.csv, all50_cam00.csv, all50_train_cameras.csv
output/
├── coffee_martini_k{2,4,8}/         # trained models (git-ignored)
└── CODEBASE_CHANGES.md              # this file
```

---

## 4. Training

```bash
python train.py \
  --configs arguments/dynerf/coffee_martini.py \
  --model_path output/coffee_martini_k8 \
  --source_path <path to coffee_martini> \
  --gap 8
```

14,000 iterations, the authors' DyNeRF configuration, run on an NVIDIA GB10.
