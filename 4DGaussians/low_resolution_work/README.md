# Low-Resolution Evaluation (676×507) — AlexNet LPIPS

All metrics computed at **training resolution (676×507)** with **LPIPS backbone: AlexNet v0.1** (`lpips.LPIPS(net='alex', version='0.1')`) for cross-method comparability with TAD-GS.

**Dynamic region:** Otsu average-flow bounding box per camera, using `bbox_lowres` coordinates from `output/dynamic_bbox_evaluation_v5_otsu/otsu_bboxes.json`.

**Scene:** `coffee_martini` | **Training:** 14,000 iterations | **Model:** 4DGS

---

## Directory Structure

```
low_resolution_work/
├── README.md                          ← this file
├── evaluate_lowres.py                 ← evaluation script
├── k2/
│   ├── midpoint_metrics.csv           ← full-image metrics (Setting 1 + 2)
│   └── midpoint_bbox_metrics.csv      ← Otsu bbox crop metrics (Setting 1 + 2)
├── k4/
│   ├── midpoint_metrics.csv
│   └── midpoint_bbox_metrics.csv
└── k8/
    ├── midpoint_metrics.csv
    ├── midpoint_bbox_metrics.csv
    ├── all50_cam00.csv                ← all 50 frames, cam00, full + bbox
    └── all50_train_cameras.csv        ← all 50 frames, 17-cam mean, full + bbox
```

---

## What Was Computed

### Midpoint Evaluation (k=2, k=4, k=8)

For each k, the model is trained on every k-th frame (0, k, 2k, ..., 48). Test midpoints are the frames halfway between training frames (k/2, 3k/2, ...).

- **Setting 1 (Training Cameras):** 17 training cameras × midpoint frames
- **Setting 2 (Held-Out Camera):** cam00 × midpoint frames

| k | Midpoints per camera | Setting 1 samples | Setting 2 samples |
|---|---------------------|-------------------|-------------------|
| 2 | 24 (frames 1,3,5,...,47) | 408 | 24 |
| 4 | 12 (frames 2,6,10,...,46) | 204 | 12 |
| 8 | 6 (frames 4,12,20,28,36,44) | 102 | 6 |

### Per-Frame All-50 (k=8 only)

All 50 frames (0–49) evaluated for both cam00 and 17 training cameras. Used for sawtooth temporal overfitting analysis. Each row marks whether the frame is a training timestamp or a test midpoint.

---

## How Metrics Are Calculated

**Rendering:** 4DGS renders at camera's native resolution (676×507). Timestamp = `frame_idx / 49.0`.

**Ground truth:** Low-res images from `coffee_martini/{camXX}/images/{XXXX}.png` (676×507 PNG).

**Full-image metrics:**
- **PSNR:** `−10 · log10(MSE)` over all pixels
- **SSIM:** Gaussian-windowed structural similarity (window_size=11), `utils/loss_utils.py`
- **LPIPS:** AlexNet v0.1 backbone, input scaled to `[−1, 1]` via `img * 2 − 1`

**Dynamic region (bbox) metrics:**
- Crop both prediction and GT to `[x0:x1, y0:y1]` using `bbox_lowres` from `otsu_bboxes.json`
- **PSNR:** MSE computed only over cropped pixels
- **SSIM:** SSIM map computed on the crop, then averaged
- **LPIPS:** AlexNet evaluated on the cropped patch

---

## CSV Column Reference

### midpoint_metrics.csv
| Column | Description |
|--------|-------------|
| setting | `training_cameras` (Setting 1) or `heldout_camera` (Setting 2) |
| camera | Camera name (e.g. `cam01`, `cam00`) |
| frame | Frame index (midpoint) |
| PSNR | Full-image PSNR (dB) |
| SSIM | Full-image SSIM |
| LPIPS_alex | Full-image LPIPS (AlexNet v0.1) |

### midpoint_bbox_metrics.csv
| Column | Description |
|--------|-------------|
| setting | `training_cameras` or `heldout_camera` |
| camera | Camera name |
| frame | Frame index |
| PSNR_bbox | Otsu bbox crop PSNR |
| SSIM_bbox | Otsu bbox crop SSIM |
| LPIPS_alex_bbox | Otsu bbox crop LPIPS (AlexNet) |
| PSNR_full | Full-image PSNR (same frame) |
| SSIM_full | Full-image SSIM |
| LPIPS_alex_full | Full-image LPIPS |

### all50_cam00.csv / all50_train_cameras.csv
| Column | Description |
|--------|-------------|
| frame | Frame index (0–49) |
| full_psnr | Full-image PSNR |
| full_ssim | Full-image SSIM |
| full_lpips_alex | Full-image LPIPS (AlexNet) |
| bbox_psnr | Otsu bbox crop PSNR |
| bbox_ssim | Otsu bbox crop SSIM |
| bbox_lpips_alex | Otsu bbox crop LPIPS (AlexNet) |
| is_train | True if frame is a training timestamp |
| is_midpoint | True if frame is a test midpoint |
| num_cameras | (train only) number of cameras averaged |

---

## Summary Results

### Full-Image — Midpoints

| k | Setting 1 PSNR | Setting 1 SSIM | Setting 1 LPIPS | Setting 2 PSNR | Setting 2 SSIM | Setting 2 LPIPS |
|---|---------------|---------------|----------------|---------------|---------------|----------------|
| 2 | 32.43 | 0.9381 | 0.0740 | 28.40 | 0.9200 | 0.0732 |
| 4 | 31.13 | 0.9304 | 0.0742 | 28.19 | 0.9105 | 0.0757 |
| 8 | 28.22 | 0.9122 | 0.0848 | 26.39 | 0.8919 | 0.0853 |

### Dynamic Region (Otsu BBox) — Midpoints

| k | Setting 1 PSNR | Setting 1 SSIM | Setting 1 LPIPS | Setting 2 PSNR | Setting 2 SSIM | Setting 2 LPIPS |
|---|---------------|---------------|----------------|---------------|---------------|----------------|
| 2 | 30.72 | 0.9252 | 0.0829 | 26.77 | 0.9045 | 0.0796 |
| 4 | 27.83 | 0.8874 | 0.0878 | 25.49 | 0.8678 | 0.0894 |
| 8 | 23.18 | 0.7994 | 0.1152 | 22.18 | 0.7843 | 0.1126 |

### k=8 All-50 Sawtooth

| Camera Set | Full Sawtooth Δ | BBox Sawtooth Δ |
|------------|----------------|-----------------|
| Training cameras (17-cam mean) | 4.19 dB | 7.83 dB |
| Held-out cam00 | 2.07 dB | 5.32 dB |
