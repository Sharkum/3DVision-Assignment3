# Assignment 3: Temporal Interpolation with Dynamic 3DGS

**Course:** E9 247 — Learning for 3D Vision and Inverse Graphics  
**Scene:** `coffee_martini` (Neural 3D Video Synthesis Dataset)  
**Method 1:** 4D Gaussian Splatting (4DGS) — Wu et al.  
**Method 2:** Temporally Aware Densification GS (TAD-GS)

---

## Deliverable 1: Quantitative Metrics

### Evaluation Protocol

- **Training cameras:** cam01–cam20 (excluding cam03, cam15, cam17) → 17 cameras
- **Held-out camera:** cam00
- **Frame gaps:** k ∈ {2, 4, 8} — train on {0, k, 2k, ...} up to frame 48, test on midpoints {k/2, 3k/2, ...}
- **Setting 1:** Training cameras at held-out midpoint timestamps
- **Setting 2:** Held-out camera (cam00) at held-out midpoint timestamps
- **Metrics:** PSNR (dB) ↑, SSIM ↑, LPIPS ↓
- **LPIPS backbone:** 4DGS uses **VGG**, TAD-GS uses **AlexNet** (not directly comparable)
- **Image resolution:** 676 × 507 (4× downsampled from 2704 × 2028)
- **Iterations:** 14,000 (for all k)

---

### Table 1: Full-Image Metrics at Midpoint Test Frames

#### Setting 1 — Training Cameras, Held-Out Time

| k | Metric | 4DGS | TAD-GS |
|---|--------|------|--------|
| 2 | PSNR (dB) | **32.43** | |
| 2 | SSIM | **0.9381** | |
| 2 | LPIPS (VGG / Alex) | **0.1552** | |
| 4 | PSNR (dB) | **31.13** | |
| 4 | SSIM | **0.9304** | |
| 4 | LPIPS (VGG / Alex) | **0.1558** | |
| 8 | PSNR (dB) | **28.22** | |
| 8 | SSIM | **0.9122** | |
| 8 | LPIPS (VGG / Alex) | **0.1696** | |

#### Setting 2 — Held-Out Camera (cam00), Held-Out Time

| k | Metric | 4DGS | TAD-GS |
|---|--------|------|--------|
| 2 | PSNR (dB) | **28.40** | |
| 2 | SSIM | **0.9200** | |
| 2 | LPIPS (VGG / Alex) | **0.1604** | |
| 4 | PSNR (dB) | **28.19** | |
| 4 | SSIM | **0.9105** | |
| 4 | LPIPS (VGG / Alex) | **0.1607** | |
| 8 | PSNR (dB) | **26.39** | |
| 8 | SSIM | **0.8919** | |
| 8 | LPIPS (VGG / Alex) | **0.1771** | |

#### Number of test samples per setting

| k | Setting 1 (train cams × midpoints) | Setting 2 (cam00 × midpoints) |
|---|-------------------------------------|-------------------------------|
| 2 | 17 × 24 = 408 | 1 × 24 = 24 |
| 4 | 17 × 12 = 204 | 1 × 12 = 12 |
| 8 | 17 × 6 = 102 | 1 × 6 = 6 |

---

### Table 2: Dynamic Region (BBox-Masked) Metrics at Midpoint Test Frames

Dynamic regions detected via forward-backward Farneback optical flow with photometric filtering and outlier removal (dilation=15px, min relative area=5%).

#### Setting 1 — Training Cameras, Held-Out Time

| k | Metric Scope | 4DGS PSNR | 4DGS SSIM | 4DGS LPIPS | TAD-GS PSNR | TAD-GS SSIM | TAD-GS LPIPS |
|---|-------------|-----------|-----------|------------|-------------|-------------|--------------|
| 2 | Full image | 32.37 | 0.9373 | 0.1509 | | | |
| 2 | BBox only | 31.52 | 0.9287 | 0.1377 | | | |
| 4 | Full image | 31.09 | 0.9296 | 0.1515 | | | |
| 4 | BBox only | 29.81 | 0.9141 | 0.1416 | | | |
| 8 | Full image | 28.20 | 0.9114 | 0.1654 | | | |
| 8 | BBox only | 26.45 | 0.8787 | 0.1668 | | | |

#### Setting 2 — Held-Out Camera (cam00), Held-Out Time

| k | Metric Scope | 4DGS PSNR | 4DGS SSIM | 4DGS LPIPS | TAD-GS PSNR | TAD-GS SSIM | TAD-GS LPIPS |
|---|-------------|-----------|-----------|------------|-------------|-------------|--------------|
| 2 | Full image | 28.52 | 0.9195 | 0.1561 | | | |
| 2 | BBox only | 27.89 | 0.9081 | 0.1412 | | | |
| 4 | Full image | 28.27 | 0.9100 | 0.1565 | | | |
| 4 | BBox only | 27.33 | 0.8974 | 0.1431 | | | |
| 8 | Full image | 26.47 | 0.8913 | 0.1734 | | | |
| 8 | BBox only | 25.03 | 0.8513 | 0.1777 | | | |

---

## Deliverable 2: Analysis Questions

### Q1. How does each method represent temporal information? What are the tradeoffs?

> *(descriptive — answer separately)*

### Q2. Why does larger k degrade temporal interpolation quality? Is the degradation uniform across the frame or concentrated in dynamic regions?

> *(descriptive — answer separately)*

**Supporting data — degradation by k (4DGS, Setting 1):**

| k | Full PSNR | BBox PSNR | Drop vs full |
|---|-----------|-----------|-------------|
| 2 | 32.37 | 31.52 | −0.85 dB |
| 4 | 31.09 | 29.81 | −1.28 dB |
| 8 | 28.20 | 26.45 | −1.75 dB |

The gap between full-image and bbox PSNR widens as k increases (0.85 → 1.28 → 1.75 dB), confirming degradation concentrates in dynamic regions.

### Q3. Compare and contrast the two methods for temporal interpolation. Which handles large temporal gaps better and why?

> *(descriptive — answer separately)*

**Supporting data — Setting 1 (train cams) PSNR comparison:**

| k | 4DGS | TAD-GS | Delta |
|---|------|--------|-------|
| 2 | 32.43 | | |
| 4 | 31.13 | | |
| 8 | 28.22 | | |

**Setting 2 (cam00) PSNR comparison:**

| k | 4DGS | TAD-GS | Delta |
|---|------|--------|-------|
| 2 | 28.40 | | |
| 4 | 28.19 | | |
| 8 | 26.39 | | |

### Q4. What are the failure modes you observe for each method?

> *(descriptive — answer separately)*

---

## Deliverable 3: Per-Frame PSNR Plot (k = 8)

Evaluated the k=8 model at **all 50 frames** (0–49) on both training cameras (17-cam mean) and held-out cam00.

### 4DGS Per-Frame PSNR Plot

![4DGS Per-Frame PSNR (k=8)](coffee_martini_k8/all50_evaluation/psnr_per_frame_k8.png)

### Summary Statistics (k = 8, all 50 frames)

| Camera set | Train-frame PSNR | Midpoint PSNR | All-50 mean PSNR | Sawtooth amplitude |
|------------|-----------------|---------------|-------------------|--------------------|
| Training cameras (17-cam mean) | 32.41 | 28.22 | 29.94 | ~4.19 dB |
| Held-out cam00 | 28.46 | 26.39 | 27.30 | ~2.07 dB |
| TAD-GS training cameras | | | | |
| TAD-GS held-out cam00 | | | | |

The sawtooth pattern is clearly visible: PSNR peaks at training timestamps {0, 8, 16, 24, 32, 40, 48} and dips at midpoints {4, 12, 20, 28, 36, 44}. The amplitude is larger for training cameras (~4.2 dB) than cam00 (~2.1 dB), because training cameras were seen during optimization at those exact frames, while cam00 was never seen at any frame.

---

## Deliverable 4: Visual Comparison

### 4DGS Visual Comparison (cam00, k = 2, 4, 8)

![4DGS Visual Comparison](figures/visual_comparison_k2_k4_k8.png)

**Frame selection:** k=2 → t=21, k=4 → t=22, k=8 → t=20 (all near the fast pouring action on cam00).

| k | Frame | PSNR (cam00) |
|---|-------|-------------|
| 2 | t=21 | 28.58 dB |
| 4 | t=22 | 28.00 dB |
| 8 | t=20 | 27.57 dB |

Columns: Ground Truth | 4DGS Render | Absolute Error (4× scaled, inferno colormap) | Zoomed Dynamic Region (GT | Render | Error).

The green rectangle highlights the dynamic region (bartender's hands/shaker area). Error increases visibly with k, particularly around the hands and pouring motion where temporal interpolation artifacts are strongest.

### Dynamic Region Detection

![Dynamic BBox Diagnostic](dynamic_bbox_evaluation_v4_outlier/diagnostics/cam00_frames20_to22.png)

Dynamic bounding boxes computed via forward-backward optical flow (Farneback) with photometric filtering and outlier removal. Flow threshold τ = 0.5 px.

---

## Appendix: File Index

| File | Description |
|------|-------------|
| `output/coffee_martini_k{2,4,8}/` | Trained 4DGS models (14k iterations) |
| `output/coffee_martini_k{2,4,8}/midpoint_evaluation_v4/gap_{k}/metrics.csv` | Per-sample midpoint metrics |
| `output/coffee_martini_k8/all50_evaluation/cam00_all50.csv` | cam00 PSNR/SSIM/LPIPS for all 50 frames |
| `output/coffee_martini_k8/all50_evaluation/train_cameras_all50.csv` | 17-cam mean metrics for all 50 frames |
| `output/coffee_martini_k8/all50_evaluation/psnr_per_frame_k8.png` | Deliverable 3 plot |
| `output/figures/visual_comparison_k2_k4_k8.png` | Deliverable 4 figure |
| `output/dynamic_bbox_evaluation_v4_outlier/` | Dynamic region detection + bbox metrics |
| `evaluate_all50_k8.py` | Script: render k=8 at all 50 timestamps |
| `plot_psnr_per_frame_k8.py` | Script: generate sawtooth PSNR plot |
| `visual_comparison.py` | Script: generate composed visual comparison |
| `evaluate_midpoints.py` | Script: midpoint evaluation (Settings 1 & 2) |
| `dynamic_bbox_evaluation_v4.py` | Script: dynamic region detection + bbox metrics |
