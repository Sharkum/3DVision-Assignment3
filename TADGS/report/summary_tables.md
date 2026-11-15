# TAD-GS Quantitative Evaluation Results (`coffee_martini`, 50 Frames, 4× Downsampling)

## Deliverable 1: Full-Image Quality on Held-Out Test Frames

| Frame Gap $k$ | Train Frames | Midpoint Test Frames | Setting 1: Train Cams, Held-Out Time (PSNR ↑ / SSIM ↑ / LPIPS ↓) | Setting 2: Held-Out Cam (`cam00`), Held-Out Time (PSNR ↑ / SSIM ↑ / LPIPS ↓) |
| :---: | :---: | :---: | :---: | :---: |
| $k = 2$ | 25 | 24 | **30.14** / **0.9282** / **0.0754** | **28.26** / **0.9114** / **0.0805** |
| $k = 4$ | 13 | 12 | **27.85** / **0.8855** / **0.1098** | **26.92** / **0.8757** / **0.1103** |
| $k = 8$ | 7 | 6 | **26.15** / **0.8581** / **0.1215** | **25.59** / **0.8510** / **0.1198** |

## Deliverable 2: Dynamic-Region Quality (Optical-Flow Motion Mask)

| Frame Gap $k$ | Setting 1: Train Cams, Held-Out Time (Masked PSNR ↑ / Masked SSIM ↑ / Masked LPIPS ↓) | Setting 2: Held-Out Cam (`cam00`), Held-Out Time (Masked PSNR ↑ / Masked SSIM ↑ / Masked LPIPS ↓) |
| :---: | :---: | :---: |
| $k = 2$ | **27.79** / **0.8859** / **0.0777** | **25.67** / **0.8578** / **0.0919** |
| $k = 4$ | **25.80** / **0.8240** / **0.1094** | **24.62** / **0.8025** / **0.1155** |
| $k = 8$ | **23.68** / **0.7751** / **0.1249** | **22.70** / **0.7420** / **0.1315** |
