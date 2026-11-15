# Assignment 3: Temporal Interpolation with Dynamic 3D Gaussian Splatting

Task: render **unseen timestamps** of a dynamic scene (`coffee_martini`, Neural 3D Video dataset, 50 frames, held-out camera `cam00`) from models trained on every k-th frame, k in {2, 4, 8}. Assignment text: [assignment/Assignment_Dynamic3DGS_Interpolation.pdf](assignment/Assignment_Dynamic3DGS_Interpolation.pdf).

**Report: [deliverables/report/report.pdf](deliverables/report/report.pdf).** All tables, figures and the scripts that build them are in [deliverables/](deliverables/).

Three dynamic Gaussian Splatting methods were compared, one folder per method:

| Method | Folder | Final metrics used by the report |
| :--- | :--- | :--- |
| **TAD-GS** (temporally aware densification) | [TADGS/](TADGS/) (see [TADGS/README.md](TADGS/README.md)) | `TADGS/results/tadgs_evaluation_metrics.json` |
| **STGS** (Spacetime Gaussians) | [STGS/](STGS/) | `STGS/results/metrics_per_image.csv`, `STGS/results/perframe_psnr_k8.csv` |
| **4DGS** (HexPlane + MLP deformation) | [4DGaussians/](4DGaussians/) (see [4DGaussians/output/CODEBASE_CHANGES.md](4DGaussians/output/CODEBASE_CHANGES.md)) | `4DGaussians/low_resolution_work_v2/k{2,4,8}/*.csv` |

All three methods are evaluated at 676x507 with LPIPS-Alex, on the same frames and splits, using the same per-camera dynamic-region box. Large local-only artefacts (datasets, checkpoints, full render sets, the Python environment) are git-ignored.

```text
.
├── README.md                 this file
├── assignment/               assignment PDF
├── deliverables/             report (LaTeX/PDF), compiled tables, figures, build scripts
├── TADGS/                    TAD-GS: method code, pipeline scripts, metrics
├── STGS/                     Spacetime Gaussians: method code, analysis scripts, metrics, renders
├── 4DGaussians/              4DGS: method code, 676x507 evaluation (low_resolution_work_v2/)
└── .env/                     local Python env (git-ignored)
```
