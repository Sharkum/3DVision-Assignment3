# Assignment 3: Temporal Interpolation with Dynamic 3D Gaussian Splatting

Task: render **unseen timestamps** of a dynamic scene (`coffee_martini`, Neural 3D Video dataset, 50 frames, held-out camera `cam00`) from models trained on every k-th frame, k in {2, 4, 8}. Assignment text: [assignment/Assignment_Dynamic3DGS_Interpolation.pdf](assignment/Assignment_Dynamic3DGS_Interpolation.pdf).

Three dynamic Gaussian Splatting methods were compared, one folder per method:

| Method | Folder | Report | Start here |
| :--- | :--- | :--- | :--- |
| **TAD-GS** (temporally aware densification) | [TADGS/](TADGS/) | [TADGS/report/TADGS_Assignment_Report.md](TADGS/report/TADGS_Assignment_Report.md) | [TADGS/README.md](TADGS/README.md) |
| **STGS** (Spacetime Gaussians) | [STGS/](STGS/) | [STGS/report/STGS_report.md](STGS/report/STGS_report.md) | |
| **4DGaussians** (HexPlane + MLP deformation) | [4DGaussians/](4DGaussians/) | [4DGaussians/output/4DGS_Assignment_Report.md](4DGaussians/output/4DGS_Assignment_Report.md) | |

Each method folder is self-contained (code, scripts, results, report). Large local-only artefacts (datasets, checkpoints, the Python environment) are git-ignored; see each method's README for how to regenerate them.

```text
.
├── README.md                 this file
├── assignment/               assignment PDF
├── TADGS/                    TAD-GS: code, pipeline scripts, results, report, docs
├── STGS/                     Spacetime Gaussians (teammate)
├── 4DGaussians/              4D Gaussians (teammate)
└── .env/                     local Python env (git-ignored, shared by TADGS scripts)
```
