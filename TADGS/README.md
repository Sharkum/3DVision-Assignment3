# TAD-GS: Temporally Aware Densification for Dynamic 3DGS

Our TAD-GS experiments on `coffee_martini` (50 frames, 676×507). Training uses frames {0, k, 2k, ..., 48}; evaluation uses the unseen midpoint frames, on the training cameras (Setting 1) and on the held-out camera `cam00` (Setting 2).

**Read first:** [report/TADGS_Assignment_Report.md](report/TADGS_Assignment_Report.md) (setup, tables, plots, answers to the questions).
Background on how TAD-GS works and what it inherits from STG/3DGS: [docs/TADGS_explained.md](docs/TADGS_explained.md).

## Layout

```text
TADGS/
├── report/                  final report + standalone metric tables
├── results/                 metrics JSON, figures, render frames, videos
├── docs/TADGS_explained.md  walkthrough of the method and code
├── scripts/                 our pipeline (see below)
├── TAD-GS/                  method code (upstream vikramsandu/TAD-GS, modified, see below)
├── data/                    (git-ignored) raw videos, preprocessed COLMAP scenes, dynamic-region masks
└── runs/                    (git-ignored) trained checkpoints, tadgs_k{1,2,4,8}/
```

## Pipeline (run from `TADGS/`)

| Step | Script | Output |
| :--- | :--- | :--- |
| 1. Environment | `scripts/setup_env.sh`, `scripts/build_cuda_ext.sh` | `../.env/tadgs`, compiled CUDA rasteriser and simple-knn |
| 2. Preprocess video | `scripts/preprocess_coffee_martini.py` | `data/neural_3d_video/scenes/coffee_martini/colmap_{0..49}` (17 training cameras for the point cloud, `cam00` held out) |
| 3. Dynamic-region masks | `scripts/otsu_avg_flow_bbox.py` (used for reported masked metrics), `scripts/dynamic_bbox_optical_flow.py`, `scripts/generate_dynamic_masks.py` | `data/neural_3d_video/flow_masks/coffee_martini/{otsu_bbox_masks,bbox_masks,valid_masks}` |
| 4. Train, evaluate, plot | `scripts/run_all_experiments.sh` (calls `evaluate_all.py` then `generate_plots_and_visuals.py`) | `runs/tadgs_k{2,4,8}`, `results/tadgs_evaluation_metrics.json`, figures, `report/summary_tables.md` |
| 5. Extras | `scripts/render_all_frames.py`, `scripts/make_video.sh`, `scripts/render_bbox_video.py`, `scripts/render_optical_flow_video.py` | videos and full 50-frame renders in `results/` |

The scripts resolve paths relative to their own location, so they can be run from anywhere. The Python environment is expected at `<repo root>/.env/tadgs`. `setup_env.sh` clones an existing CUDA-enabled env; set `SRC_ENV=/path/to/env` to choose which.

## Key results files

| Deliverable | File |
| :--- | :--- |
| Full-image and dynamic-region metrics, k = 2, 4, 8 | [report/summary_tables.md](report/summary_tables.md), [results/tadgs_evaluation_metrics.json](results/tadgs_evaluation_metrics.json) |
| Per-frame PSNR over all 50 frames (k = 8) | [results/psnr_per_frame_k8.png](results/psnr_per_frame_k8.png) |
| Visual comparison and error maps | [results/visual_comparison_k2_k4_k8.png](results/visual_comparison_k2_k4_k8.png) |
| Dynamic-region mask visual | [results/dynamic_mask_visualization.png](results/dynamic_mask_visualization.png) |
| Optical-flow / bounding-box figures and videos | `results/otsu_avg_flow_bbox_*`, `results/optical_flow_cam00.mp4`, `results/dynamic_bboxes_all_cams_frame20.png`, `results/dynamic_bbox_cc_demo.png` |

## Changes to the upstream TAD-GS code (`TAD-GS/`)

Upstream: [vikramsandu/TAD-GS](https://github.com/vikramsandu/TAD-GS) at commit `24c9f2a`. Modified for this assignment:
- `train.py`, `render.py`, `arguments/default.py`, `arguments/neural_3d_video/coffee_martini.json`: `--frame_gap k` train/test split by frame index.
- `scene/{__init__,dataset,dataset_readers,cameras}.py`: multi-anchor point-cloud loading (t = 0, 24, 48), temporal split filtering, images kept in CPU RAM to fit a 4 GB GPU.
- `scene/tad_gaussian.py`, `utils/common/{metrics,helper_model,helper3dg}.py`: model and metric fixes; masked PSNR/SSIM/LPIPS (LPIPS uses AlexNet, v0.1).
- `submodules/{gaussian_rasterization_ch3,simple-knn}/setup.py` (and `rasterizer_impl.h`): build fixes.
- `utils/common/plyfile.py`: added PLY I/O helper.

`TAD-GS/submodules/mmcv` is an unmodified upstream submodule and is not committed here. Fetch it with `git clone https://github.com/open-mmlab/mmcv.git TAD-GS/submodules/mmcv` if needed (only used for KNN during point-cloud init in the upstream pipeline).
