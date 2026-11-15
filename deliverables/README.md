# Deliverables: combined results of all three methods

**Report:** [report/report.pdf](report/report.pdf) (source: [report/report.tex](report/report.tex); build with `tectonic report.tex` from `report/`).

Scene `coffee_martini`, frames 0-49, train on every k-th frame (k = 2, 4, 8), test on the midpoints. Setting A = 17 training cameras at test times; Setting B = held-out `cam00` at test times.

| Assignment item | Where |
| :--- | :--- |
| 1. PSNR/SSIM/LPIPS per method, k, setting | Report Table 1; [tables/metrics_all_methods.md](tables/metrics_all_methods.md), [.csv](tables/metrics_all_methods.csv) |
| 2. Dynamic-region metrics + mask definition | Report Sec. 2, Fig. 1, Table 2; same tables (`dyn_*` columns) |
| 3. Per-frame PSNR, k = 8 | Report Sec. 3.2 (Figs. 2-4, Table 3); [tables/perframe_k8_all_methods.csv](tables/perframe_k8_all_methods.csv), [tables/tadgs_perframe_cam00_allk.csv](tables/tadgs_perframe_cam00_allk.csv) |
| 4. Visual results (test frame, GT, error map, fast-moving crop) | Report Sec. 3.3 (Figs. 5-8) |
| Questions 1-4 | Report Sec. 4 |
| Experiment details (hyperparameters, hardware, deviations) | Report Sec. 1 |

## Layout

```text
deliverables/
├── report/       report.tex, report.pdf
├── tables/       compiled metrics (CSV / Markdown / LaTeX rows), summary_numbers.json
│   └── raw/      copies of each method's raw metric files
├── figures/      figures used in the report
└── scripts/      compile_results.py (tables, per-frame plots), make_visuals.py (visual comparison figures)
```

Regenerate: `<repo>/.env/tadgs/bin/python deliverables/scripts/compile_results.py` and `.../make_visuals.py`, then rebuild the PDF. `make_visuals.py` and the TAD-GS all-k plot also need the git-ignored ground-truth frames and TAD-GS all-50-frame renders.

## Comparability

- All three methods: 676x507, LPIPS-Alex, same frames and splits, same per-camera Otsu box for dynamic-region metrics.
- Visual figures (`visual_uniform_*.jpg`) use the same frame (t = 1/2/4), crop and error scale for all methods.
