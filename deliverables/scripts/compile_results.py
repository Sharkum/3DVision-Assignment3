"""Compile the results of all three methods (TAD-GS, STGS, 4DGS) into deliverables/.

Run from anywhere:  <repo>/.env/tadgs/bin/python deliverables/scripts/compile_results.py

Sources
- TAD-GS : TADGS/results/tadgs_evaluation_metrics.json (+ TADGS/results/renders_k*_cam00_all50 if present)
- 4DGS   : 4DGaussians/low_resolution_work_v2/ (676x507, LPIPS-Alex, team boxes)
- STGS   : STGS/results/metrics_per_image.csv and STGS/results/perframe_psnr_k8.csv
Outputs go to deliverables/tables and deliverables/figures.
"""
import csv
import json
import shutil
import statistics as st
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deliverables"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

KS = [2, 4, 8]
SETTINGS = {"A": "Train cams, held-out time", "B": "Held-out cam00, held-out time"}
METRICS = ["PSNR", "SSIM", "LPIPS", "dPSNR", "dSSIM", "dLPIPS"]

# ---------------------------------------------------------------- TAD-GS
tad = json.loads((ROOT / "TADGS/results/tadgs_evaluation_metrics.json").read_text())
R = {}  # R[(method, k, setting)] = dict of metrics
for k in KS:
    for s, key in (("A", "setting1_train_cams_heldout_time"), ("B", "setting2_heldout_cam_heldout_time")):
        d = tad[f"k={k}"][key]
        R[("TAD-GS", k, s)] = dict(PSNR=d["PSNR"], SSIM=d["SSIM"], LPIPS=d["LPIPS"], dPSNR=d["Masked-PSNR"],
                                   dSSIM=d["Masked-SSIM"], dLPIPS=d["Masked-LPIPS"], n=d["num_views"])

# ---------------------------------------------------------------- STGS (per-image CSV from the STGS run)
STGS_ROWS = list(csv.DictReader(open(ROOT / "STGS/results/metrics_per_image.csv")))
for k in KS:
    for s, split in (("A", "train"), ("B", "heldout")):
        rr = [r for r in STGS_ROWS if int(r["k"]) == k and r["cam_split"] == split and r["time_split"] == "test"]
        m = lambda c: st.mean(float(r[c]) for r in rr)  # noqa: E731
        R[("STGS", k, s)] = dict(PSNR=m("psnr"), SSIM=m("ssim"), LPIPS=m("lpips"), dPSNR=m("dyn_psnr"),
                                 dSSIM=m("dyn_ssim"), dLPIPS=m("dyn_lpips"), n=len(rr))

# ---------------------------------------------------------------- 4DGS (676x507, LPIPS-Alex, team box)
LR = ROOT / "4DGaussians/low_resolution_work_v2"
for k in KS:
    rows = list(csv.DictReader(open(LR / f"k{k}/midpoint_bbox_metrics.csv")))
    for s, name in (("A", "training_cameras"), ("B", "heldout_camera")):
        rr = [r for r in rows if r["setting"] == name]
        m = lambda c: st.mean(float(r[c]) for r in rr)  # noqa: E731
        R[("4DGS", k, s)] = dict(PSNR=m("PSNR_full"), SSIM=m("SSIM_full"), LPIPS=m("LPIPS_alex_full"),
                                 dPSNR=m("PSNR_bbox"), dSSIM=m("SSIM_bbox"), dLPIPS=m("LPIPS_alex_bbox"), n=len(rr))

METHODS = ["TAD-GS", "STGS", "4DGS"]

# ---------------------------------------------------------------- tables
with open(TAB / "metrics_all_methods.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "k", "setting", "n_images", "PSNR", "SSIM", "LPIPS",
                "dyn_PSNR", "dyn_SSIM", "dyn_LPIPS"])
    for mth in METHODS:
        for k in KS:
            for s in "AB":
                r = R[(mth, k, s)]
                w.writerow([mth, k, SETTINGS[s], r["n"]] + [f"{r[c]:.4f}" for c in METRICS])

md = ["# Test-frame metrics, all methods (coffee_martini, held-out cam00)", "",
      "Setting A = 17 training cameras at held-out times; B = held-out cam00 at held-out times. "
      "dyn_* = restricted to the team dynamic-region box. All methods at 676x507 with LPIPS-Alex.", "",
      "| Method | k | Setting | PSNR | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS |",
      "|---|---|---|---|---|---|---|---|---|"]
for mth in METHODS:
    for k in KS:
        for s in "AB":
            r = R[(mth, k, s)]
            md.append(f"| {mth} | {k} | {s} | {r['PSNR']:.2f} | {r['SSIM']:.4f} | {r['LPIPS']:.4f} | "
                      f"{r['dPSNR']:.2f} | {r['dSSIM']:.4f} | {r['dLPIPS']:.4f} |")
(TAB / "metrics_all_methods.md").write_text("\n".join(md) + "\n")


def fmt(c, v):
    return f"{v:.2f}" if "PSNR" in c else f"{v:.3f}"


def best_marks(k, s, c, lower=False):
    """Methods with the best value at the displayed precision (ties are all bolded)."""
    vals = {m: float(fmt(c, R[(m, k, s)][c])) for m in METHODS}
    best = min(vals.values()) if lower else max(vals.values())
    return {m for m, v in vals.items() if v == best}


def tex_table(cols, fname):
    lines = []
    for s in "AB":
        for k in KS:
            cells = []
            for mth in METHODS:
                for c in cols:
                    v = R[(mth, k, s)][c]
                    txt = fmt(c, v)
                    if mth in best_marks(k, s, c, lower="LPIPS" in c):
                        txt = r"\textbf{" + txt + "}"
                    cells.append(txt)
            lab = ("A" if s == "A" else "B") if k == 2 else ""
            lines.append(f"{lab} & {k} & " + " & ".join(cells) + r" \\")
        if s == "A":
            lines.append(r"\midrule")
    (TAB / fname).write_text("\n".join(lines) + "\n")


tex_table(["PSNR", "SSIM", "LPIPS"], "table_full.tex")
tex_table(["dPSNR", "dSSIM", "dLPIPS"], "table_dyn.tex")

COL = {"TAD-GS": "#1f77b4", "STGS": "#d62728", "4DGS": "#2ca02c"}

# ---------------------------------------------------------------- per-frame k=8, all three methods
t = np.arange(50)
train = list(range(0, 49, 8))
test = list(range(4, 48, 8))
PF = {}  # PF[method][(setting, region)] = array over 50 frames; region in {"full", "dyn"}
a50 = tad["k=8"]["all_50_frames"]
PF["TAD-GS"] = {("A", "full"): np.array([a50[str(i)]["train_cams_PSNR"] for i in t]),
                ("A", "dyn"): np.array([a50[str(i)]["train_cams_Masked_PSNR"] for i in t]),
                ("B", "full"): np.array([a50[str(i)]["cam00_PSNR"] for i in t]),
                ("B", "dyn"): np.array([a50[str(i)]["cam00_Masked_PSNR"] for i in t])}
sp = {int(r["frame"]): r for r in csv.DictReader(open(ROOT / "STGS/results/perframe_psnr_k8.csv"))}
PF["STGS"] = {("A", "full"): np.array([float(sp[i]["traincams_psnr"]) for i in t]),
              ("A", "dyn"): np.array([float(sp[i]["traincams_dyn_psnr"]) for i in t]),
              ("B", "full"): np.array([float(sp[i]["cam00_psnr"]) for i in t]),
              ("B", "dyn"): np.array([float(sp[i]["cam00_dyn_psnr"]) for i in t])}
ga = {int(r["frame"]): r for r in csv.DictReader(open(LR / "k8/all50_train_cameras.csv"))}
gb = {int(r["frame"]): r for r in csv.DictReader(open(LR / "k8/all50_cam00.csv"))}
PF["4DGS"] = {("A", "full"): np.array([float(ga[i]["full_psnr"]) for i in t]),
              ("A", "dyn"): np.array([float(ga[i]["bbox_psnr"]) for i in t]),
              ("B", "full"): np.array([float(gb[i]["full_psnr"]) for i in t]),
              ("B", "dyn"): np.array([float(gb[i]["bbox_psnr"]) for i in t])}
KEYS = [("A", "full"), ("A", "dyn"), ("B", "full"), ("B", "dyn")]

with open(TAB / "perframe_k8_all_methods.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "role"] + [f"{m}_{s}_{r}" for m in METHODS for s, r in KEYS])
    for i in t:
        role = "train" if i in train else ("test" if i in test else "other")
        w.writerow([i, role] + [f"{PF[m][key][i]:.3f}" for m in METHODS for key in KEYS])

# train-frame vs test-frame means, k=8 (Table 3 in the report)
trtest = {m: {key: (PF[m][key][train].mean(), PF[m][key][test].mean()) for key in KEYS} for m in METHODS}
lines = []
for m in METHODS:
    for reg, lab in (("full", "full image"), ("dyn", "dynamic region")):
        cells = []
        for s in "AB":
            a, b = trtest[m][(s, reg)]
            cells += [f"{a:.2f}", f"{b:.2f}", f"{a - b:.2f}"]
        name = m
        lines.append(f"{name} ({lab}) & " + " & ".join(cells) + r" \\")
(TAB / "table_trtest.tex").write_text("\n".join(lines) + "\n")

REG_T = {"full": "full image", "dyn": "dynamic region"}
SET_T = {"A": "(A) training cameras, mean of 17", "B": "(B) held-out cam00"}


def perframe_fig(relative, fname, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex=True, sharey="row" if not relative else True)
    for (s, reg), ax in zip([("A", "full"), ("B", "full"), ("A", "dyn"), ("B", "dyn")], axes.ravel()):
        for m in METHODS:
            y = PF[m][(s, reg)]
            y = y - y[train].mean() if relative else y
            lab = m
            ax.plot(t, y, color=COL[m], lw=1.6, label=lab)
            ax.plot(train, y[train], "o", color=COL[m], ms=5)
            ax.plot(test, y[test], "s", color=COL[m], mfc="white", ms=5)
        for x in train:
            ax.axvline(x, color="grey", lw=0.5, ls=":")
        if relative:
            ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{SET_T[s]}, {REG_T[reg]}", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("frame index")
    ylab = "PSNR minus training-frame mean (dB)" if relative else "PSNR (dB)"
    for ax in axes[:, 0]:
        ax.set_ylabel(ylab)
    axes[0, 0].legend(fontsize=8, loc="lower right", title="filled = training frame, open = test frame",
                      title_fontsize=7.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=170)
    plt.close(fig)


perframe_fig(False, "fig_perframe_k8_uniform.png",
             "k = 8, per-frame PSNR over all 50 frames (training frames 0, 8, ..., 48; test frames 4, 12, ..., 44)")
perframe_fig(True, "fig_perframe_k8_relative.png",
             "k = 8, PSNR relative to each method's mean training-frame PSNR")
tad_a, tad_b = PF["TAD-GS"][("A", "full")], PF["TAD-GS"][("B", "full")]
g_a, g_b = PF["4DGS"][("A", "full")], PF["4DGS"][("B", "full")]

# ---------------------------------------------------------------- TAD-GS cam00, all 50 frames, k=1,2,4,8
mask_p = ROOT / "TADGS/data/neural_3d_video/flow_masks/coffee_martini/otsu_bbox_masks/mask_cam00_frame0020.png"
rend_root = ROOT / "TADGS/results"
pf = {}
if mask_p.exists() and all((rend_root / f"renders_k{k}_cam00_all50/renders").exists() for k in (1, 2, 4, 8)):
    m = cv2.imread(str(mask_p), 0) > 127
    for k in (1, 2, 4, 8):
        rows = []
        for i in t:
            fn = f"cam00_frame{i:04d}.png"
            r = cv2.imread(str(rend_root / f"renders_k{k}_cam00_all50/renders/{fn}")).astype(np.float64) / 255
            g = cv2.imread(str(rend_root / f"renders_k{k}_cam00_all50/gt/{fn}")).astype(np.float64) / 255
            e = (r - g) ** 2
            rows.append((10 * np.log10(1 / e.mean()), 10 * np.log10(1 / e[m].mean())))
        pf[k] = np.array(rows)
    with open(TAB / "tadgs_perframe_cam00_allk.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"k{k}_{c}" for k in (1, 2, 4, 8) for c in ("PSNR", "dynPSNR")])
        for i in t:
            w.writerow([i] + [f"{pf[k][i, j]:.3f}" for k in (1, 2, 4, 8) for j in (0, 1)])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    cols = {1: "#7f7f7f", 2: "#2ca02c", 4: "#ff7f0e", 8: "#1f77b4"}
    for k in (1, 2, 4, 8):
        lab = "k = 1 (all 50 frames trained)" if k == 1 else f"k = {k}"
        ax.plot(t, pf[k][:, 0], color=cols[k], label=lab, lw=1.6)
    for x in (0, 24, 48):
        ax.axvspan(x - 0.4, x + 0.4, color="purple", alpha=0.15)
    ax.set_xlabel("frame index")
    ax.set_ylabel("cam00 PSNR (dB)")
    ax.set_title("TAD-GS, held-out cam00, all 50 frames. Shaded: point-cloud anchor frames {0, 24, 48}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_tadgs_perframe_cam00_allk.png", dpi=180)
    plt.close(fig)
else:
    print("[skip] TAD-GS all-50 renders not found; fig_tadgs_perframe_cam00_allk.png not regenerated")

# ---------------------------------------------------------------- copy the mask figure (downscaled)
COPIES = {
    "tadgs_mask_cam00.jpg": "TADGS/results/dynamic_mask_visualization.png",
}
for dst, src in COPIES.items():
    im = Image.open(ROOT / src).convert("RGB")
    if im.width > 2400:
        im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
    im.save(FIG / dst, quality=90)

# raw data copies, so the folder is self-contained
RAW = TAB / "raw"
RAW.mkdir(exist_ok=True)
shutil.copy(ROOT / "TADGS/results/tadgs_evaluation_metrics.json", RAW / "tadgs_evaluation_metrics.json")
for old in RAW.glob("4dgs_*"):
    old.unlink()
for k in KS:
    for fn in ("midpoint_metrics.csv", "midpoint_bbox_metrics.csv"):
        shutil.copy(LR / f"k{k}/{fn}", RAW / f"4dgs_676_k{k}_{fn}")
for fn in ("all50_cam00.csv", "all50_train_cameras.csv"):
    shutil.copy(LR / f"k8/{fn}", RAW / f"4dgs_676_k8_{fn}")
shutil.copy(ROOT / "STGS/results/metrics_per_image.csv", RAW / "stgs_metrics_per_image.csv")
shutil.copy(ROOT / "STGS/results/perframe_psnr_k8.csv", RAW / "stgs_perframe_psnr_k8.csv")

# summary numbers used in the report text
summ = {}
for mth in METHODS:
    for s in "AB":
        summ[f"{mth}_{s}_drop_k2_k8"] = round(R[(mth, 2, s)]["PSNR"] - R[(mth, 8, s)]["PSNR"], 2)
        summ[f"{mth}_{s}_dyndrop_k2_k8"] = round(R[(mth, 2, s)]["dPSNR"] - R[(mth, 8, s)]["dPSNR"], 2)
summ["TADGS_k8_A_train_vs_test"] = (round(tad_a[train].mean(), 2), round(tad_a[test].mean(), 2))
summ["TADGS_k8_B_train_vs_test"] = (round(tad_b[train].mean(), 2), round(tad_b[test].mean(), 2))
summ["4DGS_k8_A_train_vs_test"] = (round(g_a[train].mean(), 2), round(g_a[test].mean(), 2))
summ["4DGS_k8_B_train_vs_test"] = (round(g_b[train].mean(), 2), round(g_b[test].mean(), 2))
if pf:
    for k in (1, 2, 4, 8):
        summ[f"TADGS_cam00_k{k}_mean_all50"] = round(pf[k][:, 0].mean(), 2)
    near = [i for i in t if 0 < abs(i - 24) <= 3 or 0 < abs(i - 48) <= 3]
    far = [i for i in t if i not in near and i not in (0, 24, 48)]
    for k in (4, 8):
        summ[f"TADGS_cam00_k{k}_near_anchor_vs_rest"] = (round(pf[k][near, 0].mean(), 2),
                                                       round(pf[k][far, 0].mean(), 2))
(TAB / "summary_numbers.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
print("wrote", TAB, FIG)
