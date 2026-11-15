# [A3] Report item 3: per-frame PSNR vs frame index for k=8 over ALL frames 0..49 (training + test frames).
# Reads results/metrics_per_image.csv (from masked_metrics.py). Needs matplotlib (installed on the PC; the server
# env stg_a3 may not have it -> run this on the PC:  python Z:\sohamg\assigment3\analysis\plot_perframe.py).
# Two panels: (A) mean over the 17 training cameras, (B) held-out cam00. Each panel: full-image PSNR and
# dynamic-region PSNR. Training frames = filled circles + dotted vertical lines, test frames = open squares.
# Outputs: results/perframe_psnr_k<k>.png and .pdf, results/perframe_psnr_k<k>.csv (the plotted numbers).
import os, sys, csv, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3common as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.DEFAULT_BASE)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--results", default=None, help="folder with metrics_per_image.csv (default results/); e.g. results_common")
    a = ap.parse_args()
    P = C.paths(a.base)
    RES = a.results or P["results"]

    acc = defaultdict(list)  # (cam_split, metric, frame) -> values
    with open(os.path.join(RES, "metrics_per_image.csv")) as fh:
        for r in csv.DictReader(fh):
            if int(r["k"]) != a.k:
                continue
            for m in ["psnr", "dyn_psnr"]:
                if r[m] not in ("", "None", "nan"):
                    acc[(r["cam_split"], m, int(r["frame"]))].append(float(r[m]))
    frames = list(range(C.DURATION))
    trainf, testf = C.train_frames_for(a.k), C.test_frames_for(a.k)

    def series(cs, m):
        return [sum(acc[(cs, m, f)]) / len(acc[(cs, m, f)]) if acc[(cs, m, f)] else float("nan") for f in frames]

    data = {(cs, m): series(cs, m) for cs in ["train", "heldout"] for m in ["psnr", "dyn_psnr"]}
    with open(os.path.join(RES, "perframe_psnr_k{}.csv".format(a.k)), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_split", "traincams_psnr", "traincams_dyn_psnr", "cam00_psnr", "cam00_dyn_psnr"])
        for i, f in enumerate(frames):
            w.writerow([f, C.time_split(f, a.k), data[("train", "psnr")][i], data[("train", "dyn_psnr")][i],
                        data[("heldout", "psnr")][i], data[("heldout", "dyn_psnr")][i]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    titles = {"train": "(A) training cameras (mean of 17) at all times", "heldout": "(B) held-out camera cam00"}
    style = {"psnr": ("tab:blue", "full image"), "dyn_psnr": ("tab:red", "dynamic region")}
    for ax, cs in zip(axes, ["train", "heldout"]):
        for f in trainf:
            ax.axvline(f, color="0.85", ls=":", lw=1, zorder=0)
        for m in ["psnr", "dyn_psnr"]:
            col, lab = style[m]
            y = data[(cs, m)]
            ax.plot(frames, y, color=col, lw=1.2, label=lab)
            ax.scatter(trainf, [y[f] for f in trainf], color=col, s=28, zorder=3)
            ax.scatter(testf, [y[f] for f in testf], facecolors="white", edgecolors=col, marker="s", s=30, zorder=3)
        ax.set_title(titles[cs], fontsize=10)
        ax.set_xlabel("frame index")
        ax.set_xlim(-1, C.DURATION)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("PSNR (dB)")
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], color="k", marker="o", ls="", label="training frame (t % {} = 0)".format(a.k)),
             Line2D([], [], color="k", marker="s", mfc="white", ls="", label="test frame (midpoint)")]
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h + extra, l + [e.get_label() for e in extra], fontsize=8, loc="lower right")
    fig.suptitle("STGS, coffee_martini, k = {}: per-frame PSNR".format(a.k), fontsize=11)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        out = os.path.join(RES, "perframe_psnr_k{}.{}".format(a.k, ext))
        fig.savefig(out, dpi=200)
        print("saved", out)
