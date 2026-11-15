# [A3] Merge results_common/per_image_k{2,4,8}.csv -> metrics_per_image.csv, metrics_summary.{csv,md}.
# Averages = mean of per-image metrics over the TEST frames (and, for reference, the training frames).
# dyn_* averages skip images whose mask is empty / crop too small (NaN); n_dyn reports how many were used.
# Usage: python analysis/summarize_common.py --dir results_common
import os, csv, math, argparse

M = ["psnr", "ssim", "lpips", "dyn_psnr", "dyn_ssim", "dyn_lpips"]


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    a = ap.parse_args()

    rows = []
    for k in a.ks:
        with open(os.path.join(a.dir, "per_image_k{}.csv".format(k))) as f:
            for r in csv.DictReader(f):
                for m in M + ["dyn_frac"]:
                    r[m] = float(r[m]) if r[m] not in ("", "nan", "None") else float("nan")
                rows.append(r)
    with open(os.path.join(a.dir, "metrics_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summ = []
    for k in a.ks:
        for cs in ["train", "heldout"]:
            for ts in ["test", "train"]:
                sel = [r for r in rows if int(r["k"]) == k and r["cam_split"] == cs and r["time_split"] == ts]
                d = dict(k=k, cameras=cs, frames=ts, n_images=len(sel))
                for m in M:
                    d[m], n = mean([r[m] for r in sel])
                    if m.startswith("dyn_"):
                        d["n_" + m] = n
                d["dyn_frac"] = mean([r["dyn_frac"] for r in sel])[0]
                summ.append(d)
    with open(os.path.join(a.dir, "metrics_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    L = ["# STGS full, coffee_martini 676x507, frames 0-49, iteration 30000 - team-common dynamic mask", "",
         "Full-image metrics: STGS repo (psnr, ssim, LPIPS-alex). Dynamic: TADGS bbox mask (flow t->t+2, tau 0.5 px).",
         "A = training cams at test frames, B = cam00 at test frames. Averages over test frames.", "",
         "| k | setting | #img | PSNR | SSIM | LPIPS | dyn PSNR | dyn SSIM | dyn LPIPS | #img dyn | mask area |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in summ:
        if d["frames"] == "test":
            L.append("| {} | {} | {} | {:.2f} | {:.4f} | {:.4f} | {:.2f} | {:.4f} | {:.4f} | {} | {:.1%} |".format(
                d["k"], "A" if d["cameras"] == "train" else "B (cam00)", d["n_images"], d["psnr"], d["ssim"], d["lpips"],
                d["dyn_psnr"], d["dyn_ssim"], d["dyn_lpips"], d["n_dyn_psnr"], d["dyn_frac"]))
    L += ["", "Reference: training frames", "", "| k | setting | PSNR | dyn PSNR |", "|---|---|---|---|"]
    for d in summ:
        if d["frames"] == "train":
            L.append("| {} | {} | {:.2f} | {:.2f} |".format(d["k"], "A" if d["cameras"] == "train" else "B (cam00)",
                                                         d["psnr"], d["dyn_psnr"]))
    with open(os.path.join(a.dir, "metrics_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
