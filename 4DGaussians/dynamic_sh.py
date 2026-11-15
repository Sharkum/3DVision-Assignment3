#!/usr/bin/env python3
"""Reproduce the friend's figure: cam00, frames 20 -> 22, tau in {0.0, 0.5, 1.5}."""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Friend's reported numbers: tau -> (active pixels, bbox x0,y0,x1,y1 half-open)
EXPECTED = {
    0.0: (10033, (0, 0, 676, 507)),
    0.5: (6562, (246, 200, 402, 398)),
    1.5: (839, (262, 317, 291, 397)),
}
EXPECTED_MAX = 1.96


def raw_flow(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        ga, gb, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def compute_flow(a, b):
    fwd = raw_flow(a, b)
    bwd = raw_flow(b, a)
    H, W = a.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    bwd_w = cv2.remap(bwd, gx + fwd[..., 0], gy + fwd[..., 1],
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb_err = np.linalg.norm(fwd + bwd_w, axis=-1)
    fwd_mag = np.linalg.norm(fwd, axis=-1)
    valid_fb = fb_err < (0.5 * fwd_mag + 0.75)
    photo = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0, axis=-1)
    photo = cv2.GaussianBlur(photo, (7, 7), 1.5)
    valid = valid_fb & (photo > 0.012)
    return fwd * valid[..., None].astype(np.float32)


def find_bbox(flow, tau):
    mag = np.linalg.norm(flow, axis=-1)
    active = mag > tau
    ys, xs = np.where(active)
    if xs.size == 0:
        return None, mag, active
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return bbox, mag, active


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img1", type=Path, required=True, help="cam00 frame 20, e.g. coffee_martini/cam00/images/0020.png")
    p.add_argument("--img2", type=Path, required=True, help="cam00 frame 22, e.g. coffee_martini/cam00/images/0022.png")
    p.add_argument("--out", type=Path, default=Path("verify_friend_figure.png"))
    args = p.parse_args()

    img1, img2 = load_rgb(args.img1), load_rgb(args.img2)
    flow = compute_flow(img1, img2)
    thresholds = [0.0, 0.5, 1.5]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10))
    all_match = True
    for row, tau in enumerate(thresholds):
        bbox, mag, active = find_bbox(flow, tau)
        n = int(active.sum())
        exp_n, exp_bbox = EXPECTED[tau]
        ok = (n == exp_n) and (bbox == exp_bbox)
        all_match &= ok
        print(f"tau={tau}: active={n} (friend {exp_n}), bbox={bbox} (friend {exp_bbox}) "
              f"-> {'MATCH' if ok else 'DIFFERENT'}")

        axes[row, 0].imshow(mag, cmap="magma", vmin=0, vmax=max(5.0, float(mag.max())))
        axes[row, 0].set_title(f"Optical Flow Magnitude ||u||_2\n(Max: {mag.max():.2f} px)", fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel("τ = 0.0 px (All Non-Zero)" if tau == 0.0 else f"Threshold τ = {tau} px",
                                fontsize=11, fontweight="bold")

        axes[row, 1].imshow(active.astype(np.uint8) * 255, cmap="gray")
        axes[row, 1].set_title(f"Active Pixels (||u|| > {tau} px)\n({n} active pixels)", fontsize=10, fontweight="bold")

        overlay = img1.copy()
        if bbox:
            x0, y0, x1, y1 = bbox
            axes[row, 1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, lw=2.2, ec="#ff3333", fc="none"))
            dark = (overlay.astype(np.float32) * 0.45).astype(np.uint8)
            dark[y0:y1, x0:x1] = overlay[y0:y1, x0:x1]
            overlay = dark
        axes[row, 2].imshow(overlay)
        if bbox:
            axes[row, 2].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, lw=2.5, ec="#00ff88", fc="none"))
            axes[row, 2].set_title(f"Smallest Enclosing BBox: {x1-x0}×{y1-y0} px\n[{x0}:{x1}, {y0}:{y1}]",
                                   fontsize=10, fontweight="bold")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    print(f"flow magnitude max = {np.linalg.norm(flow, axis=-1).max():.2f} (friend {EXPECTED_MAX})")
    print("ALL MATCH" if all_match else "SOME RESULTS DIFFER")
    plt.suptitle("Smallest Enclosing Bounding Box of Optical Flow > τ (cam00, t=20→22)",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()