#!/usr/bin/env python3
"""Deliverable 4: Visual comparison figures for 4DGS across k in {2, 4, 8}."""

from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"

CONFIGS = [
    (2, 21, "k = 2 (midpoint t = 21, gap 20-22)"),
    (4, 22, "k = 4 (midpoint t = 22, gap 20-24)"),
    (8, 20, "k = 8 (midpoint t = 20, gap 16-24)"),
]

IMAGE_W, IMAGE_H = 676, 507
CROP_W, CROP_H = 180, 150
CROP_CX, CROP_CY = 340, 300


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_gray_as_float(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise OSError(f"Could not read: {path}")
    return gray.astype(np.float32) / 255.0


def main():
    x0 = max(0, CROP_CX - CROP_W // 2)
    y0 = max(0, CROP_CY - CROP_H // 2)
    x0 = min(x0, IMAGE_W - CROP_W)
    y0 = min(y0, IMAGE_H - CROP_H)
    x1, y1 = x0 + CROP_W, y0 + CROP_H

    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))

    for row, (k, frame, label) in enumerate(CONFIGS):
        eval_dir = (
            OUTPUT_ROOT
            / f"coffee_martini_k{k}"
            / "midpoint_evaluation_v4"
            / f"gap_{k}"
        )
        pred_path = eval_dir / f"heldout_cam00_{frame:04d}_pred.png"
        gt_path = eval_dir / f"heldout_cam00_{frame:04d}_gt.png"

        pred = load_rgb(pred_path)
        gt = load_rgb(gt_path)

        err = np.mean(np.abs(pred - gt), axis=-1)
        mse = np.mean((pred - gt) ** 2)
        psnr_val = 10.0 * np.log10(1.0 / max(mse, 1e-10))

        # Col 0: Ground truth with crop box
        ax0 = axes[row, 0]
        ax0.imshow(gt)
        rect = patches.Rectangle(
            (x0, y0), CROP_W, CROP_H,
            linewidth=2, edgecolor="#00ff88", facecolor="none",
        )
        ax0.add_patch(rect)
        ax0.set_title(
            f"Ground Truth (cam00, t={frame})",
            fontsize=10.5, fontweight="bold",
        )
        ax0.set_ylabel(label, fontsize=10, fontweight="bold")
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Col 1: 4DGS render with crop box
        ax1 = axes[row, 1]
        ax1.imshow(pred)
        rect = patches.Rectangle(
            (x0, y0), CROP_W, CROP_H,
            linewidth=2, edgecolor="#00ff88", facecolor="none",
        )
        ax1.add_patch(rect)
        ax1.set_title(
            f"4DGS Render (k={k}, PSNR={psnr_val:.2f} dB)",
            fontsize=10.5, fontweight="bold",
        )
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Col 2: Error heatmap (4x amplified)
        ax2 = axes[row, 2]
        im = ax2.imshow(
            np.clip(err * 4.0, 0.0, 1.0),
            cmap="inferno", vmin=0.0, vmax=1.0,
        )
        rect = patches.Rectangle(
            (x0, y0), CROP_W, CROP_H,
            linewidth=1.8, edgecolor="#00ff88", facecolor="none",
        )
        ax2.add_patch(rect)
        ax2.set_title(
            f"Absolute Error (4x scaled, k={k})",
            fontsize=10.5, fontweight="bold",
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Col 3: Zoomed crop -- GT | Render | Error side by side
        ax3 = axes[row, 3]
        gt_crop = gt[y0:y1, x0:x1]
        pred_crop = pred[y0:y1, x0:x1]
        err_crop = err[y0:y1, x0:x1]
        err_crop_color = matplotlib.cm.inferno(
            np.clip(err_crop * 4.0, 0.0, 1.0)
        )[..., :3]

        sep = np.ones((CROP_H, 3, 3), dtype=np.float32)
        composite = np.concatenate(
            [gt_crop, sep, pred_crop, sep, err_crop_color], axis=1,
        )
        ax3.imshow(composite)
        ax3.set_title(
            "Dynamic region (GT | Render | Error)",
            fontsize=10.5, fontweight="bold",
        )
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.suptitle(
        "4DGS Visual Comparison on Held-Out Camera (cam00) "
        "Across Frame Gaps k = {2, 4, 8}",
        fontsize=13.5, fontweight="bold", y=0.995,
    )
    plt.tight_layout()

    out_dir = OUTPUT_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "visual_comparison_k2_k4_k8.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
