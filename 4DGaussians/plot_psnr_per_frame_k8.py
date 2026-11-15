#!/usr/bin/env python3
"""Deliverable 3: Per-frame PSNR plot for k=8 (4DGS)."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
# High-res all-50 data
OUTPUT_DIR = PROJECT_ROOT / "output" / "coffee_martini_k8" / "highres_evaluation" / "gap_8" / "all50"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    cam00_rows = load_csv(OUTPUT_DIR / "cam00_all50.csv")
    train_rows = load_csv(OUTPUT_DIR / "train_cameras_all50.csv")

    frames = [int(r["frame"]) for r in cam00_rows]
    cam00_psnr = [float(r["PSNR"]) for r in cam00_rows]
    train_psnr = [float(r["PSNR"]) for r in train_rows]

    train_frames = [0, 8, 16, 24, 32, 40, 48]
    test_frames = [4, 12, 20, 28, 36, 44]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for tf in train_frames:
        ax.axvline(tf, color="#2ca02c", linestyle="--", alpha=0.25, linewidth=1.0)
    for mf in test_frames:
        ax.axvline(mf, color="#d62728", linestyle=":", alpha=0.25, linewidth=1.0)

    ax.plot(frames, train_psnr, color="#1f77b4", linewidth=2.0,
            label="Training cameras (17-cam mean, full image)")
    ax.plot(frames, cam00_psnr, color="#ff7f0e", linewidth=2.0,
            label="Held-out camera cam00 (full image)")

    ax.scatter(
        train_frames, [train_psnr[t] for t in train_frames],
        color="#2ca02c", s=70, zorder=5, edgecolors="black", linewidth=0.7,
        label="Training timestamps {0, 8, 16, 24, 32, 40, 48}",
    )
    ax.scatter(
        test_frames, [train_psnr[t] for t in test_frames],
        color="#d62728", marker="D", s=65, zorder=5, edgecolors="black", linewidth=0.7,
        label="Midpoint test timestamps {4, 12, 20, 28, 36, 44}",
    )
    ax.scatter(
        train_frames, [cam00_psnr[t] for t in train_frames],
        color="#2ca02c", s=60, zorder=5, edgecolors="black", linewidth=0.7,
    )
    ax.scatter(
        test_frames, [cam00_psnr[t] for t in test_frames],
        color="#d62728", marker="D", s=60, zorder=5, edgecolors="black", linewidth=0.7,
    )

    ax.set_xlabel("Frame index t (0 to 49)", fontsize=11, fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontsize=11, fontweight="bold")
    ax.set_title(
        "4DGS Per-Frame PSNR Across All 50 Frames (k = 8, coffee_martini, 2704×2028)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(range(0, 50, 2))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.92)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "psnr_per_frame_k8_highres.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
