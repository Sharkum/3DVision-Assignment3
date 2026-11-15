import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE / "results"


def plot_per_frame_psnr_k8(metrics: dict):
    if "k=8" not in metrics or "all_50_frames" not in metrics["k=8"]:
        print("[Skip] k=8 all_50_frames data not found.")
        return

    all_50 = metrics["k=8"]["all_50_frames"]
    frames = sorted(int(k) for k in all_50.keys())
    cam00_psnr = [all_50[str(t) if str(t) in all_50 else t]["cam00_PSNR"] for t in frames]
    train_psnr = [all_50[str(t) if str(t) in all_50 else t]["train_cams_PSNR"] for t in frames]
    cam00_mpsnr = [all_50[str(t) if str(t) in all_50 else t]["cam00_Masked_PSNR"] for t in frames]

    train_frames = metrics["k=8"]["train_frames"]
    test_frames = metrics["k=8"]["test_frames"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]})

    # Top panel: Training cameras (temporal interpolation only) vs Held-out camera (temporal + spatial)
    ax = axes[0]
    for tf in train_frames:
        ax.axvline(tf, color="#2ca02c", linestyle="--", alpha=0.28, linewidth=1.0)
    for mf in test_frames:
        ax.axvline(mf, color="#d62728", linestyle=":", alpha=0.28, linewidth=1.0)

    ax.plot(frames, train_psnr, color="#1f77b4", linewidth=2.0, label="Training Cameras (cam01–cam17 Mean, Full Frame)")
    ax.scatter(
        train_frames, [train_psnr[t] for t in train_frames],
        color="#2ca02c", s=70, zorder=5, edgecolors="black", linewidth=0.7,
        label="Training Timestamps {0, 8, 16, 24, 32, 40, 48}"
    )
    ax.scatter(
        test_frames, [train_psnr[t] for t in test_frames],
        color="#d62728", marker="D", s=65, zorder=5, edgecolors="black", linewidth=0.7,
        label="Midpoint Test Timestamps {4, 12, 20, 28, 36, 44}"
    )

    ax.plot(frames, cam00_psnr, color="#ff7f0e", linewidth=2.0, linestyle="-", label="Held-out Camera (cam00, Full Frame)")
    ax.scatter(
        train_frames, [cam00_psnr[t] for t in train_frames],
        color="#2ca02c", s=60, zorder=5, edgecolors="black", linewidth=0.7
    )
    ax.scatter(
        test_frames, [cam00_psnr[t] for t in test_frames],
        color="#d62728", marker="D", s=60, zorder=5, edgecolors="black", linewidth=0.7
    )

    ax.set_ylabel("PSNR (dB) ↑", fontsize=11, fontweight="bold")
    ax.set_title("TAD-GS Per-Frame PSNR Across Training & Held-Out Test Frames (k = 8, coffee_martini)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.92)

    # Bottom panel: Held-out camera Full PSNR vs Dynamic-Region Masked PSNR
    ax2 = axes[1]
    for tf in train_frames:
        ax2.axvline(tf, color="#2ca02c", linestyle="--", alpha=0.28, linewidth=1.0)
    for mf in test_frames:
        ax2.axvline(mf, color="#d62728", linestyle=":", alpha=0.28, linewidth=1.0)

    ax2.plot(frames, cam00_psnr, color="#ff7f0e", linewidth=1.8, label="Held-out Camera (cam00) — Full Frame PSNR")
    ax2.plot(frames, cam00_mpsnr, color="#9467bd", linewidth=2.0, label="Held-out Camera (cam00) — Dynamic Region Masked PSNR")
    ax2.scatter(
        train_frames, [cam00_mpsnr[t] for t in train_frames],
        color="#2ca02c", s=60, zorder=5, edgecolors="black", linewidth=0.7, label="Observed Training Timestamps"
    )
    ax2.scatter(
        test_frames, [cam00_mpsnr[t] for t in test_frames],
        color="#d62728", marker="D", s=60, zorder=5, edgecolors="black", linewidth=0.7, label="Held-out Midpoint Timestamps"
    )

    ax2.set_xlabel("Frame Index t (0 to 49)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("PSNR (dB) ↑", fontsize=11, fontweight="bold")
    ax2.set_title("Effect of Temporal Interpolation on Dynamic Regions vs. Full Frame (k = 8, Held-Out cam00)", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(range(0, 50, 2))
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.92)

    plt.tight_layout()
    out_path = RESULTS_DIR / "psnr_per_frame_k8.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Deliverable 3 plot to {out_path}")


def generate_visual_comparison():
    """Generate Deliverable 4 visual comparison figure for k in {2, 4, 8}:
    Shows Ground Truth, TAD-GS Render, Error Map (|Render - GT|), and Zoomed Fast-Moving Dynamic Region."""
    # We generate two visual figures:
    # 1. Exact midpoint test frame for each k near the fast pouring action:
    #    k=2 -> t=21 (midpoint of 20,22), k=4 -> t=22 (midpoint of 20,24), k=8 -> t=20 (midpoint of 16,24)
    # 2. Common held-out frame t=20 across all k in {2, 4, 8} (or each k's own midpoint)
    configs = [
        (2, 21, "k = 2 (Midpoint Test Frame t = 21)"),
        (4, 22, "k = 4 (Midpoint Test Frame t = 22)"),
        (8, 20, "k = 8 (Midpoint Test Frame t = 20)"),
    ]

    # Fast-moving region bounding box in 676x507 (the bartender's hands/shaker/pouring stream in center-right)
    # Find centroid of optical flow dynamic mask at t=20 on cam00 automatically!
    mask_path = WORKSPACE / "data" / "neural_3d_video" / "flow_masks" / "coffee_martini" / "valid_masks" / "mask_cam00_frame0020.png"
    H, W = 507, 676
    crop_w, crop_h = 160, 140
    if mask_path.exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(m > 127)
        if len(xs) > 0:
            cx, cy = int(np.median(xs)), int(np.median(ys))
        else:
            cx, cy = W // 2, H // 2
    else:
        cx, cy = W // 2, H // 2

    x0 = int(np.clip(cx - crop_w // 2, 0, W - crop_w))
    y0 = int(np.clip(cy - crop_h // 2, 0, H - crop_h))
    x1, y1 = x0 + crop_w, y0 + crop_h

    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))

    for row, (k, frame_idx, label) in enumerate(configs):
        render_file = RESULTS_DIR / f"renders_k{k}_cam00" / "renders" / f"cam00_frame{frame_idx:04d}.png"
        gt_file = RESULTS_DIR / f"renders_k{k}_cam00" / "gt" / f"cam00_frame{frame_idx:04d}.png"
        if not render_file.exists() or not gt_file.exists():
            print(f"[Skip] Visual files missing for k={k}, frame={frame_idx}")
            continue

        rend_bgr = cv2.imread(str(render_file))
        gt_bgr = cv2.imread(str(gt_file))
        rend_rgb = cv2.cvtColor(rend_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        err = np.mean(np.abs(rend_rgb - gt_rgb), axis=-1)
        mse = np.mean((rend_rgb - gt_rgb) ** 2)
        psnr_val = 10.0 * np.log10(1.0 / max(mse, 1e-10))

        # Col 0: Ground Truth with bounding box on fast-moving region
        ax0 = axes[row, 0]
        ax0.imshow(gt_rgb)
        rect0 = patches.Rectangle((x0, y0), crop_w, crop_h, linewidth=2, edgecolor="#00ff88", facecolor="none")
        ax0.add_patch(rect0)
        ax0.set_title(f"Ground Truth (cam00, t={frame_idx})", fontsize=10.5, fontweight="bold")
        ax0.set_ylabel(label, fontsize=11, fontweight="bold")
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Col 1: TAD-GS Rendered Test Frame
        ax1 = axes[row, 1]
        ax1.imshow(rend_rgb)
        rect1 = patches.Rectangle((x0, y0), crop_w, crop_h, linewidth=2, edgecolor="#00ff88", facecolor="none")
        ax1.add_patch(rect1)
        ax1.set_title(f"TAD-GS Render (k={k}, PSNR={psnr_val:.2f} dB)", fontsize=10.5, fontweight="bold")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Col 2: L1 Error Heatmap (amplified 4x for visibility)
        ax2 = axes[row, 2]
        im2 = ax2.imshow(np.clip(err * 4.0, 0.0, 1.0), cmap="inferno", vmin=0.0, vmax=1.0)
        rect2 = patches.Rectangle((x0, y0), crop_w, crop_h, linewidth=1.8, edgecolor="#00ff88", facecolor="none")
        ax2.add_patch(rect2)
        ax2.set_title(f"Absolute Error Map (4× scaled, k={k})", fontsize=10.5, fontweight="bold")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Col 3: Side-by-side Zoomed Fast-Moving Region (Left: GT, Right: TAD-GS Render, Bottom: Error)
        ax3 = axes[row, 3]
        gt_crop = gt_rgb[y0:y1, x0:x1]
        rend_crop = rend_rgb[y0:y1, x0:x1]
        err_crop_color = matplotlib.cm.inferno(np.clip(err[y0:y1, x0:x1] * 4.0, 0.0, 1.0))[..., :3]
        sep = np.ones((crop_h, 3, 3), dtype=np.float32)
        composite_crop = np.concatenate([gt_crop, sep, rend_crop, sep, err_crop_color], axis=1)
        ax3.imshow(composite_crop)
        ax3.set_title("Fast-Moving Region (GT | Render | Error)", fontsize=10.5, fontweight="bold")
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.suptitle("TAD-GS Visual Comparison & Error Maps on Held-Out Camera (cam00) Across Frame Gaps k ∈ {2, 4, 8}",
                 fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_path = RESULTS_DIR / "visual_comparison_k2_k4_k8.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Deliverable 4 visual comparison figure to {out_path}")


def generate_dynamic_mask_figure():
    """Save an illustration of the Otsu optical-flow average bounding box mask on cam00 at t=20."""
    gt_file = WORKSPACE / "data" / "neural_3d_video" / "scenes" / "coffee_martini" / "colmap_20" / "images" / "cam00.png"
    mask_file = WORKSPACE / "data" / "neural_3d_video" / "flow_masks" / "coffee_martini" / "otsu_bbox_masks" / "mask_cam00_frame0020.png"
    if not gt_file.exists() or not mask_file.exists():
        print(f"[Warning] Mask file or GT file missing: {gt_file}, {mask_file}")
        return

    gt_rgb = cv2.cvtColor(cv2.imread(str(gt_file)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)

    overlay = gt_rgb.copy()
    # Dim background slightly and highlight dynamic box
    overlay[mask < 0.5] = overlay[mask < 0.5] * 0.45
    ys, xs = np.where(mask > 0.5)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth Frame (cam00, t=20)", fontweight="bold", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    rect1 = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.2, edgecolor="#ff3333", facecolor="none")
    axes[1].add_patch(rect1)
    axes[1].set_title(f"Otsu Average Flow BBox Mask\nBox: {x1-x0}x{y1-y0} px ({mask.mean()*100:.1f}% pixels)", fontweight="bold", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    rect2 = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.5, edgecolor="#00ff88", facecolor="none")
    axes[2].add_patch(rect2)
    axes[2].set_title(f"Encapsulated Dynamic Region Overlay\n[{x0}:{x1}, {y0}:{y1}]", fontweight="bold", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = RESULTS_DIR / "dynamic_mask_visualization.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved updated dynamic mask visualization to {out_path}")


def write_summary_tables(metrics: dict):
    md_lines = [
        "# TAD-GS Quantitative Evaluation Results (`coffee_martini`, 50 Frames, 4× Downsampling)\n",
        "## Deliverable 1: Full-Image Quality on Held-Out Test Frames\n",
        "| Frame Gap $k$ | Train Frames | Midpoint Test Frames | Setting 1: Train Cams, Held-Out Time (PSNR ↑ / SSIM ↑ / LPIPS ↓) | Setting 2: Held-Out Cam (`cam00`), Held-Out Time (PSNR ↑ / SSIM ↑ / LPIPS ↓) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for k in [2, 4, 8]:
        key = f"k={k}"
        if key not in metrics:
            continue
        m = metrics[key]
        s1 = m["setting1_train_cams_heldout_time"]
        s2 = m["setting2_heldout_cam_heldout_time"]
        md_lines.append(
            f"| $k = {k}$ | {len(m['train_frames'])} | {len(m['test_frames'])} | "
            f"**{s1['PSNR']:.2f}** / **{s1['SSIM']:.4f}** / **{s1['LPIPS']:.4f}** | "
            f"**{s2['PSNR']:.2f}** / **{s2['SSIM']:.4f}** / **{s2['LPIPS']:.4f}** |"
        )

    md_lines.extend([
        "\n## Deliverable 2: Dynamic-Region Quality (Optical-Flow Motion Mask)\n",
        "| Frame Gap $k$ | Setting 1: Train Cams, Held-Out Time (Masked PSNR ↑ / Masked SSIM ↑ / Masked LPIPS ↓) | Setting 2: Held-Out Cam (`cam00`), Held-Out Time (Masked PSNR ↑ / Masked SSIM ↑ / Masked LPIPS ↓) |",
        "| :---: | :---: | :---: |",
    ])
    for k in [2, 4, 8]:
        key = f"k={k}"
        if key not in metrics:
            continue
        m = metrics[key]
        s1 = m["setting1_train_cams_heldout_time"]
        s2 = m["setting2_heldout_cam_heldout_time"]
        md_lines.append(
            f"| $k = {k}$ | "
            f"**{s1['Masked-PSNR']:.2f}** / **{s1['Masked-SSIM']:.4f}** / **{s1['Masked-LPIPS']:.4f}** | "
            f"**{s2['Masked-PSNR']:.2f}** / **{s2['Masked-SSIM']:.4f}** / **{s2['Masked-LPIPS']:.4f}** |"
        )

    out_md = WORKSPACE / "report" / "summary_tables.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Saved Markdown summary tables to {out_md}")


def main():
    metrics_file = RESULTS_DIR / "tadgs_evaluation_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"{metrics_file} not found. Run scripts/evaluate_all.py first.")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    plot_per_frame_psnr_k8(metrics)
    generate_visual_comparison()
    generate_dynamic_mask_figure()
    write_summary_tables(metrics)


if __name__ == "__main__":
    main()

