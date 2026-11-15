#!/usr/bin/env python3
"""v5: Video-level average optical flow + Otsu threshold for dynamic bounding box
detection. Uses high-resolution (2704x2028) ground truth images.

Approach:
  1. For each camera, compute Farneback optical flow (with forward-backward
     consistency + photometric filtering) for every frame pair (t, t+stride).
  2. Average the flow magnitudes across all 50 frames → single per-camera
     temporal-mean flow map.
  3. Apply Otsu's automatic threshold on the averaged map.
  4. Extract the minimal axis-aligned bounding box enclosing all pixels above
     the Otsu threshold.
  5. This yields one static bounding box per camera (same for all frames).
  6. Evaluate 4DGS predictions against high-res GT within these boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CAMERAS = (
    "cam00", "cam01", "cam02", "cam04", "cam05", "cam06", "cam07",
    "cam08", "cam09", "cam10", "cam11", "cam12", "cam13", "cam14",
    "cam16", "cam18", "cam19", "cam20",
)
TRAIN_CAMERAS = tuple(c for c in CAMERAS if c != "cam00")
GAPS = (2, 4, 8)
FRAME_COUNT = 50
HIGHRES_W, HIGHRES_H = 2704, 2028
LOWRES_W, LOWRES_H = 676, 507
DOWNSAMPLE_FACTOR = 4

FLOW_STRIDE = 2

FARNEBACK = {
    "pyr_scale": 0.5,
    "levels": 5,
    "winsize": 21,
    "iterations": 5,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "flags": 0,
}
FB_ALPHA = 0.5
FB_BETA = 0.75
PHOTO_THRESHOLD = 0.012
PHOTO_BLUR_KERNEL = (11, 11)
PHOTO_BLUR_SIGMA = 2.0

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v5: Otsu average-flow dynamic bbox on high-res images."
    )
    parser.add_argument(
        "--highres-root", type=Path,
        default=PROJECT_ROOT.parent / "coffee_martini_highres",
    )
    parser.add_argument(
        "--lowres-root", type=Path,
        default=PROJECT_ROOT.parent / "coffee_martini",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "output" / "dynamic_bbox_evaluation_v5_otsu",
    )
    parser.add_argument(
        "--stage", choices=("diagnostic", "evaluate", "all"), default="all",
    )
    parser.add_argument(
        "--cameras", nargs="+", default=None,
        help="Specific cameras to process (default: all 18)",
    )
    return parser.parse_args()


def load_highres_rgb(highres_root: Path, camera: str, frame: int) -> np.ndarray:
    path = highres_root / camera / "images" / f"{frame:04d}.png"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    assert rgb.shape == (HIGHRES_H, HIGHRES_W, 3), f"Unexpected shape: {rgb.shape}"
    return rgb


def load_lowres_rgb(lowres_root: Path, camera: str, frame: int) -> np.ndarray:
    path = lowres_root / camera / "images" / f"{frame:04d}.png"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def raw_farneback(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=FARNEBACK["pyr_scale"],
        levels=FARNEBACK["levels"],
        winsize=FARNEBACK["winsize"],
        iterations=FARNEBACK["iterations"],
        poly_n=FARNEBACK["poly_n"],
        poly_sigma=FARNEBACK["poly_sigma"],
        flags=FARNEBACK["flags"],
    )


def compute_filtered_flow_magnitude(
    img_a_rgb: np.ndarray, img_b_rgb: np.ndarray
) -> np.ndarray:
    """Compute forward flow magnitude with FB consistency and photometric filtering."""
    gray_a = cv2.cvtColor(img_a_rgb, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b_rgb, cv2.COLOR_RGB2GRAY)

    forward = raw_farneback(gray_a, gray_b)
    backward = raw_farneback(gray_b, gray_a)
    H, W = img_a_rgb.shape[:2]

    grid_x, grid_y = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    map_x = grid_x + forward[..., 0]
    map_y = grid_y + forward[..., 1]
    backward_warped = cv2.remap(
        backward, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    fb_error = np.linalg.norm(forward + backward_warped, axis=-1)
    forward_mag = np.linalg.norm(forward, axis=-1)
    valid_fb = fb_error < (FB_ALPHA * forward_mag + FB_BETA)

    photo_diff = np.mean(
        np.abs(img_a_rgb.astype(np.float32) - img_b_rgb.astype(np.float32)) / 255.0,
        axis=-1,
    )
    photo_diff = cv2.GaussianBlur(photo_diff, PHOTO_BLUR_KERNEL, PHOTO_BLUR_SIGMA)
    valid_photo = photo_diff > PHOTO_THRESHOLD

    valid = valid_fb & valid_photo
    filtered_mag = forward_mag * valid.astype(np.float32)
    return filtered_mag


def compute_otsu_bbox_for_camera(
    highres_root: Path, camera: str,
) -> dict[str, Any]:
    """Compute video-level average optical flow on high-res frames and return
    the Otsu-thresholded bounding box."""
    frames = []
    for t in range(FRAME_COUNT):
        frames.append(load_highres_rgb(highres_root, camera, t))

    flow_mag_accum = np.zeros((HIGHRES_H, HIGHRES_W), dtype=np.float64)
    count = 0
    for t in range(FRAME_COUNT):
        t_other = min(t + FLOW_STRIDE, FRAME_COUNT - 1)
        if t_other == t:
            t_other = max(0, t - FLOW_STRIDE)
        if t_other == t:
            continue
        mag = compute_filtered_flow_magnitude(frames[t], frames[t_other])
        flow_mag_accum += mag.astype(np.float64)
        count += 1

    avg_flow_mag = (flow_mag_accum / max(count, 1)).astype(np.float32)
    max_val = float(avg_flow_mag.max())

    if max_val > 1e-8:
        avg_u8 = np.clip((avg_flow_mag / max_val) * 255.0, 0, 255).astype(np.uint8)
        otsu_u8_thresh, otsu_binary = cv2.threshold(
            avg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        otsu_thresh_px = (float(otsu_u8_thresh) / 255.0) * max_val
    else:
        otsu_u8_thresh = 0.0
        otsu_thresh_px = 0.0
        otsu_binary = np.zeros((HIGHRES_H, HIGHRES_W), dtype=np.uint8)

    ys, xs = np.where(otsu_binary > 0)
    if len(xs) > 0:
        bbox_hr = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        bbox_hr = None

    bbox_lr = None
    if bbox_hr is not None:
        x0, y0, x1, y1 = bbox_hr
        bbox_lr = (
            max(0, x0 // DOWNSAMPLE_FACTOR),
            max(0, y0 // DOWNSAMPLE_FACTOR),
            min(LOWRES_W, (x1 + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR),
            min(LOWRES_H, (y1 + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR),
        )

    return {
        "camera": camera,
        "avg_flow_mag": avg_flow_mag,
        "otsu_thresh_u8": float(otsu_u8_thresh),
        "otsu_thresh_px": otsu_thresh_px,
        "otsu_mask": otsu_binary,
        "bbox_highres": bbox_hr,
        "bbox_lowres": bbox_lr,
        "ref_frame_rgb": frames[20],
    }


def save_diagnostic_figure(
    res: dict[str, Any], out_path: Path,
) -> None:
    camera = res["camera"]
    avg_flow = res["avg_flow_mag"]
    otsu_mask = res["otsu_mask"]
    bbox_hr = res["bbox_highres"]
    tau_px = res["otsu_thresh_px"]
    tau_u8 = res["otsu_thresh_u8"]
    rgb_ref = res["ref_frame_rgb"]
    H, W = avg_flow.shape

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax0 = axes[0, 0]
    im0 = ax0.imshow(avg_flow, cmap="magma", vmin=0, vmax=max(float(avg_flow.max()), 1e-4))
    plt.colorbar(im0, ax=ax0, fraction=0.036, pad=0.03, label="Mean Flow Magnitude (px)")
    ax0.set_title(
        f"1. Temporal Mean Optical Flow ({FRAME_COUNT} frames, {W}x{H})\n"
        f"Max: {avg_flow.max():.3f} px | Mean: {avg_flow.mean():.4f} px",
        fontsize=11, fontweight="bold",
    )
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = axes[0, 1]
    nonzero = avg_flow[avg_flow > 1e-4]
    if len(nonzero) > 0:
        ax1.hist(nonzero, bins=80, color="#3b82f6", edgecolor="black", alpha=0.8, log=True)
    ax1.axvline(tau_px, color="#ef4444", linestyle="--", linewidth=2.5,
                label=f"Otsu threshold = {tau_px:.3f} px\n({tau_u8:.0f}/255 in 8-bit)")
    ax1.set_title("2. Average Flow Histogram (Log) + Otsu Threshold", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Temporal Average Flow Magnitude (px)", fontsize=10)
    ax1.set_ylabel("Pixel Count (log scale)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = axes[1, 0]
    ax2.imshow(otsu_mask, cmap="gray")
    active_px = int((otsu_mask > 0).sum())
    if bbox_hr is not None:
        x0, y0, x1, y1 = bbox_hr
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=2.5, edgecolor="#ef4444", facecolor="none")
        ax2.add_patch(rect)
    ax2.set_title(
        f"3. Otsu Binary Mask (avg flow > {tau_px:.3f} px)\n"
        f"Active: {active_px} px ({100.0 * active_px / (H * W):.2f}% of frame)",
        fontsize=11, fontweight="bold",
    )
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = axes[1, 1]
    overlay = rgb_ref.copy()
    if bbox_hr is not None:
        x0, y0, x1, y1 = bbox_hr
        darkened = (overlay.astype(np.float32) * 0.42).astype(np.uint8)
        darkened[y0:y1, x0:x1] = overlay[y0:y1, x0:x1]
        overlay = darkened
        rect2 = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   linewidth=2.8, edgecolor="#00ff88", facecolor="none")
        ax3.imshow(overlay)
        ax3.add_patch(rect2)
        bw, bh = x1 - x0, y1 - y0
        box_pct = 100.0 * bw * bh / (H * W)
        ax3.set_title(
            f"4. Otsu BBox on {camera} (t=20, {W}x{H})\n"
            f"Size: {bw}x{bh} px ({box_pct:.1f}% area) | [{x0}:{x1}, {y0}:{y1}]",
            fontsize=11, fontweight="bold",
        )
    else:
        ax3.imshow(overlay)
        ax3.set_title(f"4. No BBox detected for {camera}", fontsize=11, fontweight="bold")
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.suptitle(
        f"v5: High-Res Otsu Average-Flow BBox — {camera} ({W}x{H}, {FRAME_COUNT} frames)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved diagnostic: {out_path}")


def compute_metrics_in_bbox(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    lpips_model,
) -> dict[str, float]:
    """Compute PSNR/SSIM/LPIPS for both full image and bbox crop."""
    pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).float().cuda() / 255.0
    gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).float().cuda() / 255.0

    mse_full = F.mse_loss(pred_t, gt_t).item()
    psnr_full = 10.0 * np.log10(1.0 / max(mse_full, 1e-10))

    pred_batch = pred_t.unsqueeze(0)
    gt_batch = gt_t.unsqueeze(0)

    from pytorch_msssim import ssim as ssim_fn
    ssim_full = ssim_fn(pred_batch, gt_batch, data_range=1.0, size_average=True).item()

    lpips_full = lpips_model(
        pred_batch * 2 - 1, gt_batch * 2 - 1,
    ).mean().item()

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        pred_crop = pred_t[:, y0:y1, x0:x1].unsqueeze(0)
        gt_crop = gt_t[:, y0:y1, x0:x1].unsqueeze(0)

        mse_box = F.mse_loss(pred_crop, gt_crop).item()
        psnr_box = 10.0 * np.log10(1.0 / max(mse_box, 1e-10))

        h_crop, w_crop = y1 - y0, x1 - x0
        if h_crop >= 16 and w_crop >= 16:
            ssim_box = ssim_fn(pred_crop, gt_crop, data_range=1.0, size_average=True).item()
            lpips_box = lpips_model(
                pred_crop * 2 - 1, gt_crop * 2 - 1,
            ).mean().item()
        else:
            ssim_box = ssim_full
            lpips_box = lpips_full
    else:
        psnr_box = psnr_full
        ssim_box = ssim_full
        lpips_box = lpips_full

    return {
        "PSNR_full": psnr_full,
        "SSIM_full": ssim_full,
        "LPIPS_full": lpips_full,
        "PSNR_bbox": psnr_box,
        "SSIM_bbox": ssim_box,
        "LPIPS_bbox": lpips_box,
    }


def evaluate_predictions(
    args: argparse.Namespace,
    bbox_data: dict[str, dict],
) -> None:
    """Evaluate 4DGS low-res predictions against low-res GT using the v5 Otsu
    bounding boxes (downscaled to low-res coordinates)."""
    import lpips as lpips_lib

    lpips_model = lpips_lib.LPIPS(net="vgg").cuda().eval()

    out_dir = args.output_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for gap in GAPS:
        eval_dir = (
            args.project_root / "output" / f"coffee_martini_k{gap}"
            / "midpoint_evaluation_v4" / f"gap_{gap}"
        )
        if not eval_dir.exists():
            print(f"  [SKIP] {eval_dir} not found")
            continue

        midpoints = list(range(gap // 2, 49, gap))

        for camera in CAMERAS:
            setting = "heldout_camera" if camera == "cam00" else "training_cameras"
            prefix = "heldout_cam00" if camera == "cam00" else camera

            cam_bbox = bbox_data.get(camera, {})
            bbox_lr = cam_bbox.get("bbox_lowres")
            bbox_hr = cam_bbox.get("bbox_highres")

            for mid in midpoints:
                pred_path = eval_dir / f"{prefix}_{mid:04d}_pred.png"
                gt_path = eval_dir / f"{prefix}_{mid:04d}_gt.png"

                if not pred_path.exists() or not gt_path.exists():
                    continue

                pred_rgb = cv2.cvtColor(
                    cv2.imread(str(pred_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
                gt_rgb = cv2.cvtColor(
                    cv2.imread(str(gt_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )

                metrics = compute_metrics_in_bbox(pred_rgb, gt_rgb, bbox_lr, lpips_model)

                bw = bh = area_pct = 0
                if bbox_lr is not None:
                    x0, y0, x1, y1 = bbox_lr
                    bw, bh = x1 - x0, y1 - y0
                    area_pct = 100.0 * bw * bh / (LOWRES_W * LOWRES_H)

                row = {
                    "method": "4DGS",
                    "temporal_gap": gap,
                    "evaluation_setting": setting,
                    "camera": camera,
                    "frame_index": mid,
                    "bbox_method": "v5_otsu_highres",
                    "bbox_lr": str(bbox_lr),
                    "bbox_hr": str(bbox_hr),
                    "bbox_width_lr": bw,
                    "bbox_height_lr": bh,
                    "bbox_area_pct_lr": round(area_pct, 2),
                    **metrics,
                }
                all_rows.append(row)

        print(f"  k={gap}: {sum(1 for r in all_rows if r['temporal_gap'] == gap)} samples evaluated")

    csv_path = out_dir / "v5_otsu_metrics.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Saved metrics: {csv_path}")

    summary_rows = []
    for gap in GAPS:
        for setting in ("training_cameras", "heldout_camera"):
            subset = [r for r in all_rows if r["temporal_gap"] == gap and r["evaluation_setting"] == setting]
            if not subset:
                continue
            summary_rows.append({
                "method": "4DGS",
                "temporal_gap": gap,
                "evaluation_setting": setting,
                "metric_scope": "bbox",
                "mean_PSNR": np.mean([r["PSNR_bbox"] for r in subset]),
                "mean_SSIM": np.mean([r["SSIM_bbox"] for r in subset]),
                "mean_LPIPS": np.mean([r["LPIPS_bbox"] for r in subset]),
                "num_samples": len(subset),
            })
            summary_rows.append({
                "method": "4DGS",
                "temporal_gap": gap,
                "evaluation_setting": setting,
                "metric_scope": "full",
                "mean_PSNR": np.mean([r["PSNR_full"] for r in subset]),
                "mean_SSIM": np.mean([r["SSIM_full"] for r in subset]),
                "mean_LPIPS": np.mean([r["LPIPS_full"] for r in subset]),
                "num_samples": len(subset),
            })

    summary_path = out_dir / "v5_otsu_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Saved summary: {summary_path}")

    print("\n=== v5 Otsu Summary ===")
    for r in summary_rows:
        print(
            f"  k={r['temporal_gap']} {r['evaluation_setting']:20s} {r['metric_scope']:5s} "
            f"PSNR={r['mean_PSNR']:.2f}  SSIM={r['mean_SSIM']:.4f}  LPIPS={r['mean_LPIPS']:.4f}  "
            f"(n={r['num_samples']})"
        )


def main():
    args = parse_args()
    cameras_to_process = args.cameras if args.cameras else list(CAMERAS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bbox_data = {}

    if args.stage in ("diagnostic", "all"):
        print("=== Stage 1: Computing Otsu average-flow bounding boxes (high-res) ===")
        diag_dir = args.output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        bbox_json = {}
        for camera in tqdm(cameras_to_process, desc="Otsu bbox (high-res)"):
            print(f"\n  Processing {camera} at {HIGHRES_W}x{HIGHRES_H}...")
            res = compute_otsu_bbox_for_camera(args.highres_root, camera)
            bbox_data[camera] = res

            save_diagnostic_figure(res, diag_dir / f"otsu_bbox_{camera}.png")

            bbox_json[camera] = {
                "otsu_thresh_px": res["otsu_thresh_px"],
                "otsu_thresh_u8": res["otsu_thresh_u8"],
                "bbox_highres": res["bbox_highres"],
                "bbox_lowres": res["bbox_lowres"],
            }
            print(
                f"  {camera}: Otsu tau={res['otsu_thresh_px']:.3f} px | "
                f"HR bbox={res['bbox_highres']} | LR bbox={res['bbox_lowres']}"
            )

        json_path = args.output_dir / "otsu_bboxes.json"
        with open(json_path, "w") as f:
            json.dump(bbox_json, f, indent=2)
        print(f"\nSaved bbox data: {json_path}")

    if args.stage in ("evaluate", "all"):
        if not bbox_data:
            json_path = args.output_dir / "otsu_bboxes.json"
            if json_path.exists():
                with open(json_path) as f:
                    bbox_json = json.load(f)
                for cam, info in bbox_json.items():
                    bbox_data[cam] = {
                        "bbox_lowres": tuple(info["bbox_lowres"]) if info["bbox_lowres"] else None,
                        "bbox_highres": tuple(info["bbox_highres"]) if info["bbox_highres"] else None,
                    }
            else:
                print("No bbox data found. Run diagnostic stage first.")
                return

        print("\n=== Stage 2: Evaluating predictions with v5 Otsu bboxes ===")
        evaluate_predictions(args, bbox_data)


if __name__ == "__main__":
    main()
