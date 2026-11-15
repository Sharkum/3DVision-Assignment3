#!/usr/bin/env python3
"""All-50-frame evaluation with full-image + dynamic region (Otsu bbox) metrics
for k=8. Saves CSVs and generates 2-panel PSNR plot.

Usage:
  python dynamic_region_all50_eval.py \
    --configs arguments/dynerf/coffee_martini.py \
    --model_path output/coffee_martini_k8 \
    --source_path ~/Desktop/rohan/asmt2/coffee_martini \
    --iteration 14000
"""

import os
import csv
import copy
import json
import numpy as np
import torch
import lpips

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIGHRES_W, HIGHRES_H = 2704, 2028
PROJECT_ROOT = Path(__file__).resolve().parent
HIGHRES_ROOT = PROJECT_ROOT.parent / "coffee_martini_highres"
OUTPUT_DIR = PROJECT_ROOT / "output" / "dynamic_region_evaluation"


def load_gt_highres(camera_name, frame_idx):
    path = HIGHRES_ROOT / camera_name / "images" / f"{frame_idx:04d}.png"
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def make_highres_camera(camera, timestamp):
    cam = copy.copy(camera)
    cam.time = timestamp
    cam.image_width = HIGHRES_W
    cam.image_height = HIGHRES_H
    return cam


def render_highres(camera, timestamp, gaussians, pipeline, background, cam_type):
    hr_cam = make_highres_camera(camera, timestamp)
    with torch.no_grad():
        prediction = render(hr_cam, gaussians, pipeline, background, cam_type=cam_type)["render"].clamp(0, 1)
    return prediction


def compute_psnr(pred, gt):
    mse = ((pred - gt) ** 2).mean()
    return (-10 * torch.log10(mse)).item()


def load_otsu_bboxes():
    json_path = PROJECT_ROOT / "output" / "dynamic_bbox_evaluation_v5_otsu" / "otsu_bboxes.json"
    with open(json_path) as f:
        data = json.load(f)
    bboxes = {}
    for cam, info in data.items():
        hr = info.get("bbox_highres")
        bboxes[cam] = tuple(hr) if hr else None
    return bboxes


def get_camera_name(train_dataset, base_idx):
    first_path = train_dataset.image_paths[base_idx]
    return os.path.basename(os.path.dirname(os.path.dirname(first_path)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    hidden_params = ModelHiddenParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default="")

    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    args.temporal_gap = 8

    from utils.general_utils import safe_state
    safe_state(args.quiet)

    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    hyper = hidden_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    otsu_bboxes = load_otsu_bboxes()
    print(f"Loaded Otsu bboxes for {len(otsu_bboxes)} cameras")

    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    train_dataset = scene.train_camera.dataset
    frames_per_cam = len(range(0, 50, 8))
    num_train_cams = len(train_dataset.image_paths) // frames_per_cam

    train_frames_set = set(range(0, 49, 8))
    test_frames_set = set(range(4, 49, 8))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════
    # cam00: all 50 frames, full + bbox PSNR
    # ════════════════════════════════════════════
    cam00_bbox = otsu_bboxes.get("cam00")
    print(f"\ncam00 Otsu bbox: {cam00_bbox}")
    print("=== cam00: 50 frames ===")

    cam00_rows = []
    for frame_idx in tqdm(range(50), desc="cam00"):
        timestamp = frame_idx / 49.0
        camera = test_cameras[frame_idx]
        gt = load_gt_highres("cam00", frame_idx)
        pred = render_highres(camera, timestamp, gaussians, pipeline, background, scene.dataset_type)

        full_psnr = compute_psnr(pred, gt)
        bbox_psnr = None
        if cam00_bbox:
            x0, y0, x1, y1 = cam00_bbox
            bbox_psnr = compute_psnr(pred[:, y0:y1, x0:x1], gt[:, y0:y1, x0:x1])

        cam00_rows.append({
            "frame": frame_idx,
            "full_psnr": full_psnr,
            "bbox_psnr": bbox_psnr,
            "is_train": frame_idx in train_frames_set,
            "is_midpoint": frame_idx in test_frames_set,
        })
        del pred, gt
        torch.cuda.empty_cache()

    cam00_csv = OUTPUT_DIR / "cam00_all50_full_and_bbox.csv"
    with open(cam00_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "full_psnr", "bbox_psnr", "is_train", "is_midpoint"])
        for r in cam00_rows:
            w.writerow([r["frame"], f"{r['full_psnr']:.4f}",
                        f"{r['bbox_psnr']:.4f}" if r["bbox_psnr"] else "",
                        r["is_train"], r["is_midpoint"]])
    print(f"Saved: {cam00_csv}")

    # ════════════════════════════════════════════
    # Training cameras: all 50 frames, full + bbox
    # ════════════════════════════════════════════
    print(f"\n=== {num_train_cams} training cameras x 50 frames ===")

    # Per-frame: collect per-camera values then average
    train_full_per_frame = {f: [] for f in range(50)}
    train_bbox_per_frame = {f: [] for f in range(50)}

    for cam_idx in tqdm(range(num_train_cams), desc="train cams"):
        base_idx = cam_idx * frames_per_cam
        pose_camera = train_cameras[base_idx]
        camera_name = get_camera_name(train_dataset, base_idx)
        bbox = otsu_bboxes.get(camera_name)

        for frame_idx in range(50):
            timestamp = frame_idx / 49.0
            gt = load_gt_highres(camera_name, frame_idx)
            pred = render_highres(pose_camera, timestamp, gaussians, pipeline, background, scene.dataset_type)

            full_psnr = compute_psnr(pred, gt)
            train_full_per_frame[frame_idx].append(full_psnr)

            if bbox:
                x0, y0, x1, y1 = bbox
                bp = compute_psnr(pred[:, y0:y1, x0:x1], gt[:, y0:y1, x0:x1])
                train_bbox_per_frame[frame_idx].append(bp)

            del pred, gt
            torch.cuda.empty_cache()

    train_rows = []
    for f in range(50):
        full_mean = np.mean(train_full_per_frame[f])
        bbox_mean = np.mean(train_bbox_per_frame[f]) if train_bbox_per_frame[f] else None
        train_rows.append({
            "frame": f,
            "full_psnr": full_mean,
            "bbox_psnr": bbox_mean,
            "is_train": f in train_frames_set,
            "is_midpoint": f in test_frames_set,
            "num_cameras": len(train_full_per_frame[f]),
        })

    train_csv = OUTPUT_DIR / "train_cameras_all50_full_and_bbox.csv"
    with open(train_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "full_psnr", "bbox_psnr", "is_train", "is_midpoint", "num_cameras"])
        for r in train_rows:
            w.writerow([r["frame"], f"{r['full_psnr']:.4f}",
                        f"{r['bbox_psnr']:.4f}" if r["bbox_psnr"] else "",
                        r["is_train"], r["is_midpoint"], r["num_cameras"]])
    print(f"Saved: {train_csv}")

    # ════════════════════════════════════════════
    # Generate 2-panel plot
    # ════════════════════════════════════════════
    frames = list(range(50))
    cam00_full = [r["full_psnr"] for r in cam00_rows]
    cam00_bbox_vals = [r["bbox_psnr"] for r in cam00_rows]
    train_full = [r["full_psnr"] for r in train_rows]
    train_bbox_vals = [r["bbox_psnr"] for r in train_rows]

    train_ts = [0, 8, 16, 24, 32, 40, 48]
    mid_ts = [4, 12, 20, 28, 36, 44]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"hspace": 0.35})

    # ── Plot 1: Sawtooth (training cams + cam00, full frame) ──
    ax = axes[0]
    for tf in train_ts:
        ax.axvline(tf, color="#2ca02c", linestyle="--", alpha=0.2, linewidth=0.8)
    for mf in mid_ts:
        ax.axvline(mf, color="#d62728", linestyle=":", alpha=0.2, linewidth=0.8)

    ax.plot(frames, train_full, color="#1f77b4", linewidth=2.0,
            label="Training Cameras (cam01–cam17 Mean, Full Frame)")
    ax.plot(frames, cam00_full, color="#ff7f0e", linewidth=2.0,
            label="Held-out Camera (cam00, Full Frame)")

    ax.scatter(train_ts, [train_full[t] for t in train_ts],
               color="#2ca02c", s=70, zorder=5, edgecolors="black", linewidth=0.7,
               label="Training Timestamps {0, 8, 16, 24, 32, 40, 48}")
    ax.scatter(mid_ts, [train_full[t] for t in mid_ts],
               color="#d62728", marker="D", s=65, zorder=5, edgecolors="black", linewidth=0.7,
               label="Midpoint Test Timestamps {4, 12, 20, 28, 36, 44}")
    ax.scatter(train_ts, [cam00_full[t] for t in train_ts],
               color="#2ca02c", s=60, zorder=5, edgecolors="black", linewidth=0.7)
    ax.scatter(mid_ts, [cam00_full[t] for t in mid_ts],
               color="#d62728", marker="D", s=60, zorder=5, edgecolors="black", linewidth=0.7)

    ax.set_xlabel("Frame Index t (0 to 49)", fontsize=11, fontweight="bold")
    ax.set_ylabel("PSNR (dB) ↑", fontsize=11, fontweight="bold")
    ax.set_title("4DGS Per-Frame PSNR Across Training & Held-Out Test Frames (k = 8, coffee_martini)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 50, 2))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.92)

    # ── Plot 2: cam00 full vs dynamic region ──
    ax2 = axes[1]
    for tf in train_ts:
        ax2.axvline(tf, color="#2ca02c", linestyle="--", alpha=0.2, linewidth=0.8)
    for mf in mid_ts:
        ax2.axvline(mf, color="#d62728", linestyle=":", alpha=0.2, linewidth=0.8)

    ax2.plot(frames, cam00_full, color="#ff7f0e", linewidth=2.0,
             label="Held-out Camera (cam00) — Full Frame PSNR")
    ax2.plot(frames, cam00_bbox_vals, color="#9467bd", linewidth=2.0,
             label="Held-out Camera (cam00) — Dynamic Region Masked PSNR")

    ax2.scatter(train_ts, [cam00_full[t] for t in train_ts],
                color="#2ca02c", s=70, zorder=5, edgecolors="black", linewidth=0.7,
                label="Observed Training Timestamps")
    ax2.scatter(mid_ts, [cam00_full[t] for t in mid_ts],
                color="#d62728", marker="D", s=65, zorder=5, edgecolors="black", linewidth=0.7,
                label="Held-out Midpoint Timestamps")
    ax2.scatter(train_ts, [cam00_bbox_vals[t] for t in train_ts],
                color="#2ca02c", s=60, zorder=5, edgecolors="black", linewidth=0.7)
    ax2.scatter(mid_ts, [cam00_bbox_vals[t] for t in mid_ts],
                color="#d62728", marker="D", s=60, zorder=5, edgecolors="black", linewidth=0.7)

    ax2.set_xlabel("Frame Index t (0 to 49)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("PSNR (dB) ↑", fontsize=11, fontweight="bold")
    ax2.set_title("Effect of Temporal Interpolation on Dynamic Regions vs. Full Frame (k = 8, Held-Out cam00)",
                  fontsize=13, fontweight="bold")
    ax2.set_xticks(range(0, 50, 2))
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.92)

    plot_path = OUTPUT_DIR / "psnr_per_frame_k8_highres.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("  SUMMARY (k=8, all 50 frames, 2704x2028)")
    print(f"{'='*60}")
    for name, rows in [("cam00", cam00_rows), ("train_cameras", train_rows)]:
        full_all = np.mean([r["full_psnr"] for r in rows])
        full_train = np.mean([r["full_psnr"] for r in rows if r["is_train"]])
        full_mid = np.mean([r["full_psnr"] for r in rows if r["is_midpoint"]])
        bbox_all = np.mean([r["bbox_psnr"] for r in rows if r["bbox_psnr"] is not None])
        bbox_train = np.mean([r["bbox_psnr"] for r in rows if r["is_train"] and r["bbox_psnr"] is not None])
        bbox_mid = np.mean([r["bbox_psnr"] for r in rows if r["is_midpoint"] and r["bbox_psnr"] is not None])
        print(f"\n  {name}:")
        print(f"    Full-image:  all={full_all:.2f}  train={full_train:.2f}  midpoint={full_mid:.2f}  sawtooth={full_train-full_mid:.2f} dB")
        print(f"    Dynamic bbox: all={bbox_all:.2f}  train={bbox_train:.2f}  midpoint={bbox_mid:.2f}  sawtooth={bbox_train-bbox_mid:.2f} dB")


if __name__ == "__main__":
    main()
