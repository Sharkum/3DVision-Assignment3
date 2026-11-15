#!/usr/bin/env python3
"""Low-resolution (676x507) evaluation with AlexNet LPIPS + Otsu bbox metrics.

Runs midpoint evaluation for any k={2,4,8}. For k=8, also evaluates all 50
frames (sawtooth data). Uses the team-standard AlexNet LPIPS and Otsu
average-flow bounding boxes at low-res coordinates.

Usage:
  python low_resolution_work/evaluate_lowres.py \
    --configs arguments/dynerf/coffee_martini.py \
    --model_path output/coffee_martini_k2 \
    --source_path ~/Desktop/rohan/asmt2/coffee_martini \
    --gap 2 --iteration 14000

  # Run all three:
  for k in 2 4 8; do
    python low_resolution_work/evaluate_lowres.py \
      --configs arguments/dynerf/coffee_martini.py \
      --model_path output/coffee_martini_k${k} \
      --source_path ~/Desktop/rohan/asmt2/coffee_martini \
      --gap $k --iteration 14000
  done
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
import lpips

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from tqdm import tqdm

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import (
    ModelParams,
    PipelineParams,
    ModelHiddenParams,
    get_combined_args,
)
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim

FRAME_COUNT = 50
LOWRES_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "coffee_martini")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "low_resolution_work")


def load_gt_lowres(camera_name, frame_idx):
    path = os.path.join(LOWRES_ROOT, camera_name, "images", f"{frame_idx:04d}.png")
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def compute_metrics(pred, gt, lpips_model):
    pred_batch = pred.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    p = psnr(pred_batch, gt_batch).mean().item()
    s = ssim(pred_batch, gt_batch).mean().item()
    lp = lpips_model(pred_batch * 2 - 1, gt_batch * 2 - 1).mean().item()
    return p, s, lp


def compute_bbox_metrics(pred, gt, bbox, lpips_model):
    if bbox is None:
        return None, None, None
    x0, y0, x1, y1 = bbox
    pred_crop = pred[:, y0:y1, x0:x1].unsqueeze(0)
    gt_crop = gt[:, y0:y1, x0:x1].unsqueeze(0)
    h, w = y1 - y0, x1 - x0
    p = psnr(pred_crop, gt_crop).mean().item()
    if h >= 16 and w >= 16:
        s = ssim(pred_crop, gt_crop).mean().item()
        lp = lpips_model(pred_crop * 2 - 1, gt_crop * 2 - 1).mean().item()
    else:
        s = 0.0
        lp = 0.0
    return p, s, lp


def get_camera_name(train_dataset, base_idx):
    first_path = train_dataset.image_paths[base_idx]
    return os.path.basename(os.path.dirname(os.path.dirname(first_path)))


def load_otsu_bboxes_lowres():
    json_path = os.path.join(
        PROJECT_ROOT, "output", "dynamic_bbox_evaluation_v5_otsu", "otsu_bboxes.json"
    )
    with open(json_path) as f:
        data = json.load(f)
    bboxes = {}
    for cam, info in data.items():
        lr = info.get("bbox_lowres")
        bboxes[cam] = tuple(lr) if lr else None
    return bboxes


def render_lowres(camera, timestamp, gaussians, pipeline, background, cam_type):
    camera.time = timestamp
    with torch.no_grad():
        prediction = render(
            camera, gaussians, pipeline, background, cam_type=cam_type
        )["render"].clamp(0, 1)
    return prediction


def main():
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    hidden_params = ModelHiddenParams(parser)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default="")

    args = get_combined_args(parser)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    gap = args.gap
    args.temporal_gap = gap
    if gap not in (2, 4, 8):
        raise ValueError("gap must be 2, 4, or 8")

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

    lpips_alex = lpips.LPIPS(net="alex", version="0.1").cuda().eval()

    otsu_bboxes = load_otsu_bboxes_lowres()
    print(f"Loaded Otsu bboxes (lowres) for {len(otsu_bboxes)} cameras")

    output_dir = os.path.join(OUTPUT_BASE, f"k{gap}")
    os.makedirs(output_dir, exist_ok=True)

    midpoints = list(range(gap // 2, 49, gap))
    fullimg_rows = []
    bbox_rows = []

    # ════════════════════════════════════════════════════════════
    # Setting 1: Training cameras at held-out midpoint times
    # ════════════════════════════════════════════════════════════
    train_dataset = scene.train_camera.dataset
    train_cameras = scene.getTrainCameras()
    frames_per_cam = len(range(0, 50, gap))
    num_train_cams = len(train_dataset.image_paths) // frames_per_cam

    print(f"\n[k={gap}] Setting 1: {num_train_cams} train cams x {len(midpoints)} midpoints = {num_train_cams * len(midpoints)} renders at 676x507")

    for cam_idx in tqdm(range(num_train_cams), desc=f"k={gap} Setting 1"):
        base_idx = cam_idx * frames_per_cam
        pose_camera = train_cameras[base_idx]
        camera_name = get_camera_name(train_dataset, base_idx)

        for frame_idx in midpoints:
            timestamp = frame_idx / 49.0
            gt = load_gt_lowres(camera_name, frame_idx)
            pred = render_lowres(
                pose_camera, timestamp, gaussians, pipeline, background,
                scene.dataset_type,
            )

            p, s, lp = compute_metrics(pred, gt, lpips_alex)
            fullimg_rows.append(["training_cameras", camera_name, frame_idx, p, s, lp])

            bbox = otsu_bboxes.get(camera_name)
            bp, bs, blp = compute_bbox_metrics(pred, gt, bbox, lpips_alex)
            if bp is not None:
                bbox_rows.append([
                    "training_cameras", camera_name, frame_idx,
                    bp, bs, blp, p, s, lp,
                ])

            del pred, gt
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════
    # Setting 2: Held-out cam00 at held-out midpoint times
    # ════════════════════════════════════════════════════════════
    test_cameras = scene.getTestCameras()
    print(f"\n[k={gap}] Setting 2: cam00 x {len(midpoints)} midpoints at 676x507")

    for frame_idx in tqdm(midpoints, desc=f"k={gap} Setting 2"):
        timestamp = frame_idx / 49.0
        camera = test_cameras[frame_idx]
        gt = load_gt_lowres("cam00", frame_idx)
        pred = render_lowres(
            camera, timestamp, gaussians, pipeline, background,
            scene.dataset_type,
        )

        p, s, lp = compute_metrics(pred, gt, lpips_alex)
        fullimg_rows.append(["heldout_camera", "cam00", frame_idx, p, s, lp])

        bbox = otsu_bboxes.get("cam00")
        bp, bs, blp = compute_bbox_metrics(pred, gt, bbox, lpips_alex)
        if bp is not None:
            bbox_rows.append([
                "heldout_camera", "cam00", frame_idx,
                bp, bs, blp, p, s, lp,
            ])

        del pred, gt
        torch.cuda.empty_cache()

    # ── Save midpoint CSVs ──
    csv_path = os.path.join(output_dir, "midpoint_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "camera", "frame", "PSNR", "SSIM", "LPIPS_alex"])
        w.writerows(fullimg_rows)
    print(f"Saved: {csv_path}")

    bbox_csv = os.path.join(output_dir, "midpoint_bbox_metrics.csv")
    with open(bbox_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "setting", "camera", "frame",
            "PSNR_bbox", "SSIM_bbox", "LPIPS_alex_bbox",
            "PSNR_full", "SSIM_full", "LPIPS_alex_full",
        ])
        w.writerows(bbox_rows)
    print(f"Saved: {bbox_csv}")

    # ── Print midpoint summary ──
    print(f"\n{'='*60}")
    print(f"  LOW-RES MIDPOINT RESULTS (k={gap}, 676x507, LPIPS=AlexNet)")
    print(f"{'='*60}")

    for setting in ("training_cameras", "heldout_camera"):
        rows = [r for r in fullimg_rows if r[0] == setting]
        if not rows:
            continue
        vals = np.array([r[3:6] for r in rows])
        avg = vals.mean(axis=0)
        print(f"  {setting} (full):  PSNR={avg[0]:.2f}  SSIM={avg[1]:.4f}  LPIPS={avg[2]:.4f}  (n={len(rows)})")

        brows = [r for r in bbox_rows if r[0] == setting]
        if brows:
            bvals = np.array([r[3:6] for r in brows])
            bavg = bvals.mean(axis=0)
            print(f"  {setting} (bbox):  PSNR={bavg[0]:.2f}  SSIM={bavg[1]:.4f}  LPIPS={bavg[2]:.4f}  (n={len(brows)})")

    # ════════════════════════════════════════════════════════════
    # k=8 only: all 50 frames for sawtooth plot
    # ════════════════════════════════════════════════════════════
    if gap == 8:
        print(f"\n{'='*60}")
        print(f"  k=8: All 50 frames at 676x507 (sawtooth data)")
        print(f"{'='*60}")

        train_frames_set = set(range(0, 49, gap))
        test_frames_set = set(range(gap // 2, 49, gap))

        # ── cam00: all 50 frames ──
        cam00_rows = []
        cam00_bbox = otsu_bboxes.get("cam00")
        print(f"\n  cam00 bbox (lowres): {cam00_bbox}")

        for frame_idx in tqdm(range(FRAME_COUNT), desc="k=8 cam00 all50"):
            timestamp = frame_idx / 49.0
            camera = test_cameras[frame_idx]
            gt = load_gt_lowres("cam00", frame_idx)
            pred = render_lowres(
                camera, timestamp, gaussians, pipeline, background,
                scene.dataset_type,
            )

            fp, fs, flp = compute_metrics(pred, gt, lpips_alex)
            bp, bs, blp = compute_bbox_metrics(pred, gt, cam00_bbox, lpips_alex)

            cam00_rows.append({
                "frame": frame_idx,
                "full_psnr": fp, "full_ssim": fs, "full_lpips": flp,
                "bbox_psnr": bp, "bbox_ssim": bs, "bbox_lpips": blp,
                "is_train": frame_idx in train_frames_set,
                "is_midpoint": frame_idx in test_frames_set,
            })

            del pred, gt
            torch.cuda.empty_cache()

        cam00_csv = os.path.join(output_dir, "all50_cam00.csv")
        with open(cam00_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame",
                "full_psnr", "full_ssim", "full_lpips_alex",
                "bbox_psnr", "bbox_ssim", "bbox_lpips_alex",
                "is_train", "is_midpoint",
            ])
            for r in cam00_rows:
                w.writerow([
                    r["frame"],
                    f"{r['full_psnr']:.4f}", f"{r['full_ssim']:.4f}", f"{r['full_lpips']:.4f}",
                    f"{r['bbox_psnr']:.4f}" if r["bbox_psnr"] else "",
                    f"{r['bbox_ssim']:.4f}" if r["bbox_ssim"] else "",
                    f"{r['bbox_lpips']:.4f}" if r["bbox_lpips"] else "",
                    r["is_train"], r["is_midpoint"],
                ])
        print(f"Saved: {cam00_csv}")

        # ── Training cameras: all 50 frames ──
        train_full_per_frame = {f: [] for f in range(FRAME_COUNT)}
        train_bbox_per_frame = {f: [] for f in range(FRAME_COUNT)}

        for cam_idx in tqdm(range(num_train_cams), desc="k=8 train all50"):
            base_idx = cam_idx * frames_per_cam
            pose_camera = train_cameras[base_idx]
            camera_name = get_camera_name(train_dataset, base_idx)
            bbox = otsu_bboxes.get(camera_name)

            for frame_idx in range(FRAME_COUNT):
                timestamp = frame_idx / 49.0
                gt = load_gt_lowres(camera_name, frame_idx)
                pred = render_lowres(
                    pose_camera, timestamp, gaussians, pipeline, background,
                    scene.dataset_type,
                )

                fp, fs, flp = compute_metrics(pred, gt, lpips_alex)
                train_full_per_frame[frame_idx].append((fp, fs, flp))

                bp, bs_, blp = compute_bbox_metrics(pred, gt, bbox, lpips_alex)
                if bp is not None:
                    train_bbox_per_frame[frame_idx].append((bp, bs_, blp))

                del pred, gt
                torch.cuda.empty_cache()

        train_all50_rows = []
        for f in range(FRAME_COUNT):
            full_vals = train_full_per_frame[f]
            full_mean = np.mean(full_vals, axis=0)
            bbox_vals = train_bbox_per_frame[f]
            bbox_mean = np.mean(bbox_vals, axis=0) if bbox_vals else (None, None, None)
            train_all50_rows.append({
                "frame": f,
                "full_psnr": full_mean[0], "full_ssim": full_mean[1], "full_lpips": full_mean[2],
                "bbox_psnr": bbox_mean[0], "bbox_ssim": bbox_mean[1], "bbox_lpips": bbox_mean[2],
                "is_train": f in train_frames_set,
                "is_midpoint": f in test_frames_set,
                "num_cameras": len(full_vals),
            })

        train_csv = os.path.join(output_dir, "all50_train_cameras.csv")
        with open(train_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame",
                "full_psnr", "full_ssim", "full_lpips_alex",
                "bbox_psnr", "bbox_ssim", "bbox_lpips_alex",
                "is_train", "is_midpoint", "num_cameras",
            ])
            for r in train_all50_rows:
                w.writerow([
                    r["frame"],
                    f"{r['full_psnr']:.4f}", f"{r['full_ssim']:.4f}", f"{r['full_lpips']:.4f}",
                    f"{r['bbox_psnr']:.4f}" if r["bbox_psnr"] is not None else "",
                    f"{r['bbox_ssim']:.4f}" if r["bbox_ssim"] is not None else "",
                    f"{r['bbox_lpips']:.4f}" if r["bbox_lpips"] is not None else "",
                    r["is_train"], r["is_midpoint"], r["num_cameras"],
                ])
        print(f"Saved: {train_csv}")

        # ── All-50 summary ──
        print(f"\n{'='*60}")
        print(f"  ALL-50 SUMMARY (k=8, 676x507, LPIPS=AlexNet)")
        print(f"{'='*60}")
        for name, rows in [("cam00", cam00_rows), ("train_cameras", train_all50_rows)]:
            full_all = np.mean([r["full_psnr"] for r in rows])
            full_train = np.mean([r["full_psnr"] for r in rows if r["is_train"]])
            full_mid = np.mean([r["full_psnr"] for r in rows if r["is_midpoint"]])
            bbox_all = np.mean([r["bbox_psnr"] for r in rows if r["bbox_psnr"] is not None])
            bbox_train = np.mean([r["bbox_psnr"] for r in rows if r["is_train"] and r["bbox_psnr"] is not None])
            bbox_mid = np.mean([r["bbox_psnr"] for r in rows if r["is_midpoint"] and r["bbox_psnr"] is not None])
            print(f"\n  {name}:")
            print(f"    Full:  all={full_all:.2f}  train={full_train:.2f}  mid={full_mid:.2f}  sawtooth={full_train - full_mid:.2f} dB")
            print(f"    Bbox:  all={bbox_all:.2f}  train={bbox_train:.2f}  mid={bbox_mid:.2f}  sawtooth={bbox_train - bbox_mid:.2f} dB")

    print("\nDone.")


if __name__ == "__main__":
    main()
