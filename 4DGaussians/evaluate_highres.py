#!/usr/bin/env python3
"""High-resolution (2704x2028) evaluation of 4DGS.

Renders the trained model at native high resolution and compares against
high-res ground truth. Computes full-image and dynamic-region (Otsu bbox)
metrics. For k=8, also evaluates all 50 frames for the sawtooth plot.

Usage (same pattern as evaluate_midpoints.py):
  python evaluate_highres.py \
    --configs arguments/dynerf/coffee_martini.py \
    --model_path output/coffee_martini_k2 \
    --source_path ~/Desktop/rohan/asmt2/coffee_martini \
    --gap 2 --iteration 14000
"""

import os
import csv
import copy
import json
import argparse
import numpy as np
import torch
import lpips

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

HIGHRES_W, HIGHRES_H = 2704, 2028
FRAME_COUNT = 50
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HIGHRES_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "coffee_martini_highres")


def load_gt_highres(camera_name, frame_idx):
    path = os.path.join(HIGHRES_ROOT, camera_name, "images", f"{frame_idx:04d}.png")
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def save_image(tensor, path):
    tensor = tensor.detach().clamp(0, 1).cpu()
    if tensor.ndim == 2:
        arr = (tensor.numpy() * 255).astype(np.uint8)
    elif tensor.ndim == 3:
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected shape: {tensor.shape}")
    Image.fromarray(arr).save(path)


def make_highres_camera(camera, timestamp):
    cam = copy.copy(camera)
    cam.time = timestamp
    cam.image_width = HIGHRES_W
    cam.image_height = HIGHRES_H
    return cam


def render_highres(camera, timestamp, gaussians, pipeline, background, cam_type):
    hr_cam = make_highres_camera(camera, timestamp)
    with torch.no_grad():
        prediction = render(
            hr_cam, gaussians, pipeline, background, cam_type=cam_type,
        )["render"].clamp(0, 1)
    return prediction


def compute_metrics(pred, gt, lpips_model):
    pred_batch = pred.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    p = psnr(pred_batch, gt_batch).mean().item()
    s = ssim(pred_batch, gt_batch).mean().item()
    lp = lpips_model(pred_batch * 2 - 1, gt_batch * 2 - 1).mean().item()
    return p, s, lp


def compute_bbox_metrics(pred, gt, bbox, lpips_model):
    if bbox is None:
        return compute_metrics(pred, gt, lpips_model)
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


def get_camera_name_from_train_dataset(train_dataset, base_idx):
    first_path = train_dataset.image_paths[base_idx]
    image_dir = os.path.dirname(first_path)
    return os.path.basename(os.path.dirname(image_dir))


def load_otsu_bboxes():
    json_path = os.path.join(
        PROJECT_ROOT, "output", "dynamic_bbox_evaluation_v5_otsu", "otsu_bboxes.json"
    )
    if not os.path.exists(json_path):
        print(f"[WARNING] Otsu bbox file not found: {json_path}")
        return {}
    with open(json_path) as f:
        data = json.load(f)
    bboxes = {}
    for cam, info in data.items():
        hr = info.get("bbox_highres")
        bboxes[cam] = tuple(hr) if hr else None
    return bboxes


def main():
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    hidden_params = ModelHiddenParams(parser)

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Skip saving pred/gt/error PNGs (faster)")

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

    lpips_model = lpips.LPIPS(net="vgg").cuda().eval()

    otsu_bboxes = load_otsu_bboxes()
    print(f"Loaded Otsu bboxes for {len(otsu_bboxes)} cameras")

    save_images = not getattr(args, "no_save_images", False)

    # ── Output directory ──
    output_root = os.path.join(args.model_path, "highres_evaluation", f"gap_{gap}")
    if os.path.exists(output_root):
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    os.makedirs(output_root)

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

    print(f"\n[k={gap}] Setting 1: {num_train_cams} training cameras x {len(midpoints)} midpoints = {num_train_cams * len(midpoints)} renders at {HIGHRES_W}x{HIGHRES_H}")

    for cam_idx in tqdm(range(num_train_cams), desc=f"k={gap} train cams"):
        base_idx = cam_idx * frames_per_cam
        pose_camera = train_cameras[base_idx]
        camera_name = get_camera_name_from_train_dataset(train_dataset, base_idx)

        for frame_idx in midpoints:
            timestamp = frame_idx / 49.0
            gt_hr = load_gt_highres(camera_name, frame_idx)
            pred_hr = render_highres(
                pose_camera, timestamp, gaussians, pipeline, background,
                scene.dataset_type,
            )

            p, s, lp = compute_metrics(pred_hr, gt_hr, lpips_model)
            sample_name = f"train_{camera_name}_{frame_idx:04d}"
            fullimg_rows.append(["training_cameras", camera_name, frame_idx, p, s, lp])

            bbox = otsu_bboxes.get(camera_name)
            if bbox is not None:
                bp, bs, blp = compute_bbox_metrics(pred_hr, gt_hr, bbox, lpips_model)
                bbox_rows.append(["training_cameras", camera_name, frame_idx,
                                  bp, bs, blp, p, s, lp])

            if save_images:
                save_image(pred_hr, os.path.join(output_root, sample_name + "_pred.png"))
                save_image(gt_hr, os.path.join(output_root, sample_name + "_gt.png"))
                error = (pred_hr - gt_hr).abs().mean(dim=0)
                save_image(error / (error.max() + 1e-8),
                           os.path.join(output_root, sample_name + "_error.png"))

            del pred_hr, gt_hr
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════
    # Setting 2: Held-out camera (cam00) at held-out midpoint times
    # ════════════════════════════════════════════════════════════
    test_cameras = scene.getTestCameras()
    print(f"\n[k={gap}] Setting 2: cam00 x {len(midpoints)} midpoints at {HIGHRES_W}x{HIGHRES_H}")

    for frame_idx in tqdm(midpoints, desc=f"k={gap} cam00"):
        timestamp = frame_idx / 49.0
        camera = test_cameras[frame_idx]
        gt_hr = load_gt_highres("cam00", frame_idx)
        pred_hr = render_highres(
            camera, timestamp, gaussians, pipeline, background,
            scene.dataset_type,
        )

        p, s, lp = compute_metrics(pred_hr, gt_hr, lpips_model)
        sample_name = f"heldout_cam00_{frame_idx:04d}"
        fullimg_rows.append(["heldout_camera", "cam00", frame_idx, p, s, lp])

        bbox = otsu_bboxes.get("cam00")
        if bbox is not None:
            bp, bs, blp = compute_bbox_metrics(pred_hr, gt_hr, bbox, lpips_model)
            bbox_rows.append(["heldout_camera", "cam00", frame_idx,
                              bp, bs, blp, p, s, lp])

        if save_images:
            save_image(pred_hr, os.path.join(output_root, sample_name + "_pred.png"))
            save_image(gt_hr, os.path.join(output_root, sample_name + "_gt.png"))
            error = (pred_hr - gt_hr).abs().mean(dim=0)
            save_image(error / (error.max() + 1e-8),
                       os.path.join(output_root, sample_name + "_error.png"))

        del pred_hr, gt_hr
        torch.cuda.empty_cache()

    # ── Save full-image metrics CSV ──
    csv_path = os.path.join(output_root, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "camera", "frame", "PSNR", "SSIM", "LPIPS"])
        w.writerows(fullimg_rows)

    # ── Save bbox metrics CSV ──
    bbox_csv = os.path.join(output_root, "bbox_metrics.csv")
    with open(bbox_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "camera", "frame",
                     "PSNR_bbox", "SSIM_bbox", "LPIPS_bbox",
                     "PSNR_full", "SSIM_full", "LPIPS_full"])
        w.writerows(bbox_rows)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  HIGH-RES RESULTS (k={gap}, {HIGHRES_W}x{HIGHRES_H})")
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
    # For k=8: also evaluate ALL 50 frames (sawtooth plot)
    # ════════════════════════════════════════════════════════════
    if gap == 8:
        print(f"\n{'='*60}")
        print(f"  k=8: Evaluating all 50 frames at {HIGHRES_W}x{HIGHRES_H}")
        print(f"{'='*60}")

        all50_dir = os.path.join(output_root, "all50")
        os.makedirs(all50_dir, exist_ok=True)

        train_frames_set = set(range(0, 49, gap))
        test_frames_set = set(range(gap // 2, 49, gap))
        all_frames = list(range(FRAME_COUNT))

        # ── cam00 all 50 ──
        cam00_results = []
        print(f"\n  cam00: {FRAME_COUNT} frames")
        for frame_idx in tqdm(all_frames, desc="k=8 cam00 all50"):
            timestamp = frame_idx / 49.0
            camera = test_cameras[frame_idx]
            gt_hr = load_gt_highres("cam00", frame_idx)
            pred_hr = render_highres(
                camera, timestamp, gaussians, pipeline, background,
                scene.dataset_type,
            )
            p, s, lp = compute_metrics(pred_hr, gt_hr, lpips_model)
            cam00_results.append({
                "frame": frame_idx,
                "PSNR": p, "SSIM": s, "LPIPS": lp,
                "is_train_frame": frame_idx in train_frames_set,
                "is_test_midpoint": frame_idx in test_frames_set,
            })
            del pred_hr, gt_hr
            torch.cuda.empty_cache()

        # ── Training cameras all 50 ──
        print(f"  {num_train_cams} training cameras x {FRAME_COUNT} frames")
        train_cam_results = {f: [] for f in all_frames}

        for cam_idx in tqdm(range(num_train_cams), desc="k=8 train all50"):
            base_idx = cam_idx * frames_per_cam
            pose_camera = train_cameras[base_idx]
            camera_name = get_camera_name_from_train_dataset(train_dataset, base_idx)

            for frame_idx in all_frames:
                timestamp = frame_idx / 49.0
                gt_hr = load_gt_highres(camera_name, frame_idx)
                pred_hr = render_highres(
                    pose_camera, timestamp, gaussians, pipeline, background,
                    scene.dataset_type,
                )
                p, s, lp = compute_metrics(pred_hr, gt_hr, lpips_model)
                train_cam_results[frame_idx].append({
                    "camera": camera_name, "PSNR": p, "SSIM": s, "LPIPS": lp,
                })
                del pred_hr, gt_hr
                torch.cuda.empty_cache()

        train_cam_avg = []
        for frame_idx in all_frames:
            entries = train_cam_results[frame_idx]
            train_cam_avg.append({
                "frame": frame_idx,
                "PSNR": np.mean([e["PSNR"] for e in entries]),
                "SSIM": np.mean([e["SSIM"] for e in entries]),
                "LPIPS": np.mean([e["LPIPS"] for e in entries]),
                "is_train_frame": frame_idx in train_frames_set,
                "is_test_midpoint": frame_idx in test_frames_set,
                "num_cameras": len(entries),
            })

        cam00_csv = os.path.join(all50_dir, "cam00_all50.csv")
        with open(cam00_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "frame", "PSNR", "SSIM", "LPIPS", "is_train_frame", "is_test_midpoint",
            ])
            w.writeheader()
            w.writerows(cam00_results)

        train_csv = os.path.join(all50_dir, "train_cameras_all50.csv")
        with open(train_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "frame", "PSNR", "SSIM", "LPIPS", "is_train_frame", "is_test_midpoint",
                "num_cameras",
            ])
            w.writeheader()
            w.writerows(train_cam_avg)

        print(f"\n  Saved: {cam00_csv}")
        print(f"  Saved: {train_csv}")

    print(f"\nAll results saved to: {output_root}")


if __name__ == "__main__":
    main()
