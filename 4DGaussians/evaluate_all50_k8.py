
import os
import csv
import copy
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


def load_gt(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def evaluate_at_time(camera, timestamp, gt, gaussians, pipeline, background,
                     lpips_model, cam_type):
    camera = copy.copy(camera)
    camera.time = timestamp

    with torch.no_grad():
        prediction = render(
            camera,
            gaussians,
            pipeline,
            background,
            cam_type=cam_type,
        )["render"].clamp(0, 1)

        gt = gt.cuda().clamp(0, 1)

        pred_batch = prediction.unsqueeze(0)
        gt_batch = gt.unsqueeze(0)

        p = psnr(pred_batch, gt_batch).mean().item()
        s = ssim(pred_batch, gt_batch).mean().item()
        lp = lpips_model(
            pred_batch * 2 - 1,
            gt_batch * 2 - 1,
        ).mean().item()

    return p, s, lp


def main():
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

    safe_state(args.quiet)

    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    hyper = hidden_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, hyper)

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
    )

    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    lpips_model = lpips.LPIPS(net="vgg").cuda().eval()

    output_dir = os.path.join(args.model_path, "all50_evaluation")
    if os.path.exists(output_dir):
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    os.makedirs(output_dir)

    gap = 8
    train_frames = list(range(0, 49, gap))
    test_frames = list(range(gap // 2, 49, gap))
    all_frames = list(range(50))

    # ---------------------------------------------------------
    # Held-out camera (cam00) -- all 50 frames
    # ---------------------------------------------------------
    test_cameras = scene.getTestCameras()
    print(f"Test cameras (cam00) dataset size: {len(test_cameras)}")

    cam00_results = []
    for frame_idx in tqdm(all_frames, desc="cam00 all 50 frames"):
        timestamp = frame_idx / 49.0
        camera = test_cameras[frame_idx]
        gt = camera.original_image.cuda()

        p, s, lp = evaluate_at_time(
            camera, timestamp, gt, gaussians, pipeline,
            background, lpips_model, scene.dataset_type,
        )

        is_train = frame_idx in train_frames
        is_test = frame_idx in test_frames

        cam00_results.append({
            "frame": frame_idx,
            "PSNR": p,
            "SSIM": s,
            "LPIPS": lp,
            "is_train_frame": is_train,
            "is_test_midpoint": is_test,
        })

    # ---------------------------------------------------------
    # Training cameras -- all 50 frames (average over 17 cams)
    # ---------------------------------------------------------
    train_dataset = scene.train_camera.dataset
    train_cameras = scene.getTrainCameras()

    frames_per_camera = len(range(0, 50, gap))
    num_train_cameras = len(train_dataset.image_paths) // frames_per_camera

    print(f"Training cameras: {num_train_cameras}")
    print(f"Frames per camera in train dataset: {frames_per_camera}")

    train_cam_results = {f: [] for f in all_frames}

    for cam_idx in tqdm(range(num_train_cameras), desc="Training cameras all 50"):
        base_idx = cam_idx * frames_per_camera
        pose_camera = train_cameras[base_idx]

        first_image_path = train_dataset.image_paths[base_idx]
        image_dir = os.path.dirname(first_image_path)
        camera_name = os.path.basename(os.path.dirname(image_dir))

        for frame_idx in all_frames:
            timestamp = frame_idx / 49.0
            gt_path = os.path.join(image_dir, f"{frame_idx:04d}.png")
            gt = load_gt(gt_path)

            p, s, lp = evaluate_at_time(
                pose_camera, timestamp, gt, gaussians, pipeline,
                background, lpips_model, scene.dataset_type,
            )
            train_cam_results[frame_idx].append({
                "camera": camera_name,
                "PSNR": p,
                "SSIM": s,
                "LPIPS": lp,
            })

    train_cam_avg = []
    for frame_idx in all_frames:
        entries = train_cam_results[frame_idx]
        avg_psnr = np.mean([e["PSNR"] for e in entries])
        avg_ssim = np.mean([e["SSIM"] for e in entries])
        avg_lpips = np.mean([e["LPIPS"] for e in entries])
        train_cam_avg.append({
            "frame": frame_idx,
            "PSNR": avg_psnr,
            "SSIM": avg_ssim,
            "LPIPS": avg_lpips,
            "is_train_frame": frame_idx in train_frames,
            "is_test_midpoint": frame_idx in test_frames,
            "num_cameras": len(entries),
        })

    # Save CSVs
    cam00_csv = os.path.join(output_dir, "cam00_all50.csv")
    with open(cam00_csv, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "PSNR", "SSIM", "LPIPS", "is_train_frame", "is_test_midpoint",
        ])
        writer.writeheader()
        writer.writerows(cam00_results)

    train_csv = os.path.join(output_dir, "train_cameras_all50.csv")
    with open(train_csv, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "PSNR", "SSIM", "LPIPS", "is_train_frame", "is_test_midpoint",
            "num_cameras",
        ])
        writer.writeheader()
        writer.writerows(train_cam_avg)

    print("\n========== cam00 all 50 frames ==========")
    for r in cam00_results:
        tag = "TRAIN" if r["is_train_frame"] else ("TEST" if r["is_test_midpoint"] else "    ")
        print(f"  [{tag}] frame {r['frame']:2d}: PSNR={r['PSNR']:.4f}")

    print("\n========== Training cameras mean (all 50 frames) ==========")
    for r in train_cam_avg:
        tag = "TRAIN" if r["is_train_frame"] else ("TEST" if r["is_test_midpoint"] else "    ")
        print(f"  [{tag}] frame {r['frame']:2d}: PSNR={r['PSNR']:.4f} (n={r['num_cameras']})")

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
