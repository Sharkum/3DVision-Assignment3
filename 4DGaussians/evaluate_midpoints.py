
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


# def save_image(tensor, path):
#     arr = (
#         tensor.detach()
#         .clamp(0, 1)
#         .permute(1, 2, 0)
#         .cpu()
#         .numpy() * 255
#     ).astype(np.uint8)
#     Image.fromarray(arr).save(path)

def save_image(tensor, path):
    tensor = tensor.detach().clamp(0, 1).cpu()

    if tensor.ndim == 2:
        # Grayscale image: (H, W)
        arr = (tensor.numpy() * 255).astype(np.uint8)

    elif tensor.ndim == 3:
        # RGB image: (C, H, W) -> (H, W, C)
        arr = (
            tensor.permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)

    else:
        raise ValueError(f"Unexpected image dimensions: {tensor.shape}")

    Image.fromarray(arr).save(path)



# def evaluate_sample(camera, timestamp, gt, gaussians,
#                     pipeline, background, lpips_model, output_dir,
#                     sample_name, stage, cam_type):

def evaluate_sample(camera, timestamp, gt, gaussians,
                    pipeline, background, lpips_model, output_dir,
                    sample_name, cam_type):
    camera = copy.copy(camera)
    camera.time = timestamp

    with torch.no_grad():
        # prediction = render(
        #     camera,
        #     gaussians,
        #     pipeline,
        #     background,
        #     stage=stage,
        #     cam_type=cam_type,
        # )["render"].clamp(0, 1)

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

    save_image(prediction, os.path.join(output_dir, sample_name + "_pred.png"))
    save_image(gt, os.path.join(output_dir, sample_name + "_gt.png"))

    error = (prediction - gt).abs().mean(dim=0)
    save_image(
        error / (error.max() + 1e-8),
        os.path.join(output_dir, sample_name + "_error.png"),
    )

    return p, s, lp


def main():
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    hidden_params = ModelHiddenParams(parser)

    # parser.add_argument("--iteration", type=int, default=-1)
    # parser.add_argument("--gap", type=int, required=True)
    # parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default="")
    
    # args = get_combined_args(parser)
    # args = parser.parse_args()

    # if args.configs:
    #     import mmcv
    #     from utils.params_utils import merge_hparams

    #     config = mmcv.Config.fromfile(args.configs)
    #     args = merge_hparams(args, config)
    args = get_combined_args(parser)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    args.temporal_gap = args.gap
    
    
    # Preserve the requested temporal gap.
    args.temporal_gap = args.gap
    if args.gap not in (2, 4, 8):
        raise ValueError("gap must be 2, 4, or 8")

    # Ensure the dataset is loaded with the same temporal gap as training.
    args.temporal_gap = args.gap

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

    gap = args.gap
    midpoint_frames = list(range(gap // 2, 49, gap))

    # output_dir = os.path.join(
    #     args.model_path,
    #     "midpoint_evaluation",
    #     f"gap_{gap}",
    # )
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(
        args.model_path,
        "midpoint_evaluation_v4",
        f"gap_{gap}",
    )
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing results: {output_dir}"
        )
    os.makedirs(output_dir)
    all_results = []

    # ---------------------------------------------------------
    # Setting 1: training cameras, held-out time
    # ---------------------------------------------------------
    train_dataset = scene.train_camera.dataset
    train_cameras = scene.getTrainCameras()

    frames_per_camera = len(range(0, 50, gap))
    num_train_cameras = len(train_dataset.image_paths) // frames_per_camera

    print(f"Training cameras: {num_train_cameras}")
    print(f"Midpoint frames: {midpoint_frames}")

    for cam_idx in tqdm(range(num_train_cameras), desc="Training cameras"):
        base_idx = cam_idx * frames_per_camera

        # All frames belonging to this camera share the same pose.
        pose_camera = train_cameras[base_idx]

        first_image_path = train_dataset.image_paths[base_idx]
        image_dir = os.path.dirname(first_image_path)
        camera_name = os.path.basename(
            os.path.dirname(image_dir)
        )

        for frame_idx in midpoint_frames:
            timestamp = frame_idx / 49.0

            gt_path = os.path.join(
                image_dir,
                f"{frame_idx:04d}.png",
            )
            gt = load_gt(gt_path)

            sample_name = f"train_{camera_name}_{frame_idx:04d}"

            metrics = evaluate_sample(
                pose_camera,
                timestamp,
                gt,
                gaussians,
                pipeline,
                background,
                lpips_model,
                output_dir,
                sample_name,
                scene.dataset_type,
            )

            all_results.append(
                ["training_cameras", camera_name, frame_idx, *metrics]
            )

    # ---------------------------------------------------------
    # Setting 2: held-out camera cam00, held-out time
    # ---------------------------------------------------------
    test_cameras = scene.getTestCameras()

    for frame_idx in tqdm(midpoint_frames, desc="Held-out camera"):
        timestamp = frame_idx / 49.0

        # Test dataset contains all 50 frames from cam00.
        camera = test_cameras[frame_idx]
        gt = camera.original_image.cuda()

        sample_name = f"heldout_cam00_{frame_idx:04d}"

        metrics = evaluate_sample(
            camera,
            timestamp,
            gt,
            gaussians,
            pipeline,
            background,
            lpips_model,
            output_dir,
            sample_name,
            scene.dataset_type,
        )

        all_results.append(
            ["heldout_camera", "cam00", frame_idx, *metrics]
        )

    # Save per-sample metrics.
    csv_path = os.path.join(output_dir, "metrics.csv")

    with open(csv_path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["setting", "camera", "frame", "PSNR", "SSIM", "LPIPS"]
        )
        writer.writerows(all_results)

    # Print averages for both evaluation settings.
    print("\n========== RESULTS ==========")

    for setting in ("training_cameras", "heldout_camera"):
        rows = [r for r in all_results if r[0] == setting]

        values = np.array([r[3:6] for r in rows])
        averages = values.mean(axis=0)

        print(
            f"{setting}: "
            f"PSNR={averages[0]:.4f}, "
            f"SSIM={averages[1]:.4f}, "
            f"LPIPS={averages[2]:.4f}, "
            f"samples={len(rows)}"
        )

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
