#!/usr/bin/env python3
"""Render a single frame at low resolution (676x507) for a given k and frame index."""

import os
import sys
import torch
import numpy as np
from PIL import Image

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from utils.general_utils import safe_state

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "some_updates")


def save_image(tensor, path):
    arr = (tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    hidden_params = ModelHiddenParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default="")

    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    args.temporal_gap = args.gap
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

    test_cameras = scene.getTestCameras()
    frame_idx = args.frame
    timestamp = frame_idx / 49.0

    camera = test_cameras[frame_idx]
    camera.time = timestamp

    with torch.no_grad():
        pred = render(camera, gaussians, pipeline, background, cam_type=scene.dataset_type)["render"].clamp(0, 1)

    print(f"Rendered cam00 frame {frame_idx} at {camera.image_width}x{camera.image_height}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"cam00_k{args.gap}_frame{frame_idx:04d}_render.png")
    save_image(pred, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
