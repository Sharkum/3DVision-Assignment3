"""Render all 50 frames of cam00 (render + GT) for each frame-gap k.

Usage: python scripts/render_all_frames.py [k ...]   (default: 2 4 8)
Output: results/renders_k{k}_cam00_all50/{renders,gt}/
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_all import *  # noqa: F401,F403  (reuses loaders / eval_camera_set)


def main():
    ks = [int(a) for a in sys.argv[1:]] or [2, 4, 8]
    duration = 50
    scene_path = WORKSPACE / "data/neural_3d_video/scenes/coffee_martini/colmap_0"
    flow_masks_path = WORKSPACE / "data/neural_3d_video/flow_masks/coffee_martini/bbox_masks"

    ext = read_extrinsics_binary(str(scene_path / "sparse/0/images.bin"))
    intr = read_intrinsics_binary(str(scene_path / "sparse/0/cameras.bin"))
    infos = readColmapCameras(cam_extrinsics=ext, cam_intrinsics=intr,
                              images_folder=str(scene_path / "images"),
                              near=0.01, far=100.0, startime=0, duration=duration)
    infos = sorted([c for c in infos if "cam00" in c.image_name], key=lambda c: c.timestamp)

    render_fn, GRsetting, GRzer = getrenderpip("tad_gaussian")
    background = torch.zeros(9, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        for k in ks:
            model_path = WORKSPACE / "runs" / f"tadgs_k{k}"
            gaussians = load_trained_gaussians(model_path, duration=duration)
            ds = STGSdataset(infos, DummyArgs(scene_path, model_path, 1),
                             loader_type="colmap", split="test", resolution=1)
            out = WORKSPACE / "results" / f"renders_k{k}_cam00_all50"
            print(f"k={k}: rendering {len(ds)} frames -> {out}")
            eval_camera_set(ds, gaussians, render_fn, GRsetting, GRzer, background,
                            flow_masks_path, duration=duration, save_dir=out)


if __name__ == "__main__":
    main()
