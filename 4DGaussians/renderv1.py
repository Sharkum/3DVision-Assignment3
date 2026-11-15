
import os
import torch
import torchvision
from argparse import ArgumentParser

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
    ModelHiddenParams,
)


def main():
    parser = ArgumentParser()

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--time", required=True, type=float)
    parser.add_argument("--camera_idx", default=75, type=int)
    parser.add_argument("--output", default=None, type=str)

    args = get_combined_args(parser)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    hyper = hyperparam.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, hyper)

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False
    )

    cameras = scene.getTrainCameras()

    if not 0 <= args.camera_idx < len(cameras):
        raise IndexError("Invalid camera index")

    camera = cameras[args.camera_idx]

    original_time = camera.time
    camera.time = args.time

    print("Camera index:", args.camera_idx)
    print("Original timestamp:", original_time)
    print("Requested timestamp:", camera.time)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(
        bg_color, dtype=torch.float32, device="cuda"
    )

    with torch.no_grad():
        result = render(
            camera,
            gaussians,
            pipe,
            background,
            cam_type=scene.dataset_type
        )

        image = result["render"].clamp(0, 1)

    output_path = getattr(args, "output", None)
    if output_path is None:
        output_path = os.path.join(
            args.model_path,
            "custom_time",
            f"camera_{args.camera_idx}_time_{args.time:.6f}.png"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchvision.utils.save_image(image, output_path)

    print("Saved rendering:", output_path)


if __name__ == "__main__":
    main()
