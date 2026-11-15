# [A3] Temporal-interpolation evaluation for Spacetime Gaussians (assignment 3).
# Renders EVERY camera at EVERY frame 0..duration-1 from a trained model and computes
# PSNR / SSIM / LPIPS(alex) per image. Each image is tagged with:
#   cam_split : "heldout" (cam00, never trained) | "train" (training cameras)
#   time_split: "train" (frame % k == 0) | "test" (midpoints k/2, 3k/2, ... between trained frames) | "other"
# Outputs (in <model_path>/eval_<iter>/):
#   per_image.csv, summary.json, renders/<cam>/<frame:03d>.png (unless --no_save)
# Usage: python eval_interp.py --model_path log/k8 [--iteration 30000] [--no_save]
# (source_path, k, resolution, model type are read from <model_path>/cfg_args written by train.py)

import sys
sys.path.append("./thirdparty/gaussian_splatting")
import os, csv, json
from argparse import ArgumentParser
import torch, torchvision
from tqdm import tqdm

from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS
from helper_train import getrenderpip, getmodel, trbfunction


def test_frames_for(k, duration):
    last = max(range(0, duration, k))           # last trained frame (48 for k=2,4,8)
    return list(range(k // 2, last, k))         # midpoints only (no extrapolation past last trained frame)


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--heldout", type=str, default="cam00")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)             # merges <model_path>/cfg_args from training

    k = int(getattr(args, "trainstride", 1) or 1)
    duration = int(getattr(args, "duration", 50))
    testframes = test_frames_for(k, duration) if k > 1 else []
    print("[A3] k={} duration={} test frames={}".format(k, duration, testframes))

    dataset = model.extract(args)
    dataset.trainstride = 1                      # load ALL frames of ALL cameras for evaluation
    pipe = pipeline.extract(args)

    outdir = os.path.join(args.model_path, "eval_{}".format(args.iteration))
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        gaussians = getmodel(dataset.model)(dataset.sh_degree, args.rgbfunction)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, duration=duration, loader="colmap")
        render, GRsetting, GRzer = getrenderpip(args.rdpip)   # same renderer as training (any continuous t)
        if gaussians.rgbdecoder is not None:
            gaussians.rgbdecoder.cuda()
            gaussians.rgbdecoder.eval()
        views = scene.getTestCameras() + scene.getTrainCameras()
        if gaussians.ts is None:
            gaussians.ts = torch.ones(1, 1, views[0].image_height, views[0].image_width).cuda()
        background = torch.zeros(9, dtype=torch.float32, device="cuda")
        lpips_fn = LPIPS("alex", "0.1").cuda().eval()          # same LPIPS as repo test.py (inputs in [0,1])

        rows = []
        for view in tqdm(views, desc="[A3] render+metrics"):
            frame = int(round(view.timestamp * duration))
            img = torch.clamp(render(view, gaussians, pipe, background, scaling_modifier=1.0, basicfunction=trbfunction,
                                     GRsetting=GRsetting, GRzer=GRzer)["render"], 0.0, 1.0)
            gt = view.original_image[0:3].cuda().float()
            if gt.max() > 1.5:                   # gt stored as uint8
                gt = gt / 255.0
            a, b = img.unsqueeze(0), gt.unsqueeze(0)
            cam_split = "heldout" if view.image_name == args.heldout else "train"
            time_split = "train" if frame % k == 0 else ("test" if frame in testframes else "other")
            rows.append(dict(cam=view.image_name, frame=frame, cam_split=cam_split, time_split=time_split,
                             psnr=psnr(a, b).item(), ssim=ssim(a, b).item(), lpips=lpips_fn(a, b).item()))
            if not args.no_save:
                d = os.path.join(outdir, "renders", view.image_name)
                os.makedirs(d, exist_ok=True)
                torchvision.utils.save_image(img, os.path.join(d, "{:03d}.png".format(frame)))

    rows.sort(key=lambda r: (r["cam"], r["frame"]))
    with open(os.path.join(outdir, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = dict(k=k, iteration=args.iteration, heldout=args.heldout, test_frames=testframes,
                   train_frames=list(range(0, duration, k)))
    for cs in ["train", "heldout"]:
        for ts in ["test", "train", "other"]:
            sel = [r for r in rows if r["cam_split"] == cs and r["time_split"] == ts]
            if sel:
                key = "{}cams_{}frames".format(cs, ts)
                summary[key] = {m: mean([r[m] for r in sel]) for m in ["psnr", "ssim", "lpips"]}
                summary[key]["n_images"] = len(sel)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
