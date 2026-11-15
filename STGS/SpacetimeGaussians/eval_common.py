# [A3] Evaluation with the TEAM-COMMON dynamic mask (bbox mask from analysis/dyn_bbox_mask.py).
# Renders every camera at every frame 0..duration-1 in memory (no 8-bit PNG round trip) and computes:
#   full image (STGS repo metrics, identical to eval_interp.py / authors' test.py):
#     psnr  = utils.image_utils.psnr,  ssim = utils.loss_utils.ssim,  lpips = lpipsPyTorch LPIPS('alex','0.1') on [0,1]
#   dynamic region (mask = mask_<cam>_frame<tttt>.png, 255 = dynamic; same file for every method):
#     dyn_psnr  = 10 log10(1 / MSE over masked pixels, all 3 channels)   (repo psnr restricted to the mask)
#     dyn_ssim  = repo utils.loss_utils.ssimmap averaged over the masked pixels (and the 3 channels)
#     dyn_lpips = repo LPIPS on the bbox crop (render and GT cropped to the mask's box)  [provisional: to be
#                 replaced by the TAD-GS teammate's masked-LPIPS if it differs]
#   Empty mask (no pixel with flow > tau) -> dyn_* = NaN, excluded from averages (counted in the summary).
# Output: <out_dir>/per_image_k<k>.csv  (columns used by analysis/summarize_common.py and plot_perframe.py)
# Usage: python eval_common.py --model_path ../log/stg_full_ds4_k8 --mask_dir ../results_common/bbox_masks
#                              --out_dir ../results_common [--iteration 30000]
import sys
sys.path.append("./thirdparty/gaussian_splatting")
import os, csv, math
from argparse import ArgumentParser
import numpy as np
import torch
import cv2
from tqdm import tqdm

from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from thirdparty.gaussian_splatting.utils.loss_utils import ssim, ssimmap
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS
from helper_train import getrenderpip, getmodel, trbfunction

MIN_CROP = 32  # AlexNet-LPIPS needs a crop of at least ~32x32 px


def load_bbox_mask(mask_dir, cam, frame):
    """Fixed loader for the team bbox masks: mask_<cam>_frame<tttt>.png, 255 = dynamic.
    (The TAD-GS load_flow_mask expects '<cam>_<t-5>__<cam>_<t>.png' with 0 = dynamic, which the bbox script
    does not write.) Returns a bool HxW numpy array."""
    p = os.path.join(mask_dir, "mask_{}_frame{:04d}.png".format(cam, frame))
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(p)
    return m > 127


def test_frames_for(k, duration):
    last = max(range(0, duration, k))
    return list(range(k // 2, last, k))


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--heldout", type=str, default="cam00")
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    k = int(getattr(args, "trainstride", 1) or 1)
    duration = int(getattr(args, "duration", 50))
    testframes = test_frames_for(k, duration) if k > 1 else []
    print("[A3] k={} test frames={}".format(k, testframes))

    dataset = model.extract(args)
    dataset.trainstride = 1                      # load ALL frames of ALL cameras
    pipe = pipeline.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with torch.no_grad():
        gaussians = getmodel(dataset.model)(dataset.sh_degree, args.rgbfunction)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, duration=duration, loader="colmap")
        render, GRsetting, GRzer = getrenderpip(args.rdpip)
        if gaussians.rgbdecoder is not None:
            gaussians.rgbdecoder.cuda()
            gaussians.rgbdecoder.eval()
        views = scene.getTestCameras() + scene.getTrainCameras()
        if gaussians.ts is None:
            gaussians.ts = torch.ones(1, 1, views[0].image_height, views[0].image_width).cuda()
        background = torch.zeros(9, dtype=torch.float32, device="cuda")
        lpips_fn = LPIPS("alex", "0.1").cuda().eval()

        for view in tqdm(views, desc="[A3] render + full/dynamic metrics"):
            frame = int(round(view.timestamp * duration))
            cam = view.image_name
            img = torch.clamp(render(view, gaussians, pipe, background, scaling_modifier=1.0, basicfunction=trbfunction,
                                     GRsetting=GRsetting, GRzer=GRzer)["render"], 0.0, 1.0)
            gt = view.original_image[0:3].cuda().float()
            if gt.max() > 1.5:
                gt = gt / 255.0
            a, b = img.unsqueeze(0), gt.unsqueeze(0)                     # 1x3xHxW, float32, cuda
            row = dict(k=k, cam=cam, frame=frame,
                       cam_split="heldout" if cam == args.heldout else "train",
                       time_split="train" if frame % k == 0 else ("test" if frame in testframes else "other"),
                       psnr=psnr(a, b).mean().item(), ssim=ssim(a, b).item(), lpips=lpips_fn(a, b).item(),
                       dyn_psnr=float("nan"), dyn_ssim=float("nan"), dyn_lpips=float("nan"),
                       dyn_frac=0.0, bbox="")

            m_np = load_bbox_mask(args.mask_dir, cam, frame)
            assert m_np.shape == tuple(a.shape[2:]), (m_np.shape, a.shape)
            npx = int(m_np.sum())
            if npx > 0:
                m = torch.from_numpy(m_np).cuda()                        # HxW bool
                se = ((a - b) ** 2)[0]                                   # 3xHxW
                row["dyn_psnr"] = 10.0 * math.log10(1.0 / se[:, m].mean().item())
                row["dyn_ssim"] = ssimmap(a, b)[0][:, m].mean().item()
                ys, xs = np.where(m_np)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                if min(y1 - y0, x1 - x0) >= MIN_CROP:
                    row["dyn_lpips"] = lpips_fn(a[:, :, y0:y1, x0:x1], b[:, :, y0:y1, x0:x1]).item()
                row["dyn_frac"] = npx / float(m_np.size)
                row["bbox"] = "{} {} {} {}".format(x0, y0, x1, y1)
            rows.append(row)

    rows.sort(key=lambda r: (r["cam"], r["frame"]))
    out = os.path.join(args.out_dir, "per_image_k{}.csv".format(k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("EVAL_COMMON DONE k={} -> {}".format(k, out))
