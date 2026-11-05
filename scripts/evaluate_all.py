import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

# Add TAD-GS to path
WORKSPACE = Path(__file__).resolve().parent.parent
TADGS_ROOT = WORKSPACE / "TAD-GS"
sys.path.insert(0, str(TADGS_ROOT))
sys.path.insert(0, str(TADGS_ROOT / "utils" / "common"))

from scene.dataset_readers import read_extrinsics_binary, read_intrinsics_binary, readColmapCameras
from scene.dataset import STGSdataset
from helper_train import getrenderpip, getmodel
from metrics import psnr_ours, ssim_ours, rgb_lpips, load_flow_mask


class DummyArgs:
    def __init__(self, source_path, model_path, resolution=1):
        self.sh_degree = 3
        self.source_path = str(source_path)
        self.model_path = str(model_path)
        self.images = "images"
        self.resolution = resolution
        self.white_background = False
        self.data_device = "cpu"
        self.eval = True
        self.model = "tad_gaussian"
        self.loader = "colmap"


def load_trained_gaussians(model_path: Path, duration: int = 50):
    GaussianModel = getmodel("tad_gaussian")
    gaussians = GaussianModel(3)
    gaussians.motion_mode = "fourier_asym"
    gaussians.motion_degree = 4
    gaussians.opacity_mode = "deform_gauss_op_asym"
    gaussians.duration = duration
    gaussians.pcd_downsample = 3
    gaussians.visibility_threshold = 0.2
    gaussians.min_visible_timestamps = 10
    gaussians.span_budget = 70
    gaussians.temporal_span = 10
    gaussians.alpha = 0.8
    gaussians.beta = 0.75
    gaussians.tat = True
    gaussians.tow = True

    ply_path = model_path / "point_cloud" / "best" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Best checkpoint PLY not found at {ply_path}")
    gaussians.load_ply(str(ply_path))
    return gaussians


def eval_camera_set(views, gaussians, render_fn, GRsetting, GRzer, background, flow_masks_path, duration=50,
                    save_dir: Path = None):
    psnrs, ssims, lpipss = [], [], []
    m_psnrs, m_ssims, m_lpipss = [], [], []
    per_frame = {}

    if save_dir is not None:
        (save_dir / "renders").mkdir(parents=True, exist_ok=True)
        (save_dir / "gt").mkdir(parents=True, exist_ok=True)

    for idx in range(len(views)):
        view = views[idx]
        frame_idx = int(round(view.timestamp * duration))
        cam_name = view.image_name

        pkg = render_fn(view, gaussians, background, GRsetting=GRsetting, GRzer=GRzer)
        rendering = torch.clamp(pkg["render"], 0.0, 1.0)
        gt = view.original_image[0:3, :, :].cuda().float()

        mask = load_flow_mask(str(flow_masks_path), cam_name, frame_idx)

        r_hwc = rendering.permute(1, 2, 0)
        g_hwc = gt.permute(1, 2, 0)

        p_val, mp_val = psnr_ours(r_hwc, g_hwc, mask)
        s_val, ms_val = ssim_ours(r_hwc, g_hwc, mask.cpu().numpy() if mask is not None else None)
        if mask is not None:
            l_val, ml_val = rgb_lpips(r_hwc, g_hwc, net_name="alex", device="cuda", mask=mask)
        else:
            l_val = rgb_lpips(r_hwc, g_hwc, net_name="alex", device="cuda")
            ml_val = l_val

        psnrs.append(float(p_val))
        ssims.append(float(s_val))
        lpipss.append(float(l_val))
        m_psnrs.append(float(mp_val))
        m_ssims.append(float(ms_val))
        m_lpipss.append(float(ml_val))

        per_frame.setdefault(frame_idx, []).append({
            "cam": cam_name,
            "PSNR": float(p_val),
            "SSIM": float(s_val),
            "LPIPS": float(l_val),
            "Masked-PSNR": float(mp_val),
            "Masked-SSIM": float(ms_val),
            "Masked-LPIPS": float(ml_val),
        })

        if save_dir is not None:
            fname = f"{cam_name}_frame{frame_idx:04d}.png"
            torchvision.utils.save_image(rendering, str(save_dir / "renders" / fname))
            torchvision.utils.save_image(gt, str(save_dir / "gt" / fname))

    summary = {
        "num_views": len(views),
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "LPIPS": float(np.mean(lpipss)),
        "Masked-PSNR": float(np.mean(m_psnrs)),
        "Masked-SSIM": float(np.mean(m_ssims)),
        "Masked-LPIPS": float(np.mean(m_lpipss)),
    }
    return summary, per_frame


def main():
    duration = 50
    scene_path = WORKSPACE / "data" / "neural_3d_video" / "scenes" / "coffee_martini" / "colmap_0"
    flow_masks_path = WORKSPACE / "data" / "neural_3d_video" / "flow_masks" / "coffee_martini" / "valid_masks"
    results_dir = WORKSPACE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Read all 18 cameras x 50 frames once
    cam_extrinsics = read_extrinsics_binary(str(scene_path / "sparse" / "0" / "images.bin"))
    cam_intrinsics = read_intrinsics_binary(str(scene_path / "sparse" / "0" / "cameras.bin"))
    cam_infos = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=str(scene_path / "images"),
        near=0.01,
        far=100.0,
        startime=0,
        duration=duration,
    )
    cam_infos = sorted(cam_infos, key=lambda x: (x.image_name, x.timestamp))

    def frame_of(c):
        return int(round(c.timestamp * duration))

    render_fn, GRsetting, GRzer = getrenderpip("tad_gaussian")
    background = torch.zeros(9, dtype=torch.float32, device="cuda")

    all_metrics = {}

    with torch.no_grad():
        for k in [2, 4, 8]:
            model_path = WORKSPACE / "runs" / f"tadgs_k{k}"
            if not (model_path / "point_cloud" / "best" / "point_cloud.ply").exists():
                print(f"[Skip] {model_path} not found yet.")
                continue

            print(f"\n================ Evaluating TAD-GS (k={k}) ================")
            gaussians = load_trained_gaussians(model_path, duration=duration)
            dummy_args = DummyArgs(scene_path, model_path, resolution=1)

            max_train_frame = 48
            train_frames = sorted(range(0, max_train_frame + 1, k))
            test_frames = sorted(range(k // 2, max_train_frame, k))

            # Setting 1: Training cameras (cam01..cam17), held-out midpoint times
            s1_infos = [c for c in cam_infos if "cam00" not in c.image_name and frame_of(c) in test_frames]
            s1_dataset = STGSdataset(s1_infos, dummy_args, loader_type="colmap", split="test", resolution=1)
            print(f"Evaluating Setting 1 (Training cameras, held-out midpoint times): {len(s1_dataset)} views...")
            s1_summary, s1_per_frame = eval_camera_set(
                s1_dataset, gaussians, render_fn, GRsetting, GRzer, background, flow_masks_path, duration=duration
            )

            # Setting 2: Held-out camera (cam00), held-out midpoint times
            s2_infos = [c for c in cam_infos if "cam00" in c.image_name and frame_of(c) in test_frames]
            s2_dataset = STGSdataset(s2_infos, dummy_args, loader_type="colmap", split="test", resolution=1)
            print(f"Evaluating Setting 2 (Held-out cam00, held-out midpoint times): {len(s2_dataset)} views...")
            s2_summary, s2_per_frame = eval_camera_set(
                s2_dataset, gaussians, render_fn, GRsetting, GRzer, background, flow_masks_path, duration=duration,
                save_dir=results_dir / f"renders_k{k}_cam00"
            )

            # Also render key visual comparison frames on cam00 (e.g., frame 20, 21, 22) for visual figure
            vis_frames = sorted(set(test_frames) | {12, 20, 21, 22, 28})
            vis_infos = [c for c in cam_infos if "cam00" in c.image_name and frame_of(c) in vis_frames]
            vis_dataset = STGSdataset(vis_infos, dummy_args, loader_type="colmap", split="test", resolution=1)
            eval_camera_set(
                vis_dataset, gaussians, render_fn, GRsetting, GRzer, background, flow_masks_path, duration=duration,
                save_dir=results_dir / f"renders_k{k}_cam00"
            )

            all_metrics[f"k={k}"] = {
                "frame_gap": k,
                "num_gaussians": int(gaussians.get_xyz.shape[0]),
                "train_frames": train_frames,
                "test_frames": test_frames,
                "setting1_train_cams_heldout_time": s1_summary,
                "setting2_heldout_cam_heldout_time": s2_summary,
                "setting1_per_frame": s1_per_frame,
                "setting2_per_frame": s2_per_frame,
            }

            print(f"[k={k}] Setting 1 (Train Cams, Held-out Time): "
                  f"PSNR={s1_summary['PSNR']:.2f}, SSIM={s1_summary['SSIM']:.4f}, LPIPS={s1_summary['LPIPS']:.4f} | "
                  f"Dynamic Masked: PSNR={s1_summary['Masked-PSNR']:.2f}, SSIM={s1_summary['Masked-SSIM']:.4f}, LPIPS={s1_summary['Masked-LPIPS']:.4f}")
            print(f"[k={k}] Setting 2 (Held-out Cam, Held-out Time): "
                  f"PSNR={s2_summary['PSNR']:.2f}, SSIM={s2_summary['SSIM']:.4f}, LPIPS={s2_summary['LPIPS']:.4f} | "
                  f"Dynamic Masked: PSNR={s2_summary['Masked-PSNR']:.2f}, SSIM={s2_summary['Masked-SSIM']:.4f}, LPIPS={s2_summary['Masked-LPIPS']:.4f}")

            # Deliverable 3: For k=8, evaluate per-frame PSNR across ALL 50 frames (0..49)
            if k == 8:
                print("\n[k=8] Computing per-frame metrics across all 50 frames (0..49) for Deliverable 3 plot...")
                all_cam00_infos = [c for c in cam_infos if "cam00" in c.image_name]
                all_cam00_ds = STGSdataset(all_cam00_infos, dummy_args, loader_type="colmap", split="test", resolution=1)
                _, k8_cam00_all_frames = eval_camera_set(
                    all_cam00_ds, gaussians, render_fn, GRsetting, GRzer, background, flow_masks_path, duration=duration,
                    save_dir=results_dir / "renders_k8_cam00_all50"
                )

                # Also evaluate across training cameras (cam01..cam17) for all 50 frames
                all_train_infos = [c for c in cam_infos if "cam00" not in c.image_name]
                all_train_ds = STGSdataset(all_train_infos, dummy_args, loader_type="colmap", split="test", resolution=1)
                _, k8_train_all_frames = eval_camera_set(
                    all_train_ds, gaussians, render_fn, GRsetting, GRzer, background, flow_masks_path, duration=duration
                )

                per_frame_k8 = {}
                for t in range(duration):
                    c00_list = k8_cam00_all_frames.get(t, [])
                    tr_list = k8_train_all_frames.get(t, [])
                    per_frame_k8[t] = {
                        "frame_idx": t,
                        "is_train_frame": t in train_frames,
                        "is_midpoint_test_frame": t in test_frames,
                        "cam00_PSNR": float(np.mean([x["PSNR"] for x in c00_list])) if c00_list else 0.0,
                        "cam00_Masked_PSNR": float(np.mean([x["Masked-PSNR"] for x in c00_list])) if c00_list else 0.0,
                        "train_cams_PSNR": float(np.mean([x["PSNR"] for x in tr_list])) if tr_list else 0.0,
                        "train_cams_Masked_PSNR": float(np.mean([x["Masked-PSNR"] for x in tr_list])) if tr_list else 0.0,
                    }
                all_metrics["k=8"]["all_50_frames"] = per_frame_k8

    out_json = results_dir / "tadgs_evaluation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved complete evaluation metrics to {out_json}")


if __name__ == "__main__":
    main()

