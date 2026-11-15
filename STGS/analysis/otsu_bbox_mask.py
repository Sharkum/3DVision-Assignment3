# [A3] Team-common GLOBAL (per-camera, static) dynamic-region mask. The algorithm is COPIED UNCHANGED from the
#   team repo Sharkum/3DVision-Assignment3, branch TADGS @6e8703c, scripts/otsu_avg_flow_bbox.py
#   (compute_average_flow_and_otsu_bbox, generate_dataset_otsu_bbox_masks).
# Changes, interface only (no change to the mask itself):
#   - matplotlib figure + ffmpeg video removed (env stg_a3 has no matplotlib);
#   - PlugAndPlayOpticalFlow imported from our unchanged copy analysis/dyn_bbox_mask.py;
#   - --scene / --out arguments instead of the hard-coded data/neural_3d_video/... path;
#   - writes mask_stats.json (bbox per cam, area fraction) + overlays; no duplicate bbox_mask_* files.
# Definition (TADGS defaults): Farneback flow t -> t+2 (t-2 for the last 2 frames), FB consistency;
#   per camera, mean |flow| over all 50 frames -> Otsu threshold -> smallest box around all pixels above it.
#   The same box is used for every frame of that camera.
# Mask files: <out>/mask_<cam>_frame<tttt>.png, 255 = dynamic (inside the bbox). Same format as dyn_bbox_mask.py,
#   so eval_common.py works unchanged with --mask_dir <out>.
# Usage: python analysis/otsu_bbox_mask.py --scene data/coffee_martini --out results_otsu/otsu_bbox_masks
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from dyn_bbox_mask import PlugAndPlayOpticalFlow


def compute_average_flow_and_otsu_bbox(
    scene_dir: Path,
    cam_name: str = "cam00",
    num_frames: int = 50,
    stride: int = 2,
    backend: str = "farneback",
    fb_consistency: bool = True,
):
    """(unchanged from TADGS branch) mean flow magnitude over all frames -> Otsu -> minimal enclosing bbox."""
    frames_bgr = []
    frames_rgb = []
    for t in range(num_frames):
        img_path = scene_dir / f"colmap_{t}" / "images" / f"{cam_name}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Frame not found: {img_path}")
        bgr = cv2.imread(str(img_path))
        frames_bgr.append(bgr)
        frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    H, W = frames_rgb[0].shape[:2]
    flow_estimator = PlugAndPlayOpticalFlow(backend=backend, fb_consistency=fb_consistency)

    flow_mag_accum = np.zeros((H, W), dtype=np.float64)
    for t in range(num_frames):
        t_other = min(t + stride, num_frames - 1) if t + stride < num_frames else max(0, t - stride)
        flow = flow_estimator.compute_flow(frames_rgb[t], frames_rgb[t_other])
        flow_mag = np.linalg.norm(flow, axis=-1)
        flow_mag_accum += flow_mag

    avg_flow_mag = (flow_mag_accum / float(num_frames)).astype(np.float32)
    max_val = float(avg_flow_mag.max())

    # Normalize avg_flow_mag to uint8 [0, 255] to apply OpenCV's Otsu thresholding
    if max_val > 1e-8:
        avg_flow_u8 = np.clip((avg_flow_mag / max_val) * 255.0, 0, 255).astype(np.uint8)
        otsu_u8_thresh, otsu_binary_u8 = cv2.threshold(
            avg_flow_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_thresh_px = (float(otsu_u8_thresh) / 255.0) * max_val
    else:
        otsu_u8_thresh = 0.0
        otsu_thresh_px = 0.0
        otsu_binary_u8 = np.zeros((H, W), dtype=np.uint8)

    # Minimal bounding box enclosing all pixels above Otsu's threshold
    ys, xs = np.where(otsu_binary_u8 > 0)
    bbox_mask = np.zeros((H, W), dtype=np.uint8)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bbox = (x0, y0, x1, y1)
        bbox_mask[y0:y1, x0:x1] = 255
    else:
        bbox = None

    return {
        "frames_bgr": frames_bgr,
        "frames_rgb": frames_rgb,
        "avg_flow_mag": avg_flow_mag,
        "otsu_thresh_u8": float(otsu_u8_thresh),
        "otsu_thresh_px": otsu_thresh_px,
        "otsu_mask": otsu_binary_u8,
        "bbox": bbox,
        "bbox_mask": bbox_mask,
    }


def generate_dataset_otsu_bbox_masks(scene_dir: Path, out_dir: Path, num_frames: int = 50, stride: int = 2,
                                     backend: str = "farneback"):
    """(TADGS logic unchanged; [A3] additions: stats + overlays, no duplicate bbox_mask_* files)"""
    from tqdm import tqdm
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_files = sorted((scene_dir / "colmap_0" / "images").glob("cam*.png"))
    stats = {}

    for cam_path in tqdm(cam_files, desc="Generating Otsu Average Flow BBox Masks"):
        cam_name = cam_path.stem
        res = compute_average_flow_and_otsu_bbox(
            scene_dir=scene_dir, cam_name=cam_name, num_frames=num_frames,
            stride=stride, backend=backend, fb_consistency=True,
        )
        bbox_mask = res["bbox_mask"]
        bbox = res["bbox"]
        print(f"[{cam_name}] Otsu tau={res['otsu_thresh_px']:.3f} px | BBox={bbox}")
        for t in range(num_frames):
            cv2.imwrite(str(out_dir / f"mask_{cam_name}_frame{t:04d}.png"), bbox_mask)
        # [A3] stats + one overlay per camera (box green, above-Otsu pixels red) for checking
        stats[cam_name] = dict(bbox=bbox, area_frac=float((bbox_mask > 0).mean()),
                               otsu_thresh_px=res["otsu_thresh_px"],
                               otsu_active_frac=float((res["otsu_mask"] > 0).mean()))
        ov = res["frames_bgr"][20].copy()
        ov[res["otsu_mask"] > 0] = (0, 0, 255)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            cv2.rectangle(ov, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"overlay_{cam_name}.jpg"), ov)

    fr = [v["area_frac"] for v in stats.values()]
    empty = [c for c, v in stats.items() if v["bbox"] is None]
    summary = dict(source="Sharkum/3DVision-Assignment3@TADGS 6e8703c scripts/otsu_avg_flow_bbox.py (algorithm unchanged)",
                   backend=backend, stride=stride, num_frames=num_frames,
                   bbox_area_frac_mean=float(np.mean(fr)), bbox_area_frac_min=float(np.min(fr)),
                   bbox_area_frac_max=float(np.max(fr)), n_empty=len(empty), empty=empty, per_cam=stats)
    with open(out_dir / "mask_stats.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("bbox area fraction mean {:.1%} (min {:.1%}, max {:.1%}); empty cams: {}".format(
        summary["bbox_area_frac_mean"], summary["bbox_area_frac_min"], summary["bbox_area_frac_max"], empty))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team-common Otsu average-flow global bbox masks (TADGS algorithm)")
    parser.add_argument("--scene", required=True, help="folder with colmap_<t>/images/camXX.png (GT frames)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--backend", type=str, default="farneback", choices=["farneback", "dis", "raft_small", "raft_large"])
    args = parser.parse_args()
    generate_dataset_otsu_bbox_masks(Path(args.scene), Path(args.out), num_frames=args.num_frames,
                                     stride=args.stride, backend=args.backend)
    print("MASKS DONE ->", args.out)
