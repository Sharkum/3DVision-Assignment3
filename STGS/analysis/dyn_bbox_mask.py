# [A3] Team-common dynamic-region mask. The algorithm is COPIED UNCHANGED from the team repo
#   Sharkum/3DVision-Assignment3, branch TADGS, scripts/dynamic_bbox_optical_flow.py
#   (PlugAndPlayOpticalFlow, find_dynamic_region_bbox, generate_dataset_bbox_masks).
# Changes, interface only (no change to the mask itself):
#   - the matplotlib import and demo figure are removed (env stg_a3 has no matplotlib);
#   - --scene / --out arguments instead of the hard-coded data/neural_3d_video/... path;
#   - also writes mask_stats.json (bbox area fraction per cam/frame) and a few overlays to check the masks.
# Mask files: <out>/mask_<cam>_frame<tttt>.png, 255 = dynamic (inside the bbox), 0 = static.
# Definition (TADGS defaults): Farneback flow t -> t+2 (t-2 for the last 2 frames), forward-backward
# consistency + photometric-change filter; pixels with |flow| > 0.5 px; mask = their smallest enclosing box.
# Usage: python analysis/dyn_bbox_mask.py --scene data/coffee_martini --out results_common/bbox_masks
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch


class PlugAndPlayOpticalFlow:
    """Simple plug-and-play wrapper supporting:
    1. 'raft_small' / 'raft_large' (Pretrained deep optical flow in torchvision.models.optical_flow)
    2. 'dis' (OpenCV Dense Inverse Search optical flow - fast, accurate, zero weights required)
    3. 'farneback' (OpenCV Farneback polynomial expansion optical flow - zero weights required)
    """

    def __init__(self, backend: str = "farneback", device: str = "cuda", fb_consistency: bool = True):
        self.backend = backend.lower()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.fb_consistency = fb_consistency
        self.model = None
        self.transforms = None

        if self.backend in ("raft_small", "raft_large"):
            from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights
            if self.backend == "raft_small":
                weights = Raft_Small_Weights.DEFAULT
                self.model = raft_small(weights=weights, progress=False).to(self.device).eval()
            else:
                weights = Raft_Large_Weights.DEFAULT
                self.model = raft_large(weights=weights, progress=False).to(self.device).eval()
            self.transforms = weights.transforms()
        elif self.backend == "dis":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        elif self.backend == "farneback":
            pass
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: 'dis', 'farneback', 'raft_small', 'raft_large'.")

    @torch.no_grad()
    def _raw_flow(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> np.ndarray:
        H, W = img1_rgb.shape[:2]

        if self.backend in ("raft_small", "raft_large"):
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            t1 = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0)
            t2 = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0)
            if pad_h > 0 or pad_w > 0:
                t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h), mode="replicate")
                t2 = torch.nn.functional.pad(t2, (0, pad_w, 0, pad_h), mode="replicate")
            t1, t2 = self.transforms(t1, t2)
            flow_pred = self.model(t1.to(self.device), t2.to(self.device))[-1]
            return flow_pred[0, :, :H, :W].permute(1, 2, 0).cpu().numpy()

        gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

        if self.backend == "dis":
            return self.dis.calc(gray1, gray2, None)
        else:
            return cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

    @torch.no_grad()
    def compute_flow(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> np.ndarray:
        """Compute dense optical flow (H, W, 2) in pixels from img1_rgb to img2_rgb,
        with optional forward-backward consistency + photometric change weighting to suppress
        aperture-problem drift on static repetitive textures (e.g. window blinds)."""
        flow_fwd = self._raw_flow(img1_rgb, img2_rgb)
        if not self.fb_consistency:
            return flow_fwd

        flow_bwd = self._raw_flow(img2_rgb, img1_rgb)
        H, W = img1_rgb.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        map_x = grid_x + flow_fwd[..., 0]
        map_y = grid_y + flow_fwd[..., 1]
        bwd_warped = cv2.remap(flow_bwd, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Forward-backward cycle consistency error
        fb_err = np.linalg.norm(flow_fwd + bwd_warped, axis=-1)
        fwd_mag = np.linalg.norm(flow_fwd, axis=-1)
        valid_fb = fb_err < (0.5 * fwd_mag + 0.75)

        # Also require minimal photometric difference so static untextured/striped walls are zeroed
        photo_diff = np.mean(np.abs(img1_rgb.astype(np.float32) - img2_rgb.astype(np.float32)) / 255.0, axis=-1)
        photo_diff = cv2.GaussianBlur(photo_diff, (7, 7), 1.5)
        valid = valid_fb & (photo_diff > 0.012)

        return flow_fwd * valid[..., None].astype(np.float32)


def find_dynamic_region_bbox(
    flow: np.ndarray,
    threshold: float = 0.5,
    bbox_pad: int = 0,
):
    """Find the smallest axis-aligned bounding box capturing all pixels with
    optical flow magnitude strictly greater than `threshold`. (unchanged from TADGS branch)"""
    H, W = flow.shape[:2]
    flow_mag = np.linalg.norm(flow, axis=-1)  # (H, W)

    # Active pixels where optical flow magnitude exceeds `threshold`
    active_mask = flow_mag > threshold
    threshold_mask = active_mask.astype(np.uint8) * 255
    bbox_mask = np.zeros((H, W), dtype=np.uint8)

    ys, xs = np.where(active_mask)
    if len(xs) == 0:
        return {
            "bbox": None,
            "bbox_mask": bbox_mask,
            "threshold_mask": threshold_mask,
            "flow_mag": flow_mag,
        }

    # Smallest bounding box capturing all active optical flow pixels
    x0 = max(0, int(xs.min()) - bbox_pad)
    y0 = max(0, int(ys.min()) - bbox_pad)
    x1 = min(W, int(xs.max()) + 1 + bbox_pad)
    y1 = min(H, int(ys.max()) + 1 + bbox_pad)

    bbox_mask[y0:y1, x0:x1] = 255

    return {
        "bbox": (x0, y0, x1, y1),
        "bbox_mask": bbox_mask,
        "threshold_mask": threshold_mask,
        "flow_mag": flow_mag,
    }


def generate_dataset_bbox_masks(
    scene_dir: Path,
    out_dir: Path,
    num_frames: int = 50,
    backend: str = "farneback",
    threshold: float = 0.5,
    bbox_pad: int = 0,
    stride: int = 2,
):
    """Generate dynamic bounding-box masks across all cameras and frames in `scene_dir`.
    (TADGS logic unchanged; [A3] additions: bbox statistics + overlays, no duplicate bbox_mask_* files)"""
    from tqdm import tqdm
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_estimator = PlugAndPlayOpticalFlow(backend=backend, fb_consistency=True)
    cam_files = sorted((scene_dir / "colmap_0" / "images").glob("cam*.png"))
    stats = {}

    for cam_path in tqdm(cam_files, desc=f"Generating BBox masks (tau={threshold})"):
        cam_name = cam_path.stem
        frames_rgb = [
            cv2.cvtColor(cv2.imread(str(scene_dir / f"colmap_{t}" / "images" / f"{cam_name}.png")), cv2.COLOR_BGR2RGB)
            for t in range(num_frames)
        ]
        stats[cam_name] = {}
        for t in range(num_frames):
            t_other = min(t + stride, num_frames - 1) if t + stride < num_frames else max(0, t - stride)
            flow = flow_estimator.compute_flow(frames_rgb[t], frames_rgb[t_other])
            res = find_dynamic_region_bbox(flow, threshold=threshold, bbox_pad=bbox_pad)
            cv2.imwrite(str(out_dir / f"mask_{cam_name}_frame{t:04d}.png"), res["bbox_mask"])
            # [A3] stats + overlays for checking
            stats[cam_name][t] = dict(bbox=res["bbox"], area_frac=float((res["bbox_mask"] > 0).mean()),
                                      active_px=int((res["threshold_mask"] > 0).sum()))
            if cam_name in ("cam00", "cam01") and t in (4, 20, 28, 44):
                ov = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
                if res["bbox"] is not None:
                    x0, y0, x1, y1 = res["bbox"]
                    cv2.rectangle(ov, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
                    ov[res["threshold_mask"] > 0] = (0, 0, 255)
                cv2.imwrite(str(out_dir / f"overlay_{cam_name}_frame{t:04d}.jpg"), ov)

    fr = [v["area_frac"] for c in stats.values() for v in c.values()]
    empty = [(c, t) for c, d in stats.items() for t, v in d.items() if v["bbox"] is None]
    full = [(c, t) for c, d in stats.items() for t, v in d.items() if v["area_frac"] > 0.9]
    summary = dict(source="Sharkum/3DVision-Assignment3@TADGS scripts/dynamic_bbox_optical_flow.py (algorithm unchanged)",
                   backend=backend, threshold_px=threshold, bbox_pad=bbox_pad, stride=stride, num_frames=num_frames,
                   bbox_area_frac_mean=float(np.mean(fr)), bbox_area_frac_min=float(np.min(fr)),
                   bbox_area_frac_max=float(np.max(fr)), n_empty=len(empty), empty=empty,
                   n_over90pct=len(full), over90pct=full, per_cam=stats)
    with open(out_dir / "mask_stats.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("bbox area fraction mean {:.1%} (min {:.1%}, max {:.1%}); empty masks: {}; >90% of frame: {}".format(
        summary["bbox_area_frac_mean"], summary["bbox_area_frac_min"], summary["bbox_area_frac_max"], len(empty), len(full)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team-common optical-flow bbox dynamic masks (TADGS algorithm)")
    parser.add_argument("--scene", required=True, help="folder with colmap_<t>/images/camXX.png (GT frames)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", type=str, default="farneback", choices=["dis", "farneback", "raft_small", "raft_large"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pad", type=int, default=0)
    args = parser.parse_args()
    generate_dataset_bbox_masks(Path(args.scene), Path(args.out), backend=args.backend,
                                threshold=args.threshold, bbox_pad=args.pad)
    print("MASKS DONE ->", args.out)
