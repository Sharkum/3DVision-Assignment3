import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    optical flow magnitude strictly greater than `threshold`.

    Args:
        flow: Dense optical flow array of shape (H, W, 2).
        threshold: Hyperparameter tau_flow (in pixels). Set to 0.0 to capture all
            non-zero optical flow, or > 0.0 to capture flow above `threshold`.
        bbox_pad: Optional padding (in pixels) added around the bounding box.

    Returns:
        dict with:
            - 'bbox': (x_min, y_min, x_max, y_max) enclosing all pixels with ||flow||_2 > threshold
            - 'bbox_mask': uint8 mask (H, W) with 255 inside the bounding box and 0 outside
            - 'threshold_mask': uint8 mask (H, W) of all pixels with flow_mag > threshold
            - 'flow_mag': float32 array (H, W) of optical flow magnitudes
    """
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


def demo_threshold_sweep(scene_dir: Path, out_path: Path, cam_name: str = "cam00", frame_t: int = 20, stride: int = 2):
    """Demonstrate the simple smallest enclosing bounding box for non-zero / thresholded optical flow."""
    img1_path = scene_dir / f"colmap_{frame_t}" / "images" / f"{cam_name}.png"
    img2_path = scene_dir / f"colmap_{frame_t + stride}" / "images" / f"{cam_name}.png"

    img1 = cv2.cvtColor(cv2.imread(str(img1_path)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(str(img2_path)), cv2.COLOR_BGR2RGB)

    flow_estimator = PlugAndPlayOpticalFlow(backend="farneback", fb_consistency=True)
    flow = flow_estimator.compute_flow(img1, img2)

    thresholds = [0.0, 0.5, 1.5]
    fig, axes = plt.subplots(len(thresholds), 3, figsize=(13.5, 10))

    for row, tau in enumerate(thresholds):
        res = find_dynamic_region_bbox(flow, threshold=tau, bbox_pad=0)
        bbox = res["bbox"]
        flow_mag = res["flow_mag"]
        active_px = int((res["threshold_mask"] > 0).sum())

        # Col 0: Flow Magnitude Heatmap
        ax0 = axes[row, 0]
        ax0.imshow(flow_mag, cmap="magma", vmin=0, vmax=max(5.0, float(flow_mag.max())))
        ax0.set_title(f"Optical Flow Magnitude ||u||_2\n(Max: {flow_mag.max():.2f} px)", fontsize=10, fontweight="bold")
        label_str = "τ = 0.0 px (All Non-Zero)" if tau == 0.0 else f"Threshold τ = {tau} px"
        ax0.set_ylabel(label_str, fontsize=11, fontweight="bold")
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Col 1: Thresholded Mask + Smallest Enclosing BBox
        ax1 = axes[row, 1]
        ax1.imshow(res["threshold_mask"], cmap="gray")
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.2, edgecolor="#ff3333", facecolor="none")
            ax1.add_patch(rect)
        ax1.set_title(f"Active Pixels (||u|| > {tau} px)\n({active_px} active pixels)", fontsize=10, fontweight="bold")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Col 2: Smallest Enclosing BBox on RGB Frame
        ax2 = axes[row, 2]
        overlay = img1.copy()
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            dark = (overlay.astype(np.float32) * 0.45).astype(np.uint8)
            dark[y0:y1, x0:x1] = overlay[y0:y1, x0:x1]
            overlay = dark
        ax2.imshow(overlay)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            rect2 = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.5, edgecolor="#00ff88", facecolor="none")
            ax2.add_patch(rect2)
            ax2.set_title(f"Smallest Enclosing BBox: {x1-x0}×{y1-y0} px\n[{x0}:{x1}, {y0}:{y1}]", fontsize=10, fontweight="bold")
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.suptitle(
        f"Smallest Enclosing Bounding Box of Optical Flow > τ ({cam_name}, t={frame_t}→{frame_t+stride})",
        fontsize=13, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved simple bounding-box demonstration to {out_path}")


def generate_dataset_bbox_masks(
    scene_dir: Path,
    out_dir: Path,
    num_frames: int = 50,
    backend: str = "farneback",
    threshold: float = 0.5,
    bbox_pad: int = 0,
    stride: int = 2,
):
    """Generate dynamic bounding-box masks across all cameras and frames in `scene_dir`."""
    from tqdm import tqdm
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_estimator = PlugAndPlayOpticalFlow(backend=backend, fb_consistency=True)
    cam_files = sorted((scene_dir / "colmap_0" / "images").glob("cam*.png"))

    for cam_path in tqdm(cam_files, desc=f"Generating BBox masks (tau={threshold})"):
        cam_name = cam_path.stem
        frames_rgb = [
            cv2.cvtColor(cv2.imread(str(scene_dir / f"colmap_{t}" / "images" / f"{cam_name}.png")), cv2.COLOR_BGR2RGB)
            for t in range(num_frames)
        ]
        for t in range(num_frames):
            t_other = min(t + stride, num_frames - 1) if t + stride < num_frames else max(0, t - stride)
            flow = flow_estimator.compute_flow(frames_rgb[t], frames_rgb[t_other])
            res = find_dynamic_region_bbox(flow, threshold=threshold, bbox_pad=bbox_pad)
            cv2.imwrite(str(out_dir / f"mask_{cam_name}_frame{t:04d}.png"), res["bbox_mask"])
            cv2.imwrite(str(out_dir / f"bbox_mask_{cam_name}_frame{t:04d}.png"), res["bbox_mask"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Optical Flow Smallest Enclosing BBox Finder")
    parser.add_argument("--backend", type=str, default="farneback", choices=["dis", "farneback", "raft_small", "raft_large"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Optical flow magnitude threshold (0.0 for all non-zero flow)")
    parser.add_argument("--pad", type=int, default=0, help="Bounding box padding in pixels")
    parser.add_argument("--generate_all", action="store_true", help="Generate BBox masks for all cameras and frames")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    scene_dir = workspace / "data" / "neural_3d_video" / "scenes" / "coffee_martini"
    out_fig = workspace / "results" / "dynamic_bbox_cc_demo.png"
    demo_threshold_sweep(scene_dir, out_fig)
    if args.generate_all:
        out_masks = workspace / "data" / "neural_3d_video" / "flow_masks" / "coffee_martini" / "bbox_masks"
        generate_dataset_bbox_masks(
            scene_dir, out_masks, backend=args.backend, threshold=args.threshold, bbox_pad=args.pad
        )
