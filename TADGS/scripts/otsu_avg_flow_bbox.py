import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dynamic_bbox_optical_flow import PlugAndPlayOpticalFlow


def compute_average_flow_and_otsu_bbox(
    scene_dir: Path,
    cam_name: str = "cam00",
    num_frames: int = 50,
    stride: int = 2,
    backend: str = "farneback",
    fb_consistency: bool = True,
):
    """
    1. Compute optical flow magnitude for every frame t in [0, num_frames - 1].
    2. Average the optical flow magnitude across all frames: avg_flow_mag(H, W).
    3. Compute Otsu's threshold on avg_flow_mag.
    4. Extract the minimal axis-aligned bounding box enclosing all pixels > Otsu's threshold.
    """
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


def save_otsu_visualization(res: dict, out_fig: Path, cam_name: str = "cam00", num_frames: int = 50, ref_frame: int = 20):
    avg_flow = res["avg_flow_mag"]
    otsu_mask = res["otsu_mask"]
    bbox = res["bbox"]
    tau_px = res["otsu_thresh_px"]
    tau_u8 = res["otsu_thresh_u8"]
    rgb_ref = res["frames_rgb"][ref_frame]
    H, W = avg_flow.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (0, 0): Average Optical Flow Magnitude across all frames
    ax0 = axes[0, 0]
    im0 = ax0.imshow(avg_flow, cmap="magma", vmin=0, vmax=max(float(avg_flow.max()), 1e-4))
    plt.colorbar(im0, ax=ax0, fraction=0.036, pad=0.03, label="Mean Flow Magnitude (px)")
    ax0.set_title(
        f"1. Temporal Mean Optical Flow Magnitude (t=0..{num_frames-1})\n"
        f"Max Mean Flow: {avg_flow.max():.3f} px | Global Mean: {avg_flow.mean():.4f} px",
        fontsize=11, fontweight="bold"
    )
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Panel (0, 1): Histogram of Non-Zero Average Flow + Otsu Threshold
    ax1 = axes[0, 1]
    nonzero_vals = avg_flow[avg_flow > 1e-4]
    if len(nonzero_vals) > 0:
        ax1.hist(nonzero_vals, bins=60, color="#3b82f6", edgecolor="black", alpha=0.8, log=True)
    ax1.axvline(tau_px, color="#ef4444", linestyle="--", linewidth=2.5,
                label=f"Otsu Threshold τ = {tau_px:.3f} px\n({tau_u8:.0f}/255 in 8-bit)")
    ax1.set_title("2. Average Flow Histogram (Log-Count) & Otsu Threshold", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Temporal Average Flow Magnitude (px)", fontsize=10)
    ax1.set_ylabel("Pixel Count (log scale)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel (1, 0): Otsu Thresholded Binary Map + Minimal Enclosing Bounding Box
    ax2 = axes[1, 0]
    ax2.imshow(otsu_mask, cmap="gray")
    active_px = int((otsu_mask > 0).sum())
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.5, edgecolor="#ef4444", facecolor="none")
        ax2.add_patch(rect)
        ax2.set_title(
            f"3. Pixels Above Otsu Threshold (Mean Flow > {tau_px:.3f} px)\n"
            f"Active Pixels: {active_px} ({100.0 * active_px / (H * W):.2f}% of frame)",
            fontsize=11, fontweight="bold"
        )
    else:
        ax2.set_title("3. Pixels Above Otsu Threshold (None)", fontsize=11, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel (1, 1): Minimal Bounding Box Overlaid on Reference RGB Frame
    ax3 = axes[1, 1]
    overlay = rgb_ref.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        dark = (overlay.astype(np.float32) * 0.42).astype(np.uint8)
        dark[y0:y1, x0:x1] = overlay[y0:y1, x0:x1]
        overlay = dark
    ax3.imshow(overlay)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        rect2 = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.8, edgecolor="#00ff88", facecolor="none")
        ax3.add_patch(rect2)
        bw, bh = x1 - x0, y1 - y0
        box_pct = 100.0 * (bw * bh) / (H * W)
        ax3.set_title(
            f"4. Minimal Otsu Enclosing BBox on {cam_name} (t={ref_frame})\n"
            f"Size: {bw}×{bh} px ({box_pct:.1f}% area) | Coords: [{x0}:{x1}, {y0}:{y1}]",
            fontsize=11, fontweight="bold"
        )
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.suptitle(
        f"Video-Level Average Optical Flow + Otsu Threshold Minimal Bounding Box ({cam_name}, {num_frames} Frames)",
        fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Otsu average-flow bounding box figure to {out_fig}")


def save_otsu_video(res: dict, out_video: Path, cam_name: str = "cam00", fps: int = 15):
    frames_bgr = res["frames_bgr"]
    bbox = res["bbox"]
    tau_px = res["otsu_thresh_px"]
    num_frames = len(frames_bgr)
    H, W = frames_bgr[0].shape[:2]

    raw_mp4 = out_video.with_name(out_video.stem + "_raw.mp4")
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))

    for t, frame in enumerate(frames_bgr):
        vis = frame.copy()
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            dimmed = (vis.astype(np.float32) * 0.45).astype(np.uint8)
            dimmed[y0:y1, x0:x1] = vis[y0:y1, x0:x1]
            vis = dimmed
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (136, 255, 0), 2)
            bbox_str = f"Otsu BBox (tau={tau_px:.2f}px): {x1-x0}x{y1-y0} [{x0}:{x1}, {y0}:{y1}]"
        else:
            bbox_str = "No BBox"

        hud = f"{cam_name} | Frame {t:02d}/{num_frames-1:02d} | {bbox_str}"
        cv2.rectangle(vis, (8, 8), (8 + 560, 34), (0, 0, 0), -1)
        cv2.putText(vis, hud, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(vis)

    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_mp4), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_video)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    raw_mp4.unlink(missing_ok=True)
    print(f"Saved Otsu average-flow bounding box video to {out_video}")


def generate_dataset_otsu_bbox_masks(
    scene_dir: Path,
    out_dir: Path,
    num_frames: int = 50,
    stride: int = 2,
    backend: str = "farneback",
):
    """Compute video-level average optical flow and Otsu threshold minimal bounding box
    for each camera, and save the resulting static bounding box masks for all frames."""
    from tqdm import tqdm
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_files = sorted((scene_dir / "colmap_0" / "images").glob("cam*.png"))

    for cam_path in tqdm(cam_files, desc="Generating Otsu Average Flow BBox Masks"):
        cam_name = cam_path.stem
        res = compute_average_flow_and_otsu_bbox(
            scene_dir=scene_dir,
            cam_name=cam_name,
            num_frames=num_frames,
            stride=stride,
            backend=backend,
            fb_consistency=True,
        )
        bbox_mask = res["bbox_mask"]
        bbox = res["bbox"]
        print(f"[{cam_name}] Otsu tau={res['otsu_thresh_px']:.3f} px | BBox={bbox}")
        for t in range(num_frames):
            cv2.imwrite(str(out_dir / f"mask_{cam_name}_frame{t:04d}.png"), bbox_mask)
            cv2.imwrite(str(out_dir / f"bbox_mask_{cam_name}_frame{t:04d}.png"), bbox_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average optical flow across all frames, find Otsu's threshold, and extract minimal BBox.")
    parser.add_argument("--cam", type=str, default="cam00", help="Camera name (e.g. cam00, cam01, ...)")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--stride", type=int, default=2, help="Temporal stride for flow calculation")
    parser.add_argument("--backend", type=str, default="farneback", choices=["farneback", "dis", "raft_small", "raft_large"])
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")
    parser.add_argument("--generate_all", action="store_true", help="Generate Otsu BBox masks across all cameras and frames")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    scene_dir = workspace / "data" / "neural_3d_video" / "scenes" / "coffee_martini"
    out_fig = workspace / "results" / f"otsu_avg_flow_bbox_{args.cam}.png"
    out_video = workspace / "results" / f"otsu_avg_flow_bbox_{args.cam}.mp4"

    if args.generate_all:
        out_masks = workspace / "data" / "neural_3d_video" / "flow_masks" / "coffee_martini" / "otsu_bbox_masks"
        generate_dataset_otsu_bbox_masks(
            scene_dir, out_masks, num_frames=args.num_frames, stride=args.stride, backend=args.backend
        )
    else:
        res = compute_average_flow_and_otsu_bbox(
            scene_dir=scene_dir,
            cam_name=args.cam,
            num_frames=args.num_frames,
            stride=args.stride,
            backend=args.backend,
            fb_consistency=True,
        )
        print(
            f"[{args.cam}] Otsu Threshold = {res['otsu_thresh_px']:.4f} px "
            f"({res['otsu_thresh_u8']:.0f}/255) | Minimal BBox = {res['bbox']} "
            f"({res['bbox'][2]-res['bbox'][0]}x{res['bbox'][3]-res['bbox'][1]} px)"
        )
        save_otsu_visualization(res, out_fig, cam_name=args.cam, num_frames=args.num_frames)
        save_otsu_video(res, out_video, cam_name=args.cam, fps=args.fps)
