import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np

# Import the exact flow estimator used across the assignment
from dynamic_bbox_optical_flow import PlugAndPlayOpticalFlow


def flow_to_hsv_bgr(flow: np.ndarray, max_speed: float = 4.0) -> np.ndarray:
    """Standard Middlebury-style optical flow color visualization:
    - Hue: direction of displacement vector
    - Saturation: 255 (full color)
    - Value (brightness): flow magnitude / max_speed
    """
    H, W = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2.0).astype(np.uint8)  # OpenCV Hue in [0, 180)
    hsv[..., 1] = 255
    val = np.clip((mag / max(max_speed, 1e-4)) * 255.0, 0, 255).astype(np.uint8)
    hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_to_magma_bgr(flow: np.ndarray, max_speed: float = 4.0) -> np.ndarray:
    """Magnitude heatmap using cv2.COLORMAP_MAGMA (matches Figure 1 in the report)."""
    mag = np.linalg.norm(flow, axis=-1)
    norm = np.clip((mag / max(max_speed, 1e-4)) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)


def render_optical_flow_video(
    scene_dir: Path,
    out_path: Path,
    cam_name: str = "cam00",
    num_frames: int = 50,
    stride: int = 2,
    fps: int = 15,
    backend: str = "farneback",
    max_speed: float = 4.0,
):
    """Compute and render optical flow across all frames of `cam_name` as an MP4 video."""
    first_img_path = scene_dir / "colmap_0" / "images" / f"{cam_name}.png"
    if not first_img_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_img_path}")

    sample = cv2.imread(str(first_img_path))
    H, W = sample.shape[:2]

    # 3-panel layout: [RGB Frame | Flow Direction (HSV wheel) | Flow Magnitude Heatmap]
    out_w = W * 3
    out_h = H

    raw_mp4 = out_path.with_name(out_path.stem + "_raw.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_mp4), fourcc, float(fps), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {raw_mp4}")

    print(f"Loading {num_frames} frames for {cam_name}...")
    frames_bgr = [
        cv2.imread(str(scene_dir / f"colmap_{t}" / "images" / f"{cam_name}.png"))
        for t in range(num_frames)
    ]
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    print(f"Initializing optical flow estimator (backend={backend}, fb_consistency=True)...")
    flow_estimator = PlugAndPlayOpticalFlow(backend=backend, fb_consistency=True)

    print("Computing flow and writing video panels...")
    for t in range(num_frames):
        t_other = min(t + stride, num_frames - 1) if t + stride < num_frames else max(0, t - stride)
        flow = flow_estimator.compute_flow(frames_rgb[t], frames_rgb[t_other])
        flow_mag = np.linalg.norm(flow, axis=-1)
        max_f = float(flow_mag.max())
        mean_f = float(flow_mag.mean())

        # Panel 1: Original RGB Frame
        p1 = frames_bgr[t].copy()
        cv2.rectangle(p1, (8, 8), (8 + 320, 36), (0, 0, 0), -1)
        cv2.putText(
            p1, f"{cam_name} | Frame {t:02d}/{num_frames - 1:02d} (t->t+{stride})",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Panel 2: Flow Direction & Magnitude (HSV color wheel)
        p2 = flow_to_hsv_bgr(flow, max_speed=max_speed)
        cv2.rectangle(p2, (8, 8), (8 + 320, 36), (0, 0, 0), -1)
        cv2.putText(
            p2, "Flow Direction (HSV Angle) & Velocity",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Panel 3: Magnitude Heatmap (Magma)
        p3 = flow_to_magma_bgr(flow, max_speed=max_speed)
        cv2.rectangle(p3, (8, 8), (8 + 320, 36), (0, 0, 0), -1)
        cv2.putText(
            p3, f"Magnitude ||u||_2 (Max: {max_f:.2f} px, Mean: {mean_f:.2f})",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )

        combined = np.hstack([p1, p2, p3])
        writer.write(combined)

    writer.release()
    print(f"Saved raw video to {raw_mp4}")

    # Convert to standard browser-compatible H.264 MP4 using ffmpeg
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_mp4),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_mp4.unlink(missing_ok=True)
        print(f"Converted browser-ready H.264 video saved to: {out_path}")
    except Exception as e:
        print(f"Warning: ffmpeg conversion failed ({e}), kept raw file at {raw_mp4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render dense optical flow across all frames into a video.")
    parser.add_argument("--cam", type=str, default="cam00", help="Camera name (e.g. cam00, cam01, ...)")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--stride", type=int, default=2, help="Temporal stride dt for flow (matches pipeline default dt=2)")
    parser.add_argument("--fps", type=int, default=15, help="Video framerate")
    parser.add_argument("--backend", type=str, default="farneback", choices=["farneback", "dis", "raft_small", "raft_large"])
    parser.add_argument("--max_speed", type=float, default=4.0, help="Max pixel speed for color normalization")
    parser.add_argument("--out", type=str, default=None, help="Custom output path")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    scene_dir = workspace / "data" / "neural_3d_video" / "scenes" / "coffee_martini"
    out_path = Path(args.out) if args.out else workspace / "results" / f"optical_flow_{args.cam}.mp4"

    render_optical_flow_video(
        scene_dir=scene_dir,
        out_path=out_path,
        cam_name=args.cam,
        num_frames=args.num_frames,
        stride=args.stride,
        fps=args.fps,
        backend=args.backend,
        max_speed=args.max_speed,
    )
