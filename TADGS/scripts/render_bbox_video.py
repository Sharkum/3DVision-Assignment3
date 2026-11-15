import argparse
from pathlib import Path
import cv2
import numpy as np


def render_bbox_video(
    scene_dir: Path,
    mask_dir: Path,
    out_path: Path,
    cam_name: str = "cam00",
    num_frames: int = 50,
    fps: int = 15,
    dim_outside: bool = True,
    side_by_side_mask: bool = False,
):
    """Overlay the dynamic-region bounding box onto all frames of `cam_name` and save as an MP4 video."""
    first_img_path = scene_dir / "colmap_0" / "images" / f"{cam_name}.png"
    if not first_img_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_img_path}")

    sample = cv2.imread(str(first_img_path))
    H, W = sample.shape[:2]
    out_w = W * 2 if side_by_side_mask else W
    out_h = H

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    for t in range(num_frames):
        img_path = scene_dir / f"colmap_{t}" / "images" / f"{cam_name}.png"
        mask_path = mask_dir / f"mask_{cam_name}_frame{t:04d}.png"

        frame = cv2.imread(str(img_path))
        if frame is None:
            raise FileNotFoundError(f"Missing frame: {img_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        vis = frame.copy()

        bbox_text = "No dynamic bbox"
        if mask is not None and np.any(mask > 127):
            ys, xs = np.where(mask > 127)
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1

            if dim_outside:
                dimmed = (vis.astype(np.float32) * 0.45).astype(np.uint8)
                dimmed[y0:y1, x0:x1] = vis[y0:y1, x0:x1]
                vis = dimmed

            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (136, 255, 0), 2)
            bbox_text = f"BBox: {x1 - x0}x{y1 - y0} [{x0}:{x1}, {y0}:{y1}]"
        elif dim_outside:
            vis = (vis.astype(np.float32) * 0.45).astype(np.uint8)

        # HUD label at top-left
        hud = f"{cam_name} | Frame {t:02d}/{num_frames - 1:02d} | {bbox_text}"
        cv2.rectangle(vis, (8, 8), (8 + 470, 34), (0, 0, 0), -1)
        cv2.putText(vis, hud, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        if side_by_side_mask:
            if mask is None:
                mask_vis = np.zeros_like(frame)
            else:
                mask_vis = cv2.cvtColor(((mask > 127).astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
                if np.any(mask > 127):
                    cv2.rectangle(mask_vis, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
            cv2.putText(mask_vis, "Binary BBox Mask", (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 136), 1, cv2.LINE_AA)
            out_frame = np.hstack([vis, mask_vis])
        else:
            out_frame = vis

        writer.write(out_frame)

    writer.release()
    print(f"Saved dynamic bbox video ({num_frames} frames @ {fps} fps) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an MP4 video showing the dynamic-region bounding box across all frames.")
    parser.add_argument("--cam", type=str, default="cam00", help="Camera name (e.g. cam00, cam01, ...)")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of consecutive frames (0..N-1)")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS")
    parser.add_argument("--no_dim", action="store_true", help="Do not dim pixels outside the bounding box")
    parser.add_argument("--side_by_side", action="store_true", help="Include binary mask side-by-side")
    parser.add_argument("--out", type=str, default=None, help="Optional custom output MP4 path")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    scene_dir = workspace / "data" / "neural_3d_video" / "scenes" / "coffee_martini"
    mask_dir = workspace / "data" / "neural_3d_video" / "flow_masks" / "coffee_martini" / "bbox_masks"
    out_path = Path(args.out) if args.out else workspace / "results" / f"dynamic_bbox_{args.cam}.mp4"

    render_bbox_video(
        scene_dir=scene_dir,
        mask_dir=mask_dir,
        out_path=out_path,
        cam_name=args.cam,
        num_frames=args.num_frames,
        fps=args.fps,
        dim_outside=not args.no_dim,
        side_by_side_mask=args.side_by_side,
    )
