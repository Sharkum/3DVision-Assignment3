import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def compute_camera_dynamic_masks(scene_dir: Path, out_dir: Path, num_frames: int = 50,
                                 tau_flow: float = 0.75, tau_std: float = 0.06):
    """Compute dynamic-region binary masks for all cameras (cam00..cam17) and frames (0..49)
    using dense Farneback optical flow + temporal intensity standard deviation on the 50 GT frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_files = sorted((scene_dir / "colmap_0" / "images").glob("cam*.png"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    total_dyn_ratios = []
    for cam_path in tqdm(cam_files, desc="Generating optical-flow dynamic masks"):
        cam_name = cam_path.stem  # e.g., cam00
        cam_id = int(cam_name.replace("cam", ""))

        grays = []
        rgbs = []
        for t in range(num_frames):
            img_p = scene_dir / f"colmap_{t}" / "images" / f"{cam_name}.png"
            bgr = cv2.imread(str(img_p))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
            rgbs.append(bgr.astype(np.float32) / 255.0)

        rgbs_arr = np.stack(rgbs, axis=0)  # (T, H, W, 3)
        # Temporal intensity standard deviation across the 50 frames
        temp_std = np.std(rgbs_arr, axis=0).mean(axis=-1)  # (H, W)

        # Compute forward/backward 1-step and 4-step dense optical flow magnitudes
        flow_mags = np.zeros((num_frames, grays[0].shape[0], grays[0].shape[1]), dtype=np.float32)
        for t in range(num_frames):
            t_next1 = min(t + 1, num_frames - 1)
            t_prev1 = max(t - 1, 0)
            t_next4 = min(t + 4, num_frames - 1)
            t_prev4 = max(t - 4, 0)

            mag_list = []
            if t_next1 != t:
                fl1 = cv2.calcOpticalFlowFarneback(grays[t], grays[t_next1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag_list.append(np.linalg.norm(fl1, axis=-1))
            if t_prev1 != t:
                fl2 = cv2.calcOpticalFlowFarneback(grays[t], grays[t_prev1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag_list.append(np.linalg.norm(fl2, axis=-1))
            if t_next4 != t:
                fl4 = cv2.calcOpticalFlowFarneback(grays[t], grays[t_next4], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag_list.append(np.linalg.norm(fl4, axis=-1) / float(t_next4 - t))
            if t_prev4 != t:
                flp4 = cv2.calcOpticalFlowFarneback(grays[t], grays[t_prev4], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag_list.append(np.linalg.norm(flp4, axis=-1) / float(t - t_prev4))

            flow_mags[t] = np.max(np.stack(mag_list, axis=0), axis=0)

        for t in range(num_frames):
            raw_mask = ((flow_mags[t] > tau_flow) | (temp_std > tau_std)).astype(np.uint8) * 255
            clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
            clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)
            total_dyn_ratios.append((clean_mask > 127).mean())

            # Save standard binary mask (255 = dynamic region, 0 = static background)
            mask_fname = out_dir / f"mask_{cam_name}_frame{t:04d}.png"
            cv2.imwrite(str(mask_fname), clean_mask)

            # Also save TAD-GS legacy format (0 = dynamic, 255 = static) for compatibility
            legacy_fname = out_dir / f"{cam_id:04d}_{t - 5:04d}__{cam_id:04d}_{t:04d}.png"
            cv2.imwrite(str(legacy_fname), 255 - clean_mask)

    print(f"Saved dynamic masks to {out_dir}. Mean dynamic pixel fraction: {np.mean(total_dyn_ratios)*100:.2f}%")


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parent.parent
    scene_dir = workspace / "data" / "neural_3d_video" / "scenes" / "coffee_martini"
    out_dir = workspace / "data" / "neural_3d_video" / "flow_masks" / "coffee_martini" / "valid_masks"
    compute_camera_dynamic_masks(scene_dir, out_dir)

