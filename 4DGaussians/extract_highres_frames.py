#!/usr/bin/env python3
"""Extract high-resolution (2704x2028) frames from MP4 videos into the same
directory structure as the low-res dataset."""

import cv2
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path("/home/val_gx10/Desktop/rohan/asmt2/coffee_martini")
DST_DIR = Path("/home/val_gx10/Desktop/rohan/asmt2/coffee_martini_highres")
NUM_FRAMES = 50


def main():
    mp4s = sorted(SRC_DIR.glob("cam*.mp4"))
    print(f"Found {len(mp4s)} MP4 files")

    for mp4 in tqdm(mp4s, desc="Cameras"):
        cam_name = mp4.stem
        out_dir = DST_DIR / cam_name / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(mp4))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {cam_name}: {w}x{h}, {total} frames in MP4, extracting first {NUM_FRAMES}")

        for idx in range(NUM_FRAMES):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from {mp4}")
            out_path = out_dir / f"{idx:04d}.png"
            cv2.imwrite(str(out_path), frame)

        cap.release()

    print(f"\nDone. High-res frames saved to {DST_DIR}")


if __name__ == "__main__":
    main()
