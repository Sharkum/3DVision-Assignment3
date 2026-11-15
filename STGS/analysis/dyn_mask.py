# [A3] Report item 2: dynamic-region mask from optical flow on the GROUND-TRUTH video (method-independent).
# Definition (per camera, one static mask for all 50 frames and for every k / method):
#   1) Farneback optical flow between consecutive GT frames t -> t+1, t = 0..48 (676x507 grayscale)
#   2) M = max_t |F_t| > TAU pixels/frame   (default TAU = 1.0 px)
#   3) clean-up: morphological opening (OPEN px ellipse) removes isolated flow-noise pixels,
#      then dilation (DILATE px ellipse) so the mask covers the full extent of moving objects/edges.
# Outputs: results/masks/<cam>.png (255 = dynamic), results/masks/overlay_<cam>.jpg (GT frame 25 + red mask),
#          results/masks/mask_meta.json (definition, parameters, dynamic-pixel fraction per camera).
# Usage: python analysis/dyn_mask.py [--tau 1.0 --open 5 --dilate 9]
import os, sys, json, argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3common as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.DEFAULT_BASE)
    ap.add_argument("--tau", type=float, default=1.0, help="flow-magnitude threshold, px/frame")
    ap.add_argument("--open", type=int, default=5, help="opening kernel size (px), 0 = off")
    ap.add_argument("--dilate", type=int, default=9, help="dilation kernel size (px), 0 = off")
    a = ap.parse_args()

    P = C.paths(a.base)
    os.makedirs(P["masks"], exist_ok=True)
    meta = dict(definition="per camera: max over t=0..48 of |Farneback flow(GT_t -> GT_t+1)| > tau, "
                           "then morphological opening and dilation; same mask for all k and methods",
                tau_px=a.tau, open_px=a.open, dilate_px=a.dilate, farneback=C.FARNEBACK,
                frames=C.DURATION, cams={})
    for cam in C.cameras(a.base):
        grays = C.read_gt_gray_stack(a.base, cam)
        mags = C.flow_magnitudes(grays)
        maxmag = np.max(np.stack(mags, 0), 0)
        m = (maxmag > a.tau).astype(np.uint8)
        if a.open > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.open, a.open)))
        if a.dilate > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.dilate, a.dilate)))
        cv2.imwrite(os.path.join(P["masks"], "{}.png".format(cam)), m * 255)
        # visual check
        bgr = cv2.imread(C.gt_path(a.base, cam, C.DURATION // 2))
        ov = bgr.copy()
        ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(P["masks"], "overlay_{}.jpg".format(cam)), ov)
        mean_motion = [float(x[m > 0].mean()) if m.any() else 0.0 for x in mags]
        meta["cams"][cam] = dict(dynamic_fraction=float(m.mean()),
                                 max_flow_px=float(maxmag.max()),
                                 mean_flow_in_mask_per_step=mean_motion)
        print("{}: dynamic fraction {:.2%}, max flow {:.1f}px".format(cam, m.mean(), maxmag.max()), flush=True)
    with open(os.path.join(P["masks"], "mask_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("MASKS DONE ->", P["masks"])
