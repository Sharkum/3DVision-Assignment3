# [A3] Shared paths/helpers for the analysis scripts (python 3.7 compatible).
# Layout (BASE = assignment root, default = parent of this folder):
#   GT frames : BASE/data/coffee_martini/colmap_<f>/images/<cam>.png   (676x507, frames 0..49)
#   renders   : BASE/log/stg_full_ds4_k<k>/eval_30000/renders/<cam>/<fff>.png  (from eval_interp.py)
#   outputs   : BASE/results/
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.dirname(HERE)
DURATION = 50
HELDOUT = "cam00"
KS = [2, 4, 8]


def paths(base):
    return dict(
        data=os.path.join(base, "data", "coffee_martini"),
        log=os.path.join(base, "log"),
        results=os.path.join(base, "results"),
        masks=os.path.join(base, "results", "masks"),
    )


def model_dir(base, k, iteration=30000):
    return os.path.join(base, "log", "stg_full_ds4_k{}".format(k), "eval_{}".format(iteration))


def cameras(base):
    d = os.path.join(paths(base)["data"], "colmap_0", "images")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".png"))


def train_frames_for(k, duration=DURATION):
    return list(range(0, duration, k))


def test_frames_for(k, duration=DURATION):
    # same definition as eval_interp.py: midpoints k/2, 3k/2, ... strictly between trained frames
    last = max(range(0, duration, k))
    return list(range(k // 2, last, k))


def time_split(frame, k):
    if frame % k == 0:
        return "train"
    return "test" if frame in test_frames_for(k) else "other"


def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def gt_path(base, cam, frame):
    return os.path.join(paths(base)["data"], "colmap_{}".format(frame), "images", "{}.png".format(cam))


def render_path(base, k, cam, frame, iteration=30000):
    return os.path.join(model_dir(base, k, iteration), "renders", cam, "{:03d}.png".format(frame))


def read_gt_gray_stack(base, cam, duration=DURATION):
    return [cv2.cvtColor(cv2.imread(gt_path(base, cam, f)), cv2.COLOR_BGR2GRAY) for f in range(duration)]


# Optical flow (Farneback, OpenCV) between consecutive GT frames. Same parameters everywhere.
FARNEBACK = dict(pyr_scale=0.5, levels=4, winsize=21, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)


def flow_magnitudes(grays):
    """returns list of len(grays)-1 arrays |F_{t->t+1}| in pixels (float32, HxW)"""
    mags = []
    for a, b in zip(grays[:-1], grays[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, **FARNEBACK)
        mags.append(np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2).astype(np.float32))
    return mags


def motion_at_frame(mags, f):
    """per-pixel motion at frame f = mean of |F_{f-1->f}| and |F_{f->f+1}| (one-sided at the ends)"""
    parts = []
    if f - 1 >= 0:
        parts.append(mags[f - 1])
    if f < len(mags):
        parts.append(mags[f])
    return sum(parts) / len(parts)


def load_mask(base, cam):
    m = cv2.imread(os.path.join(paths(base)["masks"], "{}.png".format(cam)), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError("mask for {} missing: run analysis/dyn_mask.py first".format(cam))
    return (m > 127)
