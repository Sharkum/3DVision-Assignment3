"""Uniform visual comparison (Deliverable 4) for TAD-GS, STGS and 4DGS.

For each k in {2, 4, 8} the test frame is the one with the most ground-truth motion (STGS rule):
k=2 -> t=1, k=4 -> t=2, k=8 -> t=4, held-out camera cam00, 676x507.
Every method gets the same frame, the same 160x160 crop (centred on the ground-truth flow peak inside the
dynamic-region mask) and the same error map: mean |render - GT| over RGB, colour scale 0..0.25 (magma).

Inputs (any missing method is skipped with a message):
  GT      : TADGS/data/neural_3d_video/scenes/coffee_martini/colmap_{t}/images/cam00.png
  TAD-GS  : TADGS/results/renders_k{k}_cam00_all50/renders/cam00_frame{t:04d}.png
  STGS    : STGS/results/renders/render_k{k}_cam00_f{t:03d}.png
  4DGS    : 4DGaussians/low_resolution_work_v2/renders/cam00_k{k}_frame{t:04d}_render.png (rendered at 676x507)
Output: deliverables/figures/visual_uniform_k{k}.jpg and visual_uniform_crops.jpg
"""
import glob
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "deliverables/figures"
SCENE = ROOT / "TADGS/data/neural_3d_video/scenes/coffee_martini"
MASK = ROOT / "TADGS/data/neural_3d_video/flow_masks/coffee_martini/otsu_bbox_masks/mask_cam00_frame0020.png"
FRAMES = {2: 1, 4: 2, 8: 4}
CROP = 160
EMAX = 0.25
FOURDGS_GLOBS = [
    "4DGaussians/low_resolution_work_v2/renders/cam00_k{k}_frame{t:04d}_render.png",
]


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def find_render(method, k, t):
    if method == "TAD-GS":
        p = ROOT / f"TADGS/results/renders_k{k}_cam00_all50/renders/cam00_frame{t:04d}.png"
        return p if p.exists() else None
    if method == "STGS":
        p = ROOT / f"STGS/results/renders/render_k{k}_cam00_f{t:03d}.png"
        return p if p.exists() else None
    for g in FOURDGS_GLOBS:
        hits = sorted(glob.glob(str(ROOT / g.format(k=k, t=t)), recursive=True))
        hits = [h for h in hits if "gt" not in Path(h).name.lower()]
        if hits:
            return Path(hits[0])
    return None


def crop_box(t):
    """160x160 box centred on the ground-truth flow peak (frame t -> t+1) inside the dynamic mask."""
    a = cv2.imread(str(SCENE / f"colmap_{t}/images/cam00.png"), 0)
    b = cv2.imread(str(SCENE / f"colmap_{t + 1}/images/cam00.png"), 0)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    mag[cv2.imread(str(MASK), 0) <= 127] = 0
    mag = cv2.GaussianBlur(mag, (0, 0), 15)
    cy, cx = np.unravel_index(np.argmax(mag), mag.shape)
    H, W = mag.shape
    x0 = int(np.clip(cx - CROP // 2, 0, W - CROP))
    y0 = int(np.clip(cy - CROP // 2, 0, H - CROP))
    return x0, y0


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


METHODS = ["TAD-GS", "STGS", "4DGS"]
crops = {}
for k, t in FRAMES.items():
    gt = rgb(SCENE / f"colmap_{t}/images/cam00.png")
    x0, y0 = crop_box(t)
    rows = []
    for m in METHODS:
        p = find_render(m, k, t)
        if p is None:
            print(f"[skip] {m} k={k} t={t}: no 676x507 render found")
            continue
        r = rgb(p)
        if r.shape != gt.shape:
            print(f"[skip] {m} k={k}: render is {r.shape[1]}x{r.shape[0]}, expected 676x507 ({p})")
            continue
        rows.append((m, r))
    crops[k] = (gt, x0, y0, rows)
    n = len(rows)
    fig, ax = plt.subplots(n, 5, figsize=(17, 2.75 * n + 0.4),
                           gridspec_kw=dict(width_ratios=[1.33, 1.33, 1, 1, 1]), squeeze=False)
    for i, (m, r) in enumerate(rows):
        err = np.abs(r - gt).mean(axis=2)
        s = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        panels = [(r, f"{m} render  ({psnr(r, gt):.2f} dB)"),
                  (err, "|render - GT|"),
                  (gt[s], "GT crop"),
                  (r[s], f"{m} crop  ({psnr(r[s], gt[s]):.2f} dB)"),
                  (err[s], "|error| crop")]
        for j, (img, title) in enumerate(panels):
            a = ax[i, j]
            if img.ndim == 2:
                im = a.imshow(img, cmap="magma", vmin=0, vmax=EMAX)
            else:
                a.imshow(np.clip(img, 0, 1))
            if j < 2:
                a.add_patch(patches.Rectangle((x0, y0), CROP, CROP, fill=False, ec="#00ff66", lw=1.2))
            a.set_title(title, fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
    fig.subplots_adjust(right=0.92)
    cb = fig.colorbar(im, cax=fig.add_axes([0.935, 0.2, 0.008, 0.6]))
    cb.set_label("|error| (0 to 0.25)", fontsize=8)
    fig.suptitle(f"k = {k}, held-out cam00, test frame t = {t} (676x507). "
                 f"Same frame, crop and error scale for all methods.", fontsize=11)
    fig.savefig(FIG / f"visual_uniform_k{k}.jpg", dpi=150, bbox_inches="tight", pil_kwargs={"quality": 90})
    plt.close(fig)
    print(f"wrote visual_uniform_k{k}.jpg  ({', '.join(m for m, _ in rows)}); crop x0={x0} y0={y0}")

# compact overview: rows = k, columns = GT crop + (render crop, error crop) per method
avail = [m for m in METHODS if all(any(mm == m for mm, _ in crops[k][3]) for k in FRAMES)]
fig, ax = plt.subplots(3, 1 + 2 * len(avail), figsize=(2.3 * (1 + 2 * len(avail)), 7.3), squeeze=False)
for i, k in enumerate(FRAMES):
    gt, x0, y0, rows = crops[k]
    s = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    rr = dict(rows)
    ax[i, 0].imshow(gt[s])
    ax[i, 0].set_ylabel(f"k = {k}, t = {FRAMES[k]}", fontsize=10)
    if i == 0:
        ax[i, 0].set_title("ground truth", fontsize=9)
    for j, m in enumerate(avail):
        r = rr[m]
        ax[i, 1 + 2 * j].imshow(np.clip(r[s], 0, 1))
        ax[i, 2 + 2 * j].imshow(np.abs(r - gt).mean(axis=2)[s], cmap="magma", vmin=0, vmax=EMAX)
        ax[i, 1 + 2 * j].set_xlabel(f"{psnr(r[s], gt[s]):.2f} dB", fontsize=8)
        if i == 0:
            ax[i, 1 + 2 * j].set_title(f"{m}", fontsize=9)
            ax[i, 2 + 2 * j].set_title(f"{m} |error|", fontsize=9)
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("Fast-moving region (160x160 crop at the flow peak), held-out cam00, |error| scale 0 to 0.25",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "visual_uniform_crops.jpg", dpi=170, pil_kwargs={"quality": 90})
plt.close(fig)
print("wrote visual_uniform_crops.jpg with", avail)
