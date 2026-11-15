# [A3] Report item 4: for each k, a TEST frame shown as  render | ground truth | error map, full frame + zoom on a
# fast-moving region. OpenCV only (runs on server env stg_a3 or the PC).
# Choice (automatic, from GT optical flow only, so not biased towards good/bad renders):
#   frame  = test frame of this k with the largest mean flow magnitude inside the dynamic mask of that camera
#   region = CROPxCROP window centred on the peak of the (Gaussian-blurred, masked) flow magnitude at that frame
# Error map = mean over RGB of |render - GT|, colour map INFERNO, fixed scale 0..EMAX (same for every k / camera).
# Outputs in results/visuals/:
#   k<k>_<cam>_f<fff>_panel.png      : 2 rows (full frames with crop box / zoomed crops) x 3 cols
#   k<k>_<cam>_f<fff>_{render,gt,err}_crop.png : the individual crops (for custom figures)
#   zoom_<cam>_allk.png              : rows k=2,4,8, columns render | GT | error (crops) - compact report figure
#   choices.json                     : chosen frame / crop box / crop PSNR per k and camera
# Usage: python analysis/visuals.py [--cams cam00 cam01] [--crop 160] [--emax 0.25]
import os, sys, json, argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3common as C

FONT = cv2.FONT_HERSHEY_SIMPLEX


def err_map(r, g, emax):
    e = np.abs(r.astype(np.float32) - g.astype(np.float32)).mean(2) / 255.0
    e8 = np.clip(e / emax * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(e8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) / 255.0 - b.astype(np.float64) / 255.0) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def label(img, text, h=28):
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, h - 9), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def colorbar(width, emax, h=22):
    grad = np.tile(np.linspace(0, 255, width).astype(np.uint8), (h, 1))
    cb = cv2.cvtColor(cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    bar = np.full((h, width, 3), 255, np.uint8)
    cv2.putText(bar, "0", (2, 16), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    t = ">= {:.2f} |err|".format(emax)
    cv2.putText(bar, t, (width - 9 * len(t), 16), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([cb, bar])


def fit(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def save_rgb(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=C.DEFAULT_BASE)
    ap.add_argument("--ks", type=int, nargs="+", default=C.KS)
    ap.add_argument("--cams", nargs="+", default=["cam00", "cam01"], help="cam00 = held-out; others = training cams")
    ap.add_argument("--crop", type=int, default=160, help="crop size in px (image is 676x507)")
    ap.add_argument("--emax", type=float, default=0.25, help="error-map colour scale max (|err| in [0,1])")
    ap.add_argument("--iteration", type=int, default=30000)
    a = ap.parse_args()
    P = C.paths(a.base)
    out = os.path.join(P["results"], "visuals")
    os.makedirs(out, exist_ok=True)
    choices = {}

    for cam in a.cams:
        mask = C.load_mask(a.base, cam).astype(np.float32)
        mags = C.flow_magnitudes(C.read_gt_gray_stack(a.base, cam))
        zoom_rows = []
        for k in a.ks:
            tf = C.test_frames_for(k)
            score = [float((C.motion_at_frame(mags, f) * mask).sum() / max(mask.sum(), 1)) for f in tf]
            f = tf[int(np.argmax(score))]
            mot = cv2.GaussianBlur(C.motion_at_frame(mags, f) * mask, (0, 0), 15)
            cy, cx = np.unravel_index(int(np.argmax(mot)), mot.shape)
            H, W = mot.shape
            s = min(a.crop, H, W)
            y0 = int(np.clip(cy - s // 2, 0, H - s))
            x0 = int(np.clip(cx - s // 2, 0, W - s))

            g = C.read_rgb(C.gt_path(a.base, cam, f))
            r = C.read_rgb(C.render_path(a.base, k, cam, f, a.iteration))
            e = err_map(r, g, a.emax)
            crop = lambda im: im[y0:y0 + s, x0:x0 + s].copy()
            rc, gc, ec = crop(r), crop(g), crop(e)
            p_full, p_crop = psnr(r, g), psnr(rc, gc)
            tag = "k{}_{}_f{:03d}".format(k, cam, f)
            for nm, im in [("render", rc), ("gt", gc), ("err", ec)]:
                save_rgb(os.path.join(out, "{}_{}_crop.png".format(tag, nm)), im)

            boxed = []
            for im in (r, g, e):
                im = im.copy()
                cv2.rectangle(im, (x0, y0), (x0 + s - 1, y0 + s - 1), (0, 255, 0), 2)
                boxed.append(im)
            who = "held-out cam" if cam == C.HELDOUT else "training cam"
            row1 = np.hstack([label(boxed[0], "STGS render (k={}, {} {}, test frame {})".format(k, who, cam, f)),
                              label(boxed[1], "ground truth"),
                              label(boxed[2], "|error|  PSNR full {:.2f} dB".format(p_full))])
            Z = W  # zoomed crops shown as WxW squares
            row2 = np.hstack([label(fit(rc, Z), "zoom: render"), label(fit(gc, Z), "zoom: ground truth"),
                              label(fit(ec, Z), "zoom: |error|  PSNR crop {:.2f} dB".format(p_crop))])
            cb = np.hstack([np.full((44, 2 * W, 3), 255, np.uint8), colorbar(W, a.emax)])
            save_rgb(os.path.join(out, tag + "_panel.png"), np.vstack([row1, row2, cb]))

            zs = 256
            zoom_rows.append(np.hstack([label(fit(rc, zs), "k={} f={} render".format(k, f)),
                                        label(fit(gc, zs), "ground truth"),
                                        label(fit(ec, zs), "|err| {:.2f} dB".format(p_crop))]))
            choices["{}_k{}".format(cam, k)] = dict(cam=cam, k=k, frame=f, crop_xywh=[x0, y0, s, s],
                                                    mean_motion_in_mask_px=max(score), psnr_full=p_full, psnr_crop=p_crop)
            print(tag, "crop", (x0, y0, s), "PSNR full {:.2f} crop {:.2f}".format(p_full, p_crop), flush=True)
        grid = np.vstack(zoom_rows + [colorbar(zoom_rows[0].shape[1], a.emax)])
        save_rgb(os.path.join(out, "zoom_{}_allk.png".format(cam)), grid)

    with open(os.path.join(out, "choices.json"), "w") as fh:
        json.dump(dict(emax=a.emax, crop=a.crop, rule="frame = test frame with max mean GT-flow in mask; "
                       "crop centred on blurred masked GT-flow peak", choices=choices), fh, indent=1)
    print("VISUALS DONE ->", out)
