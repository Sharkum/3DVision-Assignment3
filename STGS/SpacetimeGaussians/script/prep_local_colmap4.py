# [A3] LOCAL (PC, CPU) preprocessing for coffee_martini with COLMAP 4.x.
# Replaces script/pre_n3d.py (which uses the COLMAP-3.x db schema, symlinks and `rm -r`).
# Same steps/outputs as the authors' pipeline:
#   1) frames 0..49 of every camXX.mp4, 4x downsample with cv2.INTER_AREA (2704x2028 -> 676x507)
#   2) GT poses/intrinsics from poses_bounds.npy via the authors' posetow2c_matrcs/rotmat2qvec (focal/4, cx=W//2, cy=H//2)
#   3) per frame: COLMAP feature_extractor -> exhaustive_matcher -> point_triangulator with FIXED GT poses
#   4) write <out>/colmap_<f>/images/camXX.png and sparse/0/{cameras.txt, images.txt, points3D.bin}
# The input scene folder is only READ. COLMAP runs in a local work dir (sqlite on SSHFS is unsafe); results are copied to --out.
# Needs: python3 with numpy + opencv-python, and `colmap` on PATH.
# Example (PowerShell, from the SpacetimeGaussians folder):
#   python script\prep_local_colmap4.py --scene Z:\sohamg\assigment3\coffee_martini --out Z:\sohamg\assigment3\data\coffee_martini --work C:\Users\ADMIN\stg_prep\coffee_martini
import argparse
import importlib.util
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("my_utils", HERE.parent / "thirdparty/gaussian_splatting/utils/my_utils.py")
my_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(my_utils)  # authors' numpy-only pose helpers


def run(cmd):
    cmd = [str(c) for c in cmd]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def extract_frames(videos, work, start, end, ds):
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for f in range(start, end):
            dst = work / f"colmap_{f}" / "images" / f"{v.stem}.png"
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {f} of {v}")
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if ds > 1:
                frame = cv2.resize(frame, (int(frame.shape[1] / ds), int(frame.shape[0] / ds)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), frame)
        cap.release()
        print("extracted", v.name, flush=True)


def gt_cameras(scene, videos, ds):
    pb = np.load(scene / "poses_bounds.npy")
    assert pb.shape[0] == len(videos), f"poses_bounds rows {pb.shape[0]} != videos {len(videos)}"
    poses = pb[:, :15].reshape(-1, 3, 5)
    w2c = my_utils.posetow2c_matrcs(poses.copy().transpose(1, 2, 0))
    cams = {}
    for i, v in enumerate(videos):  # authors: row i <-> i-th video sorted by name
        H, W, focal = poses[i, :, -1] / ds
        W, H = int(round(W)), int(round(H))
        cams[f"{v.stem}.png"] = dict(w=W, h=H, fx=focal, fy=focal, cx=float(W // 2), cy=float(H // 2),
                                     q=my_utils.rotmat2qvec(w2c[i][:3, :3]), t=w2c[i][:3, 3])
    return cams


def write_model_txt(folder, cams, ids):
    folder.mkdir(parents=True, exist_ok=True)
    camlines, imlines = [], []
    for name, (image_id, camera_id) in sorted(ids.items(), key=lambda x: x[1][0]):
        c = cams[name]
        camlines.append(f"{camera_id} PINHOLE {c['w']} {c['h']} {c['fx']} {c['fy']} {c['cx']} {c['cy']}\n")
        imlines.append(f"{image_id} " + " ".join(map(str, c["q"])) + " " + " ".join(map(str, c["t"])) + f" {camera_id} {name}\n\n")
    (folder / "cameras.txt").write_text("".join(camlines))
    (folder / "images.txt").write_text("".join(imlines))
    (folder / "points3D.txt").write_text("")


def points_txt_to_bin(txt, binpath):
    pts = []
    for line in Path(txt).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        e = line.split()
        pts.append((int(e[0]), float(e[1]), float(e[2]), float(e[3]), int(e[4]), int(e[5]), int(e[6]), float(e[7])))
    with open(binpath, "wb") as f:  # layout read by colmap_loader.read_points3D_binary (track length 0)
        f.write(struct.pack("<Q", len(pts)))
        for p in pts:
            f.write(struct.pack("<QdddBBBd", *p))
            f.write(struct.pack("<Q", 0))
    return len(pts)


def colmap_frame(colmap, fdir, cams, tri_flags):
    db, imgs = fdir / "database.db", fdir / "images"
    for p in [db, fdir / "manual", fdir / "tri", fdir / "tri_txt", fdir / "sparse"]:
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    run([colmap, "feature_extractor", "--database_path", db, "--image_path", imgs,
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "0"])
    # put GT intrinsics in the db (used by matcher geometric verification) and read COLMAP's ids
    con = sqlite3.connect(str(db))
    ids = {}
    for image_id, name, camera_id in con.execute("SELECT image_id, name, camera_id FROM images").fetchall():
        c = cams[name]
        ids[name] = (image_id, camera_id)
        con.execute("UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                    (c["w"], c["h"], np.array([c["fx"], c["fy"], c["cx"], c["cy"]], np.float64).tobytes(), camera_id))
    con.commit()
    con.close()
    assert len(ids) == len(cams), f"db has {len(ids)} images, expected {len(cams)}"
    run([colmap, "exhaustive_matcher", "--database_path", db])
    write_model_txt(fdir / "manual", cams, ids)
    (fdir / "tri").mkdir()
    run([colmap, "point_triangulator", "--database_path", db, "--image_path", imgs,
         "--input_path", fdir / "manual", "--output_path", fdir / "tri"] + tri_flags)
    (fdir / "tri_txt").mkdir()
    run([colmap, "model_converter", "--input_path", fdir / "tri", "--output_path", fdir / "tri_txt", "--output_type", "TXT"])
    sp = fdir / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fdir / "manual" / "cameras.txt", sp / "cameras.txt")  # GT poses (triangulator keeps poses fixed)
    shutil.copy2(fdir / "manual" / "images.txt", sp / "images.txt")
    return points_txt_to_bin(fdir / "tri_txt" / "points3D.txt", sp / "points3D.bin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="original scene dir (read-only): camXX.mp4 + poses_bounds.npy")
    ap.add_argument("--out", required=True, help="output dir used for training (on Z:)")
    ap.add_argument("--work", required=True, help="local scratch dir for COLMAP (not on SSHFS)")
    ap.add_argument("--startframe", type=int, default=0)
    ap.add_argument("--endframe", type=int, default=50)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--colmap", default="colmap")
    a = ap.parse_args()

    scene, out, work = Path(a.scene), Path(a.out), Path(a.work)
    assert scene.resolve() != out.resolve(), "--out must differ from --scene (original data must not be altered)"
    colmap = shutil.which(a.colmap) or a.colmap
    print(subprocess.run([colmap, "--version"], capture_output=True, text=True).stdout.strip())
    helptxt = subprocess.run([colmap, "point_triangulator", "-h"], capture_output=True, text=True).stdout
    tri_flags = ["--Mapper.ba_global_function_tolerance=0.000001"] if "ba_global_function_tolerance" in helptxt else []
    print("point_triangulator extra flags (authors' setting, if supported):", tri_flags)

    videos = sorted(scene.glob("cam*.mp4"))
    print("videos:", [v.stem for v in videos])
    cams = gt_cameras(scene, videos, a.downscale)
    extract_frames(videos, work, a.startframe, a.endframe, a.downscale)

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene / "poses_bounds.npy", out / "poses_bounds.npy")  # loader reads <out>/poses_bounds.npy (near/far)
    for f in range(a.startframe, a.endframe):
        dst = out / f"colmap_{f}"
        if (dst / "sparse" / "0" / "points3D.bin").exists():
            print(f"frame {f}: already done, skip")
            continue
        n = colmap_frame(colmap, work / f"colmap_{f}", cams, tri_flags)
        print(f"frame {f}: {n} points", flush=True)
        shutil.copytree(work / f"colmap_{f}" / "images", dst / "images", dirs_exist_ok=True)
        shutil.copytree(work / f"colmap_{f}" / "sparse", dst / "sparse", dirs_exist_ok=True)
    print("PREP DONE ->", out)
