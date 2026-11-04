import os
import sys
import struct
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add TAD-GS preprocessing utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TAD-GS" / "utils" / "preprocessing" / "neural_3d_video"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TAD-GS" / "utils" / "common"))
from pose_utils import posetow2c_matrcs, rotmat2qvec, qvec2rotmat
from plyfile import PlyData, PlyElement


def write_cameras_binary(cameras, path):
    """Write COLMAP cameras.bin (PINHOLE model_id = 1)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            # camera_id (i), model_id=1 (i), width (Q), height (Q)
            f.write(struct.pack("<iiQQ", int(cam["id"]), 1, int(cam["w"]), int(cam["h"])))
            f.write(struct.pack("<dddd", float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])))


def write_images_binary(cameras, path):
    """Write COLMAP images.bin with known poses and empty 2D point lists."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            img_id = int(cam["id"])
            q = [float(x) for x in cam["q"]]
            t = [float(x) for x in cam["t"]]
            cam_id = int(cam["id"])
            f.write(struct.pack("<idddddddi", img_id, q[0], q[1], q[2], q[3], t[0], t[1], t[2], cam_id))
            f.write(cam["filename"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2D = 0


def write_points3D_binary(xyz, rgb, path):
    """Write COLMAP points3D.bin."""
    num_pts = xyz.shape[0]
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", num_pts))
        for i in range(num_pts):
            # point3D_id (Q), x, y, z (ddd), r, g, b (BBB), error (d), track_len=0 (Q)
            f.write(struct.pack(
                "<QdddBBBdQ",
                i + 1,
                float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
                int(rgb_u8[i, 0]), int(rgb_u8[i, 1]), int(rgb_u8[i, 2]),
                0.5,
                0
            ))


def write_ply_xyzrgb(xyz, rgb, path):
    """Write fused_downsample.ply (x, y, z, nx, ny, nz, red, green, blue)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")
    ]
    arr = np.empty(xyz.shape[0], dtype=dtype)
    arr["x"] = xyz[:, 0].astype(np.float32)
    arr["y"] = xyz[:, 1].astype(np.float32)
    arr["z"] = xyz[:, 2].astype(np.float32)
    arr["nx"] = 0.0
    arr["ny"] = 0.0
    arr["nz"] = 0.0
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    arr["red"] = rgb_u8[:, 0]
    arr["green"] = rgb_u8[:, 1]
    arr["blue"] = rgb_u8[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def triangulate_sift_and_mvs(images_dir: Path, cameras: list, near: float, far: float, max_points: int = 36000):
    """Triangulate 3D points using ONLY training cameras (cam01..cam17, excluding cam00)
    via SIFT matching + GPU Multi-View Plane-Sweep Stereo + multi-view consistency filtering."""
    train_cams = [c for c in cameras if c["filename"] != "cam00.png"]
    n_cams = len(train_cams)

    imgs_bgr = []
    imgs_rgb = []
    Ks = []
    Rs = []
    Ts = []
    Ps = []
    C_worlds = []

    for c in train_cams:
        img_path = images_dir / c["filename"]
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs_bgr.append(bgr)
        imgs_rgb.append(rgb)

        K = np.array([
            [c["fx"], 0.0, c["cx"]],
            [0.0, c["fy"], c["cy"]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        R = qvec2rotmat(c["q"]).astype(np.float64)
        t = np.array(c["t"], dtype=np.float64).reshape(3, 1)
        P = K @ np.hstack([R, t])
        C_w = (-R.T @ t).flatten()

        Ks.append(K)
        Rs.append(R)
        Ts.append(t)
        Ps.append(P)
        C_worlds.append(C_w)

    C_worlds = np.stack(C_worlds, axis=0)  # (N, 3)

    # 1. Sparse SIFT triangulation across neighboring camera pairs
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01)
    kps_list, des_list = [], []
    for bgr in imgs_bgr:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kps, des = sift.detectAndCompute(gray, None)
        kps_list.append(kps)
        des_list.append(des)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    sift_xyz = []
    sift_rgb = []

    # Stack all training images/matrices on GPU for fast multi-view photometric verification
    H, W = imgs_rgb[0].shape[:2]
    imgs_t = torch.from_numpy(np.stack(imgs_rgb, axis=0)).float().cuda() / 255.0  # (N, H, W, 3)
    Ps_t = torch.from_numpy(np.stack(Ps, axis=0)).float().cuda()  # (N, 3, 4)

    for i in range(n_cams):
        # Find 5 nearest neighbor cameras to camera i
        dists = np.linalg.norm(C_worlds - C_worlds[i], axis=1)
        nbrs = np.argsort(dists)[1:6]
        for j in nbrs:
            if j <= i:
                continue
            matches = bf.knnMatch(des_list[i], des_list[j], k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
            if len(good) < 20:
                continue
            pts_i = np.float64([kps_list[i][m.queryIdx].pt for m in good]).T
            pts_j = np.float64([kps_list[j][m.trainIdx].pt for m in good]).T

            pts4d = cv2.triangulatePoints(Ps[i], Ps[j], pts_i, pts_j)
            pts3d = (pts4d[:3] / (pts4d[3:] + 1e-12)).T  # (M, 3)

            # Check depth in cam i and cam j and world z >= 4.6
            z_i = (Rs[i] @ pts3d.T + Ts[i])[2]
            z_j = (Rs[j] @ pts3d.T + Ts[j])[2]
            proj_i = (Ps[i][:, :3] @ pts3d.T + Ps[i][:, 3:4])
            proj_i = proj_i[:2] / (proj_i[2:3] + 1e-12)
            proj_j = (Ps[j][:, :3] @ pts3d.T + Ps[j][:, 3:4])
            proj_j = proj_j[:2] / (proj_j[2:3] + 1e-12)

            err_i = np.linalg.norm(proj_i - pts_i, axis=0)
            err_j = np.linalg.norm(proj_j - pts_j, axis=0)

            valid = (
                (z_i > near * 0.8) & (z_i < far * 1.1) &
                (z_j > near * 0.8) & (z_j < far * 1.1) &
                (pts3d[:, 2] >= 4.6) & (pts3d[:, 2] <= far * 1.1) &
                (err_i < 1.5) & (err_j < 1.5)
            )
            if valid.sum() > 0:
                v_pts = pts3d[valid]
                v_px = np.clip(np.round(pts_i[:, valid]).astype(int), [[0], [0]], [[W - 1], [H - 1]])
                v_rgb = imgs_rgb[i][v_px[1], v_px[0]]
                sift_xyz.append(v_pts)
                sift_rgb.append(v_rgb)

    sift_xyz = np.concatenate(sift_xyz, axis=0) if sift_xyz else np.zeros((0, 3))
    sift_rgb = np.concatenate(sift_rgb, axis=0) if sift_rgb else np.zeros((0, 3))

    # 2. GPU Multi-View Plane Sweep across reference training cameras for dense coverage
    mvs_xyz = []
    mvs_rgb = []
    # Use a stride of 4 pixels on 6 reference views spanning the rig
    ref_indices = np.linspace(0, n_cams - 1, min(6, n_cams), dtype=int)
    stride = 4
    ys, xs = torch.meshgrid(
        torch.arange(2, H - 2, stride, device="cuda", dtype=torch.float32),
        torch.arange(2, W - 2, stride, device="cuda", dtype=torch.float32),
        indexing="ij"
    )
    n_pix = ys.numel()
    ones = torch.ones_like(xs)
    pix_h = torch.stack([xs.flatten(), ys.flatten(), ones.flatten()], dim=0)  # (3, P)

    # 96 inverse depth planes between near and far
    inv_depths = torch.linspace(1.0 / (far * 0.9), 1.0 / (near * 0.95), 96, device="cuda")
    depths = 1.0 / inv_depths  # (D,)

    # 3x3 patch offsets for robust multi-view photometric matching
    imgs_nchw = imgs_t.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs_blur = F.avg_pool2d(imgs_nchw, kernel_size=3, stride=1, padding=1)

    for ref_idx in ref_indices:
        dists = np.linalg.norm(C_worlds - C_worlds[ref_idx], axis=1)
        src_indices = np.argsort(dists)[1:5]

        K_ref_inv = torch.from_numpy(np.linalg.inv(Ks[ref_idx])).float().cuda()
        R_ref_inv = torch.from_numpy(Rs[ref_idx].T).float().cuda()
        t_ref = torch.from_numpy(Ts[ref_idx]).float().cuda()

        # Rays in camera space: K^{-1} [u, v, 1]^T -> (3, P)
        rays_cam = K_ref_inv @ pix_h
        # 3D points for all D depths: (D, 3, P)
        pts_cam = rays_cam.unsqueeze(0) * depths.view(-1, 1, 1)
        pts_world = torch.einsum("ij,djp->dip", R_ref_inv, pts_cam - t_ref.unsqueeze(0))  # (D, 3, P)

        ref_colors = imgs_blur[ref_idx, :, ys.long().flatten(), xs.long().flatten()]  # (3, P)

        cost_volume = torch.zeros((len(depths), n_pix), device="cuda")
        valid_cnt = torch.zeros((len(depths), n_pix), device="cuda")

        for src_idx in src_indices:
            P_src = Ps_t[src_idx]  # (3, 4)
            proj = torch.einsum("ij,djp->dip", P_src[:, :3], pts_world) + P_src[:, 3:4].unsqueeze(0)  # (D, 3, P)
            u_src = proj[:, 0, :] / (proj[:, 2, :].clamp(min=1e-5))
            v_src = proj[:, 1, :] / (proj[:, 2, :].clamp(min=1e-5))
            in_bounds = (proj[:, 2, :] > 0) & (u_src >= 2) & (u_src < W - 2) & (v_src >= 2) & (v_src < H - 2)

            grid_x = (u_src / (W - 1)) * 2.0 - 1.0
            grid_y = (v_src / (H - 1)) * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, D, P, 2)

            sampled = F.grid_sample(
                imgs_blur[src_idx:src_idx + 1], grid, mode="bilinear", padding_mode="border", align_corners=True
            ).squeeze(0)  # (3, D, P)
            diff = (sampled - ref_colors.unsqueeze(1)).abs().mean(dim=0)  # (D, P)
            cost_volume += torch.where(in_bounds, diff, torch.full_like(diff, 0.5))
            valid_cnt += in_bounds.float()

        avg_cost = cost_volume / valid_cnt.clamp(min=1.0)
        best_cost, best_d_idx = avg_cost.min(dim=0)  # (P,)

        # Gather best 3D points
        best_pts = pts_world[best_d_idx, :, torch.arange(n_pix, device="cuda")]  # (P, 3)
        good_mask = (
            (best_cost < 0.032) &
            (valid_cnt[best_d_idx, torch.arange(n_pix, device="cuda")] >= 3) &
            (best_pts[:, 2] >= 4.6) &
            (best_pts[:, 2] <= far * 1.05)
        )
        if good_mask.sum() > 0:
            pts_np = best_pts[good_mask].cpu().numpy()
            rgb_np = (imgs_t[ref_idx, ys.long().flatten()[good_mask], xs.long().flatten()[good_mask]] * 255.0).cpu().numpy()
            mvs_xyz.append(pts_np)
            mvs_rgb.append(rgb_np)

    all_xyz = np.concatenate([sift_xyz] + mvs_xyz, axis=0)
    all_rgb = np.concatenate([sift_rgb] + mvs_rgb, axis=0)

    # Voxel-grid filter + cap at max_points (always keeping high-precision SIFT points)
    if all_xyz.shape[0] > max_points:
        # Keep all SIFT points (up to max_points // 2) and randomly/voxel sample MVS points
        n_sift = min(len(sift_xyz), max_points // 2)
        sift_idx = np.random.choice(len(sift_xyz), n_sift, replace=False) if len(sift_xyz) > n_sift else np.arange(len(sift_xyz))
        n_rem = max_points - len(sift_idx)
        mvs_total_xyz = np.concatenate(mvs_xyz, axis=0)
        mvs_total_rgb = np.concatenate(mvs_rgb, axis=0)
        mvs_idx = np.random.choice(len(mvs_total_xyz), min(len(mvs_total_xyz), n_rem), replace=False)
        all_xyz = np.concatenate([sift_xyz[sift_idx], mvs_total_xyz[mvs_idx]], axis=0)
        all_rgb = np.concatenate([sift_rgb[sift_idx], mvs_total_rgb[mvs_idx]], axis=0)

    print(f"Triangulated {len(all_xyz)} 3D points (SIFT: {len(sift_xyz)}, MVS kept: {len(all_xyz) - min(len(sift_xyz), max_points // 2)})")
    return all_xyz, all_rgb


def main():
    np.random.seed(1995)
    torch.manual_seed(1995)

    workspace = Path(__file__).resolve().parent.parent
    raw_scene = workspace / "data" / "neural_3d_video" / "coffee_martini"
    scenes_dir = workspace / "data" / "neural_3d_video" / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = scenes_dir / "coffee_martini"

    if raw_scene.exists() and not raw_scene.is_symlink() and not scene_dir.exists():
        shutil.move(str(raw_scene), str(scene_dir))
        raw_scene.symlink_to(scene_dir)
    elif not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found at {scene_dir}")

    startframe = 0
    endframe = 50
    downscale = 4
    # Anchor offsets: 0, 24, 48 are all multiples of 8, so they belong to T_train(k) for ALL k in {2, 4, 8}!
    anchor_offsets = [0, 24, 48]

    video_paths = sorted(scene_dir.glob("cam*.mp4"))
    seq_names = {v.stem: f"cam{i:02d}" for i, v in enumerate(video_paths)}
    print(f"Found {len(video_paths)} camera videos. Held-out test camera: {video_paths[0].stem} -> cam00.png")

    # Load poses_bounds.npy
    poses_bounds = np.load(scene_dir / "poses_bounds.npy")
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    bounds = poses_bounds[:, -2:]
    near = float(bounds.min() * 0.95)
    far = float(bounds.max() * 1.05)

    llffposes = poses.copy().transpose(1, 2, 0)
    w2c_list = posetow2c_matrcs(llffposes)

    cameras = []
    for i in range(len(poses)):
        m = w2c_list[i]
        colmapR = m[:3, :3]
        T = m[:3, 3]
        H, W, focal = poses[i, :, -1] / downscale
        origstem = video_paths[i].stem
        cameras.append({
            "id": i + 1,
            "orig_stem": origstem,
            "filename": f"{seq_names[origstem]}.png",
            "w": int(round(W)),
            "h": int(round(H)),
            "fx": float(focal),
            "fy": float(focal),
            "cx": float(int(round(W)) // 2),
            "cy": float(int(round(H)) // 2),
            "q": rotmat2qvec(colmapR),
            "t": T,
        })

    # Extract first 50 frames (0..49) at 4x downscale directly into colmap_<t>/images/camXX.png
    for offset in range(startframe, endframe):
        (scene_dir / f"colmap_{offset}" / "images").mkdir(parents=True, exist_ok=True)

    for v in tqdm(video_paths, desc="Extracting 50 frames (4x downscaled to 676x507)"):
        seq_png = f"{seq_names[v.stem]}.png"
        if all((scene_dir / f"colmap_{t}" / "images" / seq_png).exists() for t in range(startframe, endframe)):
            continue
        cap = cv2.VideoCapture(str(v))
        cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
        for t in range(startframe, endframe):
            ok, frame = cap.read()
            assert ok, f"Failed reading frame {t} from {v}"
            out_path = scene_dir / f"colmap_{t}" / "images" / seq_png
            if not out_path.exists():
                new_w = int(round(frame.shape[1] / downscale))
                new_h = int(round(frame.shape[0] / downscale))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_path), resized)
        cap.release()

    # Write manual/ and sparse/0/ camera files for all 50 frames
    for offset in range(startframe, endframe):
        pfolder = scene_dir / f"colmap_{offset}"
        sparse0 = pfolder / "sparse" / "0"
        sparse0.mkdir(parents=True, exist_ok=True)
        write_cameras_binary(cameras, sparse0 / "cameras.bin")
        write_images_binary(cameras, sparse0 / "images.bin")

    # Triangulate initial point clouds at anchor_offsets = [0, 24, 48] (all in training set for k=2,4,8)
    for offset in anchor_offsets:
        pfolder = scene_dir / f"colmap_{offset}"
        sparse0 = pfolder / "sparse" / "0"
        dense_ply = pfolder / "dense" / "workspace" / "fused_downsample.ply"
        pts_bin = sparse0 / "points3D.bin"
        if dense_ply.exists() and pts_bin.exists():
            print(f"Anchor frame {offset} point cloud already exists.")
            continue
        print(f"Triangulating point cloud for anchor frame t={offset} (using training cameras only)...")
        xyz, rgb = triangulate_sift_and_mvs(pfolder / "images", cameras, near, far, max_points=32000)
        write_points3D_binary(xyz, rgb, pts_bin)
        write_ply_xyzrgb(xyz, rgb, dense_ply)

    print("Preprocessing of coffee_martini (50 frames, 4x downscale) complete!")


if __name__ == "__main__":
    main()
