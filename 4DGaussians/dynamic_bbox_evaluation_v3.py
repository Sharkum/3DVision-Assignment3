#!/usr/bin/env python3
"""Forward-backward and photometric filtered flow boxes for 4DGS evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAMERAS = (
    "cam00",
    "cam01",
    "cam02",
    "cam04",
    "cam05",
    "cam06",
    "cam07",
    "cam08",
    "cam09",
    "cam10",
    "cam11",
    "cam12",
    "cam13",
    "cam14",
    "cam16",
    "cam18",
    "cam19",
    "cam20",
)
TRAIN_CAMERAS = tuple(camera for camera in CAMERAS if camera != "cam00")
GAPS = (2, 4, 8)
EXPECTED_COUNTS = {
    2: {"training_cameras": 408, "heldout_camera": 24},
    4: {"training_cameras": 204, "heldout_camera": 12},
    8: {"training_cameras": 102, "heldout_camera": 6},
}
FRAME_COUNT = 50
IMAGE_WIDTH = 676
IMAGE_HEIGHT = 507
FLOW_THRESHOLD = 0.5
FB_ALPHA = 0.5
FB_BETA = 0.75
PHOTO_THRESHOLD = 0.012
PHOTO_BLUR_KERNEL = (7, 7)
PHOTO_BLUR_SIGMA = 1.5
FARNEBACK = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}
PROJECT_ROOT = Path(__file__).resolve().parent

BOX_COLUMNS = [
    "method",
    "temporal_gap",
    "evaluation_setting",
    "camera",
    "midpoint_frame",
    "source_frame",
    "target_frame",
    "fb_valid_pixel_count",
    "photo_valid_pixel_count",
    "combined_valid_pixel_count",
    "active_pixel_count",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max_inclusive",
    "bbox_y_max_inclusive",
    "bbox_width",
    "bbox_height",
    "bbox_area_percentage",
    "raw_flow_magnitude_max",
    "filtered_flow_magnitude_max",
]

METRIC_COLUMNS = [
    "method",
    "temporal_gap",
    "evaluation_setting",
    "camera",
    "frame_index",
    "source_frame",
    "target_frame",
    "active_pixel_count",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max_inclusive",
    "bbox_y_max_inclusive",
    "bbox_width",
    "bbox_height",
    "bbox_area_percentage",
    "PSNR_bbox",
    "SSIM_bbox",
    "LPIPS_bbox",
    "PSNR_full",
    "SSIM_full",
    "LPIPS_full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the forward-backward/photometric filtered Farneback "
            "bbox diagnostic, then optionally evaluate saved 4DGS predictions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT.parent / "coffee_martini",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "dynamic_bbox_evaluation_v3",
    )
    parser.add_argument(
        "--stage",
        choices=("diagnostic", "evaluate", "all"),
        default="diagnostic",
        help=(
            "The default only generates cam00 20->22 diagnostics. "
            "Run evaluate separately after reviewing them."
        ),
    )
    return parser.parse_args()


def ensure_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read RGB ground-truth image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(
            f"Unexpected dimensions for {path}: {rgb.shape}, expected "
            f"{(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}"
        )
    return rgb


def raw_farneback(img_a_rgb: np.ndarray, img_b_rgb: np.ndarray) -> np.ndarray:
    gray_a = cv2.cvtColor(img_a_rgb, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=FARNEBACK["pyr_scale"],
        levels=FARNEBACK["levels"],
        winsize=FARNEBACK["winsize"],
        iterations=FARNEBACK["iterations"],
        poly_n=FARNEBACK["poly_n"],
        poly_sigma=FARNEBACK["poly_sigma"],
        flags=FARNEBACK["flags"],
    )


def compute_filtered_flow(
    img_a_rgb: np.ndarray, img_b_rgb: np.ndarray
) -> dict[str, np.ndarray]:
    """Apply the friend's forward-backward and photometric filtering verbatim."""
    forward = raw_farneback(img_a_rgb, img_b_rgb)
    backward = raw_farneback(img_b_rgb, img_a_rgb)
    height, width = img_a_rgb.shape[:2]

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + forward[..., 0]
    map_y = grid_y + forward[..., 1]
    backward_warped = cv2.remap(
        backward,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    fb_error = np.linalg.norm(forward + backward_warped, axis=-1)
    forward_magnitude = np.linalg.norm(forward, axis=-1)
    valid_fb = fb_error < (FB_ALPHA * forward_magnitude + FB_BETA)

    photo_difference = np.mean(
        np.abs(img_a_rgb.astype(np.float32) - img_b_rgb.astype(np.float32))
        / 255.0,
        axis=-1,
    )
    photo_difference = cv2.GaussianBlur(
        photo_difference,
        PHOTO_BLUR_KERNEL,
        PHOTO_BLUR_SIGMA,
    )
    valid_photo = photo_difference > PHOTO_THRESHOLD
    valid = valid_fb & valid_photo
    filtered_flow = forward * valid[..., None].astype(np.float32)
    filtered_magnitude = np.linalg.norm(filtered_flow, axis=-1)
    active = filtered_magnitude > FLOW_THRESHOLD

    active_y, active_x = np.nonzero(active)
    if active_x.size == 0:
        bbox = None
    else:
        bbox = (
            int(active_x.min()),
            int(active_y.min()),
            int(active_x.max()) + 1,
            int(active_y.max()) + 1,
        )
    return {
        "forward": forward,
        "backward": backward,
        "backward_warped": backward_warped,
        "fb_error": fb_error,
        "forward_magnitude": forward_magnitude,
        "valid_fb": valid_fb,
        "photo_difference": photo_difference,
        "valid_photo": valid_photo,
        "valid": valid,
        "filtered_flow": filtered_flow,
        "filtered_magnitude": filtered_magnitude,
        "active": active,
        "bbox": bbox,
    }


def make_record(
    *,
    gap: int,
    camera: str,
    midpoint: int,
    result: dict[str, np.ndarray],
) -> dict[str, Any]:
    source_frame = midpoint - gap // 2
    target_frame = midpoint + gap // 2
    active_count = int(result["active"].sum())
    bbox = result["bbox"]
    if bbox is None:
        x0 = y0 = x1 = y1 = width = height = 0
        area_percentage = 0.0
    else:
        x0, y0, x1, y1 = bbox
        width, height = x1 - x0, y1 - y0
        area_percentage = 100.0 * width * height / (IMAGE_WIDTH * IMAGE_HEIGHT)
    return {
        "method": "4DGS",
        "temporal_gap": gap,
        "evaluation_setting": (
            "heldout_camera" if camera == "cam00" else "training_cameras"
        ),
        "camera": camera,
        "midpoint_frame": midpoint,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "fb_valid_pixel_count": int(result["valid_fb"].sum()),
        "photo_valid_pixel_count": int(result["valid_photo"].sum()),
        "combined_valid_pixel_count": int(result["valid"].sum()),
        "active_pixel_count": active_count,
        "bbox_x_min": x0,
        "bbox_y_min": y0,
        "bbox_x_max_inclusive": x1 - 1 if bbox else 0,
        "bbox_y_max_inclusive": y1 - 1 if bbox else 0,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area_percentage": area_percentage,
        "raw_flow_magnitude_max": float(result["forward_magnitude"].max()),
        "filtered_flow_magnitude_max": float(result["filtered_magnitude"].max()),
    }


def plot_reference_diagnostic(
    image_a: np.ndarray,
    result: dict[str, np.ndarray],
    record: dict[str, Any],
    output_path: Path,
) -> None:
    bbox = result["bbox"]
    overlay = image_a.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        darkened = (overlay.astype(np.float32) * 0.45).astype(np.uint8)
        darkened[y0:y1, x0:x1] = overlay[y0:y1, x0:x1]
        overlay = darkened

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    flow_max = float(result["forward_magnitude"].max())
    axes[0].imshow(
        result["forward_magnitude"],
        cmap="magma",
        vmin=0,
        vmax=max(5.0, flow_max),
    )
    axes[0].set_title(f"Raw forward flow magnitude (max={flow_max:.2f}px)")

    axes[1].imshow(result["valid_fb"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        "Forward-backward consistency mask\n"
        f"{record['fb_valid_pixel_count']} valid pixels"
    )

    photo = axes[2].imshow(
        result["photo_difference"],
        cmap="viridis",
        vmin=0,
        vmax=max(0.05, float(result["photo_difference"].max())),
    )
    axes[2].contour(
        result["valid_photo"],
        levels=[0.5],
        colors=["red"],
        linewidths=0.5,
    )
    axes[2].set_title(
        "Gaussian-smoothed RGB photometric difference\n"
        f"valid > {PHOTO_THRESHOLD:g}: "
        f"{record['photo_valid_pixel_count']} pixels"
    )
    fig.colorbar(photo, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(result["active"], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(
        f"Final filtered flow mask (magnitude > {FLOW_THRESHOLD:g}px)\n"
        f"{record['active_pixel_count']} active pixels"
    )

    axes[4].imshow(result["valid"], cmap="gray", vmin=0, vmax=1)
    axes[4].set_title(
        "Combined FB + photometric validity\n"
        f"{record['combined_valid_pixel_count']} valid pixels"
    )

    axes[5].imshow(overlay)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        rect = matplotlib.patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2.5,
            edgecolor="#00ff88",
            facecolor="none",
        )
        axes[5].add_patch(rect)
        box_title = (
            f"BBox: {record['bbox_width']}x{record['bbox_height']} px\n"
            f"x=[{x0}:{x1}), y=[{y0}:{y1})"
        )
    else:
        box_title = "No active pixels; no bounding box"
    axes[5].set_title(f"GT frame 20 overlay\n{box_title}")

    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        "Friend-reference filtering: cam00 GT pair 20→22, "
        f"threshold > {FLOW_THRESHOLD:g}px"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_reference_diagnostic(dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    image_a = load_rgb(dataset_root / "cam00" / "images" / "0020.png")
    image_b = load_rgb(dataset_root / "cam00" / "images" / "0022.png")
    result = compute_filtered_flow(image_a, image_b)
    record = make_record(gap=2, camera="cam00", midpoint=21, result=result)

    diagnostic_path = (
        output_dir / "diagnostics" / "cam00_frames20_to22.png"
    )
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    plot_reference_diagnostic(image_a, result, record, diagnostic_path)
    with (output_dir / "diagnostics" / "cam00_frames20_to22.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(
            {
                **record,
                "bbox_coordinates": (
                    "half-open x0,y0,x1,y1 in diagnostic title; inclusive "
                    "max coordinates in record"
                ),
                "opencv_version": cv2.__version__,
                "thresholds": {
                    "forward_backward_alpha": FB_ALPHA,
                    "forward_backward_beta": FB_BETA,
                    "photometric_difference": PHOTO_THRESHOLD,
                    "final_flow_magnitude": FLOW_THRESHOLD,
                },
                "diagnostic_image": str(diagnostic_path),
            },
            file,
            indent=2,
        )
        file.write("\n")

    bbox = result["bbox"]
    if bbox is None:
        coords = "none (no active pixels)"
        dimensions = "none"
    else:
        x0, y0, x1, y1 = bbox
        coords = f"x=[{x0}:{x1}), y=[{y0}:{y1})"
        dimensions = f"{x1 - x0}x{y1 - y0}"
    print(
        "REFERENCE DIAGNOSTIC cam00 20->22: "
        f"active_pixels={record['active_pixel_count']}; "
        f"bbox={coords}; dimensions={dimensions}; "
        f"raw_mag_max={record['raw_flow_magnitude_max']:.4f}; "
        f"fb_valid={record['fb_valid_pixel_count']}; "
        f"photo_valid={record['photo_valid_pixel_count']}; "
        f"combined_valid={record['combined_valid_pixel_count']}"
    )
    print(f"Diagnostic image: {diagnostic_path}")
    return record


def expected_midpoints(gap: int) -> list[int]:
    return list(range(gap // 2, 49, gap))


def preflight_predictions(project_root: Path, dataset_root: Path) -> int:
    total = 0
    for gap in GAPS:
        counts = {"training_cameras": 0, "heldout_camera": 0}
        for camera in TRAIN_CAMERAS:
            for frame in expected_midpoints(gap):
                pred, saved_gt, metrics = prediction_paths(
                    project_root, gap, camera, frame
                )
                source_gt = dataset_root / camera / "images" / f"{frame:04d}.png"
                for path in (pred, saved_gt, metrics, source_gt):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing required file: {path}")
                counts["training_cameras"] += 1
        for frame in expected_midpoints(gap):
            pred, saved_gt, metrics = prediction_paths(
                project_root, gap, "cam00", frame
            )
            source_gt = dataset_root / "cam00" / "images" / f"{frame:04d}.png"
            for path in (pred, saved_gt, metrics, source_gt):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing required file: {path}")
            counts["heldout_camera"] += 1
        if counts != EXPECTED_COUNTS[gap]:
            raise RuntimeError(
                f"k={gap} counts {counts}; expected {EXPECTED_COUNTS[gap]}"
            )
        total += sum(counts.values())
        print(
            f"Preflight k={gap}: training={counts['training_cameras']}, "
            f"heldout={counts['heldout_camera']}"
        )
    if total != 756:
        raise RuntimeError(f"Expected 756 prediction samples, found {total}")
    return total


def prediction_paths(
    project_root: Path, gap: int, camera: str, frame: int
) -> tuple[Path, Path, Path]:
    eval_dir = (
        project_root
        / "output"
        / f"coffee_martini_k{gap}"
        / "midpoint_evaluation_v4"
        / f"gap_{gap}"
    )
    stem = (
        f"heldout_cam00_{frame:04d}"
        if camera == "cam00"
        else f"train_{camera}_{frame:04d}"
    )
    return (
        eval_dir / f"{stem}_pred.png",
        eval_dir / f"{stem}_gt.png",
        eval_dir / "metrics.csv",
    )


def write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def compute_all_boxes(
    dataset_root: Path, boxes_dir: Path
) -> dict[tuple[int, str, int], dict[str, Any]]:
    ensure_absent([boxes_dir])
    boxes_dir.mkdir()
    records: dict[tuple[int, str, int], dict[str, Any]] = {}
    rows: list[list[Any]] = []
    debug_rows: list[str] = []
    total = sum(len(expected_midpoints(gap)) * len(CAMERAS) for gap in GAPS)
    progress = tqdm(total=total, desc="Filtered GT endpoint flows")
    for camera in CAMERAS:
        frames = [
            load_rgb(dataset_root / camera / "images" / f"{index:04d}.png")
            for index in range(FRAME_COUNT)
        ]
        for gap in GAPS:
            for midpoint in expected_midpoints(gap):
                source_frame = midpoint - gap // 2
                target_frame = midpoint + gap // 2
                result = compute_filtered_flow(
                    frames[source_frame], frames[target_frame]
                )
                record = make_record(
                    gap=gap,
                    camera=camera,
                    midpoint=midpoint,
                    result=result,
                )
                key = (gap, camera, midpoint)
                records[key] = record
                rows.append([record[column] for column in BOX_COLUMNS])
                x0 = record["bbox_x_min"]
                y0 = record["bbox_y_min"]
                x1 = record["bbox_x_max_inclusive"]
                y1 = record["bbox_y_max_inclusive"]
                message = (
                    f"gap={gap} camera={camera} midpoint={midpoint:04d} "
                    f"pair={source_frame:04d}->{target_frame:04d} "
                    f"active={record['active_pixel_count']} "
                    f"bbox_inclusive=({x0},{y0},{x1},{y1}) "
                    f"size={record['bbox_width']}x{record['bbox_height']}"
                )
                print(message)
                debug_rows.append(message)
                progress.update(1)
    progress.close()
    if len(records) != total:
        raise RuntimeError(f"Computed {len(records)} boxes; expected {total}")
    write_csv(boxes_dir / "flow_boxes.csv", BOX_COLUMNS, rows)
    with (boxes_dir / "debug_log.txt").open("x", encoding="utf-8") as file:
        file.write("\n".join(debug_rows) + "\n")
    return records


def load_rgb_prediction(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return array


def gaussian_window(
    size: int, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coord = torch.arange(size, device=device, dtype=dtype) - size // 2
    vals = torch.exp(-(coord**2) / (2 * 1.5**2))
    vals /= vals.sum()
    kernel = vals[:, None] @ vals[None, :]
    return kernel[None, None].expand(channels, 1, size, size).contiguous()


def ssim_map(
    prediction: torch.Tensor, target: torch.Tensor, size: int = 11
) -> torch.Tensor:
    channels = prediction.shape[1]
    window = gaussian_window(size, channels, prediction.device, prediction.dtype)
    padding = size // 2
    mu1 = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu2 = F.conv2d(target, window, padding=padding, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1 = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mu1_sq
    )
    sigma2 = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mu12
    )
    c1, c2 = 0.01**2, 0.03**2
    values = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return values.mean(dim=1, keepdim=True)


def metrics_for_sample(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    record: dict[str, Any],
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    if pred_rgb.shape != gt_rgb.shape or pred_rgb.shape != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3,
    ):
        raise ValueError(
            f"Prediction/GT dimensions mismatch: {pred_rgb.shape} vs {gt_rgb.shape}"
        )
    if record["active_pixel_count"] <= 0:
        raise ValueError(
            "Cannot evaluate an empty bbox for "
            f"{record['camera']} frame {record['midpoint_frame']}"
        )
    pred = (
        torch.from_numpy(pred_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    gt = (
        torch.from_numpy(gt_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    x0 = int(record["bbox_x_min"])
    y0 = int(record["bbox_y_min"])
    x1 = int(record["bbox_x_max_inclusive"]) + 1
    y1 = int(record["bbox_y_max_inclusive"]) + 1
    diff = pred - gt
    mse_bbox = diff[:, :, y0:y1, x0:x1].square().mean()
    mse_full = diff.square().mean()

    spatial_ssim = ssim_map(pred, gt)[0, 0]
    ssim_bbox = spatial_ssim[y0:y1, x0:x1].mean()
    lpips_spatial = lpips_model(pred * 2 - 1, gt * 2 - 1)
    if lpips_spatial.shape != (1, 1, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Unexpected LPIPS map shape: {lpips_spatial.shape}")
    lpips_bbox = lpips_spatial[0, 0, y0:y1, x0:x1].mean()

    def psnr(mse: torch.Tensor) -> float:
        value = float(mse.item())
        return math.inf if value == 0 else -10.0 * math.log10(value)

    return {
        "PSNR_bbox": psnr(mse_bbox),
        "SSIM_bbox": float(ssim_bbox.item()),
        "LPIPS_bbox": float(lpips_bbox.item()),
        "PSNR_full": psnr(mse_full),
        "SSIM_full": float(spatial_ssim.mean().item()),
        "LPIPS_full": float(lpips_spatial.mean().item()),
    }


def write_metrics(
    output_dir: Path,
    project_root: Path,
    dataset_root: Path,
    records: dict[tuple[int, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    import lpips

    preflight_predictions(project_root, dataset_root)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    ensure_absent([metrics_dir, figures_dir])
    metrics_dir.mkdir()
    figures_dir.mkdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Metric device: {device}")
    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()

    results: list[dict[str, Any]] = []
    for gap in GAPS:
        total = sum(EXPECTED_COUNTS[gap].values())
        progress = tqdm(total=total, desc=f"Evaluating k={gap}")
        for setting, cameras in (
            ("training_cameras", TRAIN_CAMERAS),
            ("heldout_camera", ("cam00",)),
        ):
            for camera in cameras:
                for frame in expected_midpoints(gap):
                    record = records[(gap, camera, frame)]
                    pred_path, saved_gt_path, _ = prediction_paths(
                        project_root, gap, camera, frame
                    )
                    source_gt_path = (
                        dataset_root / camera / "images" / f"{frame:04d}.png"
                    )
                    pred = load_rgb_prediction(pred_path)
                    gt = load_rgb_prediction(source_gt_path)
                    saved_gt = load_rgb_prediction(saved_gt_path)
                    if not np.array_equal(gt, saved_gt):
                        raise ValueError(
                            f"Saved midpoint GT differs from source: {saved_gt_path}"
                        )
                    values = metrics_for_sample(
                        pred, gt, record, lpips_model, device
                    )
                    results.append(
                        {
                            "method": "4DGS",
                            "temporal_gap": gap,
                            "evaluation_setting": setting,
                            "camera": camera,
                            "frame_index": frame,
                            "source_frame": record["source_frame"],
                            "target_frame": record["target_frame"],
                            "active_pixel_count": record["active_pixel_count"],
                            "bbox_x_min": record["bbox_x_min"],
                            "bbox_y_min": record["bbox_y_min"],
                            "bbox_x_max_inclusive": record[
                                "bbox_x_max_inclusive"
                            ],
                            "bbox_y_max_inclusive": record[
                                "bbox_y_max_inclusive"
                            ],
                            "bbox_width": record["bbox_width"],
                            "bbox_height": record["bbox_height"],
                            "bbox_area_percentage": record[
                                "bbox_area_percentage"
                            ],
                            **values,
                        }
                    )
                    progress.update(1)
        progress.close()
        actual = sum(row["temporal_gap"] == gap for row in results)
        if actual != total:
            raise RuntimeError(f"k={gap}: got {actual} results, expected {total}")

    if len(results) != 756:
        raise RuntimeError(f"Got {len(results)} samples; expected 756")
    save_metric_csvs(metrics_dir, results)
    save_metric_figure(figures_dir, results)
    return results


def save_metric_csvs(
    metrics_dir: Path, results: list[dict[str, Any]]
) -> None:
    write_csv(
        metrics_dir / "4dgs_bbox_v3_metrics.csv",
        METRIC_COLUMNS,
        [[row[column] for column in METRIC_COLUMNS] for row in results],
    )
    summary: list[list[Any]] = []
    for gap in GAPS:
        for setting in ("training_cameras", "heldout_camera"):
            subset = [
                row
                for row in results
                if row["temporal_gap"] == gap
                and row["evaluation_setting"] == setting
            ]
            if len(subset) != EXPECTED_COUNTS[gap][setting]:
                raise RuntimeError(
                    f"Incomplete summary group: k={gap}, setting={setting}"
                )
            for scope in ("bbox", "full"):
                means = [
                    float(np.mean([row[f"{name}_{scope}"] for row in subset]))
                    for name in ("PSNR", "SSIM", "LPIPS")
                ]
                summary.append(
                    ["4DGS", gap, setting, scope, *means, len(subset)]
                )
    write_csv(
        metrics_dir / "4dgs_bbox_v3_summary.csv",
        [
            "method",
            "temporal_gap",
            "evaluation_setting",
            "metric_scope",
            "mean_PSNR",
            "mean_SSIM",
            "mean_LPIPS",
            "num_samples",
        ],
        summary,
    )


def save_metric_figure(
    figures_dir: Path, results: list[dict[str, Any]]
) -> None:
    metrics = (
        ("PSNR", "PSNR (dB)"),
        ("SSIM", "SSIM"),
        ("LPIPS", "LPIPS distance"),
    )
    groups = (
        ("training_cameras", "full", "#4C78A8"),
        ("training_cameras", "bbox", "#F58518"),
        ("heldout_camera", "full", "#72B7B2"),
        ("heldout_camera", "bbox", "#E45756"),
    )
    x = np.arange(len(GAPS))
    width = 0.18
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for index, (setting, scope, color) in enumerate(groups):
            means = []
            for gap in GAPS:
                values = [
                    row[f"{metric}_{scope}"]
                    for row in results
                    if row["temporal_gap"] == gap
                    and row["evaluation_setting"] == setting
                ]
                means.append(float(np.mean(values)))
            axis.bar(
                x + (index - 1.5) * width,
                means,
                width,
                color=color,
                label=f"{setting.replace('_', ' ')} / {scope}",
            )
        axis.set_xticks(x, [f"k={gap}" for gap in GAPS])
        axis.set_ylabel(ylabel)
        axis.set_title(metric)
        axis.legend(fontsize=7)
    fig.suptitle("V3 filtered-flow bbox vs full-image metrics")
    fig.savefig(figures_dir / "bbox_v3_vs_full.png", dpi=180)
    plt.close(fig)


def write_readme(output_dir: Path) -> None:
    text = f"""# Dynamic-region evaluation V3: filtered optical-flow boxes

This implementation is separate from previous bbox/pixel-mask scripts and
outputs. It uses the friend's `PlugAndPlayOpticalFlow.compute_flow()` filtering:

1. RGB ground-truth frames converted to grayscale for Farneback.
2. Farneback parameters: `{json.dumps(FARNEBACK, sort_keys=True)}`.
3. Forward and backward flow; backward flow warped with `grid + forward`.
4. Forward-backward validity: `fb_error < {FB_ALPHA} * forward_magnitude + {FB_BETA}`.
5. RGB mean absolute difference divided by 255, GaussianBlur
   `ksize={PHOTO_BLUR_KERNEL}`, `sigmaX={PHOTO_BLUR_SIGMA}`, then
   `photo_difference > {PHOTO_THRESHOLD}`.
6. Intersect both validity masks, set invalid forward vectors to zero, then
   select filtered-flow magnitude strictly `> {FLOW_THRESHOLD}`.
7. Bounding box uses min/max active coordinates with no padding or additional
   filtering.

The default `--stage diagnostic` only processes cam00 frames 20->22 and creates
the requested diagnostic. Review it before running
`--stage evaluate --output-dir output/dynamic_bbox_evaluation_v3`. Evaluation
uses the source frames around each midpoint: t-k/2 and t+k/2. Existing
predictions are reused; no model is rerendered.

Thresholds and filtering parameters are not adjusted to force small boxes.
"""
    with (output_dir / "README.md").open("x", encoding="utf-8") as file:
        file.write(text)


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.project_root = args.project_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_root}")

    if args.stage in ("diagnostic", "all"):
        ensure_absent([args.output_dir])
        args.output_dir.mkdir(parents=True)
        run_reference_diagnostic(args.dataset_root, args.output_dir)
        write_readme(args.output_dir)
        if args.stage == "diagnostic":
            print(
                "Stopped after the requested reference diagnostic. "
                "Use --stage evaluate only after reviewing it."
            )
            return
    else:
        diagnostic = (
            args.output_dir / "diagnostics" / "cam00_frames20_to22.png"
        )
        if not diagnostic.is_file():
            raise FileNotFoundError(
                f"Run --stage diagnostic first: {diagnostic}"
            )

    records = compute_all_boxes(
        args.dataset_root,
        args.output_dir / "boxes",
    )
    results = write_metrics(
        args.output_dir,
        args.project_root,
        args.dataset_root,
        records,
    )
    print(f"Evaluated {len(results)} midpoint samples.")
    print(f"V3 metrics: {args.output_dir / 'metrics'}")


if __name__ == "__main__":
    main()
