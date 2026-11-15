#!/usr/bin/env python3
"""Gap-endpoint Farneback boxes and 4DGS midpoint evaluation."""

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
from PIL import Image, ImageDraw
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
    "temporal_gap",
    "camera",
    "evaluation_setting",
    "midpoint_frame",
    "source_frame",
    "target_frame",
    "active_pixel_count",
    "active_pixel_percentage",
    "x_min",
    "y_min",
    "x_max_inclusive",
    "y_max_inclusive",
    "bbox_width",
    "bbox_height",
    "bbox_area_pixels",
    "bbox_area_percentage",
    "flow_magnitude_max",
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
            "Compute independent GT Farneback boxes from temporal-gap "
            "endpoints and evaluate saved 4DGS midpoint predictions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT.parent / "coffee_martini",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "dynamic_bbox_reference_v1",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "boxes", "evaluate"),
        default="all",
    )
    return parser.parse_args()


def ensure_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def midpoint_frames(gap: int) -> list[int]:
    return list(range(gap // 2, 49, gap))


def flow_endpoints(gap: int, midpoint: int) -> tuple[int, int]:
    return midpoint - gap // 2, midpoint + gap // 2


def setting_for(camera: str) -> str:
    return "heldout_camera" if camera == "cam00" else "training_cameras"


def load_gray_frames(dataset_root: Path, camera: str) -> np.ndarray:
    image_dir = dataset_root / camera / "images"
    expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    frames: list[np.ndarray] = []
    for frame_index in range(FRAME_COUNT):
        path = image_dir / f"{frame_index:04d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ground-truth frame: {path}")
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise OSError(f"Could not read ground-truth frame: {path}")
        if frame.shape != expected_shape:
            raise ValueError(
                f"Unexpected dimensions for {path}: {frame.shape}, "
                f"expected {expected_shape}"
            )
        frames.append(frame)
    return np.stack(frames, axis=0)


def calculate_flow_and_box(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], int]:
    flow = cv2.calcOpticalFlowFarneback(
        source,
        target,
        None,
        FARNEBACK["pyr_scale"],
        FARNEBACK["levels"],
        FARNEBACK["winsize"],
        FARNEBACK["iterations"],
        FARNEBACK["poly_n"],
        FARNEBACK["poly_sigma"],
        FARNEBACK["flags"],
    )
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    active = magnitude > FLOW_THRESHOLD
    active_y, active_x = np.nonzero(active)
    active_count = int(active_x.size)
    if active_count == 0:
        raise ValueError(
            f"No active pixels for magnitude > {FLOW_THRESHOLD} px"
        )
    box = (
        int(active_x.min()),
        int(active_y.min()),
        int(active_x.max()),
        int(active_y.max()),
    )
    return magnitude, active, box, active_count


def box_record(
    gap: int,
    camera: str,
    midpoint: int,
    magnitude: np.ndarray,
    box: tuple[int, int, int, int],
    active_count: int,
) -> dict[str, Any]:
    source_frame, target_frame = flow_endpoints(gap, midpoint)
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = width * height
    return {
        "temporal_gap": gap,
        "camera": camera,
        "evaluation_setting": setting_for(camera),
        "midpoint_frame": midpoint,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "active_pixel_count": active_count,
        "active_pixel_percentage": 100.0
        * active_count
        / (IMAGE_WIDTH * IMAGE_HEIGHT),
        "x_min": x_min,
        "y_min": y_min,
        "x_max_inclusive": x_max,
        "y_max_inclusive": y_max,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area_pixels": area,
        "bbox_area_percentage": 100.0
        * area
        / (IMAGE_WIDTH * IMAGE_HEIGHT),
        "flow_magnitude_max": float(magnitude.max()),
    }


def save_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def save_bbox_overlay(
    dataset_root: Path,
    camera: str,
    midpoint: int,
    record: dict[str, Any],
    path: Path,
) -> None:
    image_path = (
        dataset_root / camera / "images" / f"{midpoint:04d}.png"
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (
            record["x_min"],
            record["y_min"],
            record["x_max_inclusive"],
            record["y_max_inclusive"],
        ),
        outline=(0, 255, 120),
        width=3,
    )
    label = (
        f"{camera} gap={record['temporal_gap']} "
        f"{record['source_frame']:04d}->{record['target_frame']:04d} "
        f"active={record['active_pixel_count']}"
    )
    draw.text(
        (8, 8),
        label,
        fill=(255, 255, 0),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    image.save(path)


def save_reference_diagnostic(
    dataset_root: Path,
    output_dir: Path,
    camera: str,
    gap: int,
    midpoint: int,
    source: np.ndarray,
    magnitude: np.ndarray,
    active: np.ndarray,
    record: dict[str, Any],
) -> None:
    source_frame, target_frame = flow_endpoints(gap, midpoint)
    image_path = dataset_root / camera / "images" / f"{midpoint:04d}.png"
    with Image.open(image_path) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = Image.fromarray(original.copy())
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (
            record["x_min"],
            record["y_min"],
            record["x_max_inclusive"],
            record["y_max_inclusive"],
        ),
        outline=(0, 255, 120),
        width=4,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(magnitude, cmap="magma", vmin=0)
    axes[0].set_title(
        f"Flow magnitude ||u||₂, max={magnitude.max():.2f}px\n"
        f"GT pair {source_frame:04d}→{target_frame:04d}"
    )
    axes[1].imshow(active, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Active pixels: ||u||₂ > {FLOW_THRESHOLD:g}px\n"
        f"{record['active_pixel_count']} active pixels"
    )
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"GT midpoint {midpoint:04d}, box "
        f"{record['bbox_width']}×{record['bbox_height']} px\n"
        f"x=[{record['x_min']},{record['x_max_inclusive']}], "
        f"y=[{record['y_min']},{record['y_max_inclusive']}]"
    )
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"Independent endpoint-flow bbox ({camera}, "
        f"gap={gap}, pair={source_frame}→{target_frame})"
    )
    fig.savefig(output_dir, dpi=180)
    plt.close(fig)


def generate_boxes(
    dataset_root: Path,
    output_dir: Path,
) -> dict[tuple[int, str, int], dict[str, Any]]:
    boxes_dir = output_dir / "boxes"
    diagnostic_dir = output_dir / "diagnostics"
    overlay_dir = diagnostic_dir / "bbox_overlays"
    ensure_absent([boxes_dir, diagnostic_dir, output_dir / "README.md"])
    boxes_dir.mkdir()
    overlay_dir.mkdir(parents=True)

    records: dict[tuple[int, str, int], dict[str, Any]] = {}
    all_rows: list[list[Any]] = []
    debug_rows: list[str] = []
    reference_visual: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ] | None = None

    total = sum(len(midpoint_frames(gap)) * len(CAMERAS) for gap in GAPS)
    progress = tqdm(total=total, desc="Independent GT endpoint flow")
    for camera in CAMERAS:
        frames = load_gray_frames(dataset_root, camera)
        camera_overlay_dir = overlay_dir / camera
        camera_overlay_dir.mkdir()
        for gap in GAPS:
            for midpoint in midpoint_frames(gap):
                source_frame, target_frame = flow_endpoints(gap, midpoint)
                magnitude, active, box, active_count = calculate_flow_and_box(
                    frames[source_frame], frames[target_frame]
                )
                record = box_record(
                    gap, camera, midpoint, magnitude, box, active_count
                )
                key = (gap, camera, midpoint)
                records[key] = record
                all_rows.append([record[column] for column in BOX_COLUMNS])
                save_bbox_overlay(
                    dataset_root,
                    camera,
                    midpoint,
                    record,
                    camera_overlay_dir
                    / (
                        f"gap_{gap}_mid_{midpoint:04d}_"
                        f"pair_{source_frame:04d}_{target_frame:04d}.png"
                    ),
                )
                debug = (
                    f"gap={gap} camera={camera} midpoint={midpoint:04d} "
                    f"pair={source_frame:04d}->{target_frame:04d} "
                    f"active_pixels={active_count} "
                    f"bbox_xyxy_inclusive="
                    f"({box[0]},{box[1]},{box[2]},{box[3]}) "
                    f"size={record['bbox_width']}x{record['bbox_height']}"
                )
                print(debug)
                debug_rows.append(debug)
                if camera == "cam00" and gap == 2 and midpoint == 21:
                    reference_visual = (frames[source_frame], magnitude, active, record)
                    print("REFERENCE PAIR DEBUG:", debug)
                progress.update(1)
    progress.close()

    expected_count = total
    if len(records) != expected_count:
        raise RuntimeError(
            f"Created {len(records)} individual boxes, expected {expected_count}"
        )
    save_csv(boxes_dir / "flow_boxes.csv", BOX_COLUMNS, all_rows)
    with (boxes_dir / "debug_log.txt").open("x", encoding="utf-8") as file:
        file.write("\n".join(debug_rows))
        file.write("\n")

    diagnostic_dir.mkdir(exist_ok=True)
    if reference_visual is None:
        raise RuntimeError("Reference cam00 pair 20->22 diagnostic was not made")
    source, magnitude, active, record = reference_visual
    save_reference_diagnostic(
        dataset_root,
        diagnostic_dir / "cam00_gap2_pair20_to22.png",
        "cam00",
        2,
        21,
        source,
        magnitude,
        active,
        record,
    )
    save_overlay_montage(dataset_root, diagnostic_dir, records)

    active_percentages = [
        record["active_pixel_percentage"] for record in records.values()
    ]
    bbox_percentages = [
        record["bbox_area_percentage"] for record in records.values()
    ]
    metadata = {
        "dataset": "Neural 3D Video / coffee_martini",
        "cameras": list(CAMERAS),
        "gaps": list(GAPS),
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "flow_method": "OpenCV Farneback dense optical flow",
        "farneback_parameters": FARNEBACK,
        "flow_magnitude": "sqrt(flow_x**2 + flow_y**2)",
        "strict_active_pixel_rule": f"magnitude > {FLOW_THRESHOLD} pixels",
        "individual_pair_rule": (
            "For each evaluated midpoint t and gap k, calculate a new flow "
            "field only between GT frame t-k/2 and GT frame t+k/2; threshold "
            "that field and box only its active pixels. No aggregation."
        ),
        "bbox_coordinates": (
            "minimum and maximum active x/y; maxima inclusive; box is "
            "overlaid on GT midpoint frame"
        ),
        "postprocessing": "none",
        "evaluated_individual_pairs": len(records),
        "active_pixel_percentage_range": [
            float(min(active_percentages)),
            float(max(active_percentages)),
        ],
        "bbox_area_percentage_range": [
            float(min(bbox_percentages)),
            float(max(bbox_percentages)),
        ],
        "bbox_area_percentage_median": float(np.median(bbox_percentages)),
        "reference_pair": records[(2, "cam00", 21)],
        "flow_noise_note": (
            "Any scattered active pixels are retained in min/max coordinate "
            "calculation; no filtering or threshold changes are applied."
        ),
    }
    with (boxes_dir / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    write_readme(output_dir, metadata)
    print(
        "All pair boxes: "
        f"active-pixel range={min(active_percentages):.3f}%.."
        f"{max(active_percentages):.3f}%, "
        f"bbox-area median={np.median(bbox_percentages):.2f}%, "
        f"full-frame boxes={sum(p == 100 for p in bbox_percentages)}"
    )
    return records


def save_overlay_montage(
    dataset_root: Path,
    diagnostic_dir: Path,
    records: dict[tuple[int, str, int], dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(18, 10), constrained_layout=True)
    for axis, camera in zip(axes.flat, CAMERAS):
        record = records[(2, camera, 21)]
        image_path = dataset_root / camera / "images" / "0021.png"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        ImageDraw.Draw(image).rectangle(
            (
                record["x_min"],
                record["y_min"],
                record["x_max_inclusive"],
                record["y_max_inclusive"],
            ),
            outline=(0, 255, 120),
            width=3,
        )
        axis.imshow(image)
        axis.set_title(
            f"{camera}: active={record['active_pixel_count']}\n"
            f"box={record['bbox_width']}x{record['bbox_height']}"
        )
        axis.axis("off")
    fig.suptitle("GT endpoint pair 20→22 boxes over midpoint frame 21")
    fig.savefig(diagnostic_dir / "cam_all_gap2_pair20_to22.png", dpi=180)
    plt.close(fig)


def load_boxes(
    boxes_path: Path,
) -> dict[tuple[int, str, int], dict[str, Any]]:
    if not boxes_path.is_file():
        raise FileNotFoundError(f"Missing per-sample boxes: {boxes_path}")
    records: dict[tuple[int, str, int], dict[str, Any]] = {}
    integer_columns = {
        "temporal_gap",
        "midpoint_frame",
        "source_frame",
        "target_frame",
        "active_pixel_count",
        "x_min",
        "y_min",
        "x_max_inclusive",
        "y_max_inclusive",
        "bbox_width",
        "bbox_height",
        "bbox_area_pixels",
    }
    with boxes_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            record: dict[str, Any] = {}
            for column in BOX_COLUMNS:
                if column == "camera" or column == "evaluation_setting":
                    record[column] = row[column]
                elif column in integer_columns:
                    record[column] = int(row[column])
                else:
                    record[column] = float(row[column])
            key = (
                record["temporal_gap"],
                record["camera"],
                record["midpoint_frame"],
            )
            if key in records:
                raise ValueError(f"Duplicate bbox row: {key}")
            if (
                record["source_frame"],
                record["target_frame"],
            ) != flow_endpoints(record["temporal_gap"], record["midpoint_frame"]):
                raise ValueError(f"Incorrect endpoint mapping in bbox row: {row}")
            if not (
                0 <= record["x_min"] <= record["x_max_inclusive"] < IMAGE_WIDTH
                and 0
                <= record["y_min"]
                <= record["y_max_inclusive"]
                < IMAGE_HEIGHT
            ):
                raise ValueError(f"Out-of-image bbox row: {row}")
            records[key] = record
    expected = sum(len(midpoint_frames(gap)) * len(CAMERAS) for gap in GAPS)
    if len(records) != expected:
        raise RuntimeError(f"Loaded {len(records)} boxes; expected {expected}")
    return records


def prediction_paths(
    project_root: Path, gap: int, camera: str, frame_index: int
) -> tuple[Path, Path, Path]:
    directory = (
        project_root
        / "output"
        / f"coffee_martini_k{gap}"
        / "midpoint_evaluation_v4"
        / f"gap_{gap}"
    )
    stem = (
        f"heldout_cam00_{frame_index:04d}"
        if camera == "cam00"
        else f"train_{camera}_{frame_index:04d}"
    )
    return (
        directory / f"{stem}_pred.png",
        directory / f"{stem}_gt.png",
        directory / "metrics.csv",
    )


def preflight_predictions(project_root: Path, dataset_root: Path) -> int:
    total = 0
    for gap in GAPS:
        found = {"training_cameras": 0, "heldout_camera": 0}
        for camera in TRAIN_CAMERAS:
            for frame in midpoint_frames(gap):
                pred, saved_gt, metrics = prediction_paths(
                    project_root, gap, camera, frame
                )
                source_gt = dataset_root / camera / "images" / f"{frame:04d}.png"
                for path in (pred, saved_gt, metrics, source_gt):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing required file: {path}")
                found["training_cameras"] += 1
        for frame in midpoint_frames(gap):
            pred, saved_gt, metrics = prediction_paths(
                project_root, gap, "cam00", frame
            )
            source_gt = (
                dataset_root / "cam00" / "images" / f"{frame:04d}.png"
            )
            for path in (pred, saved_gt, metrics, source_gt):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing required file: {path}")
            found["heldout_camera"] += 1
        if found != EXPECTED_COUNTS[gap]:
            raise RuntimeError(
                f"k={gap} counts {found}; expected {EXPECTED_COUNTS[gap]}"
            )
        total += sum(found.values())
        print(
            f"Preflight k={gap}: training={found['training_cameras']}, "
            f"heldout={found['heldout_camera']}"
        )
    if total != 756:
        raise RuntimeError(f"Expected 756 prediction samples, found {total}")
    return total


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def gaussian_window(
    size: int, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coordinate = torch.arange(size, device=device, dtype=dtype) - size // 2
    values = torch.exp(-(coordinate**2) / (2 * 1.5**2))
    values /= values.sum()
    kernel = values[:, None] @ values[None, :]
    return kernel[None, None].expand(channels, 1, size, size).contiguous()


def ssim_map(
    prediction: torch.Tensor, target: torch.Tensor, size: int = 11
) -> torch.Tensor:
    channels = prediction.shape[1]
    window = gaussian_window(size, channels, prediction.device, prediction.dtype)
    padding = size // 2
    mean_pred = F.conv2d(prediction, window, padding=padding, groups=channels)
    mean_target = F.conv2d(target, window, padding=padding, groups=channels)
    mean_pred_sq, mean_target_sq = mean_pred.square(), mean_target.square()
    mean_cross = mean_pred * mean_target
    var_pred = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mean_pred_sq
    )
    var_target = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mean_target_sq
    )
    covariance = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mean_cross
    )
    c1, c2 = 0.01**2, 0.03**2
    channel_map = ((2 * mean_cross + c1) * (2 * covariance + c2)) / (
        (mean_pred_sq + mean_target_sq + c1)
        * (var_pred + var_target + c2)
    )
    return channel_map.mean(dim=1, keepdim=True)


def compute_metrics(
    prediction_rgb: np.ndarray,
    target_rgb: np.ndarray,
    box: dict[str, Any],
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    if prediction_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: {prediction_rgb.shape} "
            f"!= {target_rgb.shape}"
        )
    if prediction_rgb.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Unexpected image size: {prediction_rgb.shape}")
    prediction = (
        torch.from_numpy(prediction_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    target = (
        torch.from_numpy(target_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    x0, y0 = box["x_min"], box["y_min"]
    x1, y1 = box["x_max_inclusive"], box["y_max_inclusive"]
    difference = prediction - target
    bbox_diff = difference[:, :, y0 : y1 + 1, x0 : x1 + 1]
    bbox_mse = bbox_diff.square().mean()
    full_mse = difference.square().mean()

    image_ssim = ssim_map(prediction, target)[0, 0]
    bbox_ssim = image_ssim[y0 : y1 + 1, x0 : x1 + 1].mean()
    lpips_map = lpips_model(
        prediction * 2 - 1,
        target * 2 - 1,
    )
    if lpips_map.shape != (1, 1, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Unexpected spatial LPIPS shape: {lpips_map.shape}")
    bbox_lpips = lpips_map[0, 0, y0 : y1 + 1, x0 : x1 + 1].mean()

    def psnr(mse: torch.Tensor) -> float:
        value = float(mse.item())
        return math.inf if value == 0 else -10 * math.log10(value)

    return {
        "PSNR_bbox": psnr(bbox_mse),
        "SSIM_bbox": float(bbox_ssim.item()),
        "LPIPS_bbox": float(bbox_lpips.item()),
        "PSNR_full": psnr(full_mse),
        "SSIM_full": float(image_ssim.mean().item()),
        "LPIPS_full": float(lpips_map.mean().item()),
    }


def evaluate_predictions(
    project_root: Path,
    dataset_root: Path,
    output_dir: Path,
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
        gap_total = sum(EXPECTED_COUNTS[gap].values())
        progress = tqdm(total=gap_total, desc=f"Evaluating gap {gap}")
        for setting, cameras in (
            ("training_cameras", TRAIN_CAMERAS),
            ("heldout_camera", ("cam00",)),
        ):
            for camera in cameras:
                for frame in midpoint_frames(gap):
                    box = records[(gap, camera, frame)]
                    pred_path, saved_gt_path, _ = prediction_paths(
                        project_root, gap, camera, frame
                    )
                    source_gt_path = (
                        dataset_root / camera / "images" / f"{frame:04d}.png"
                    )
                    prediction = load_rgb(pred_path)
                    target = load_rgb(source_gt_path)
                    saved_gt = load_rgb(saved_gt_path)
                    if not np.array_equal(target, saved_gt):
                        raise ValueError(
                            f"Saved GT differs from source image: {saved_gt_path}"
                        )
                    values = compute_metrics(
                        prediction, target, box, lpips_model, device
                    )
                    results.append(
                        {
                            "method": "4DGS",
                            "temporal_gap": gap,
                            "evaluation_setting": setting,
                            "camera": camera,
                            "frame_index": frame,
                            "source_frame": box["source_frame"],
                            "target_frame": box["target_frame"],
                            "active_pixel_count": box["active_pixel_count"],
                            "bbox_x_min": box["x_min"],
                            "bbox_y_min": box["y_min"],
                            "bbox_x_max_inclusive": box["x_max_inclusive"],
                            "bbox_y_max_inclusive": box["y_max_inclusive"],
                            "bbox_width": box["bbox_width"],
                            "bbox_height": box["bbox_height"],
                            "bbox_area_percentage": box["bbox_area_percentage"],
                            **values,
                        }
                    )
                    progress.update(1)
        progress.close()
        count = sum(row["temporal_gap"] == gap for row in results)
        if count != gap_total:
            raise RuntimeError(f"k={gap}: evaluated {count}; expected {gap_total}")
    if len(results) != 756:
        raise RuntimeError(f"Evaluated {len(results)} samples; expected 756")
    save_metrics(metrics_dir, results)
    save_metric_figures(figures_dir, results)
    return results


def save_metrics(metrics_dir: Path, results: list[dict[str, Any]]) -> None:
    save_csv(
        metrics_dir / "4dgs_bbox_reference_metrics.csv",
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
            expected = EXPECTED_COUNTS[gap][setting]
            if len(subset) != expected:
                raise RuntimeError(
                    f"{gap}/{setting}: {len(subset)} samples; expected {expected}"
                )
            for scope in ("bbox", "full"):
                means = [
                    float(np.mean([row[f"{metric}_{scope}"] for row in subset]))
                    for metric in ("PSNR", "SSIM", "LPIPS")
                ]
                summary.append(
                    ["4DGS", gap, setting, scope, *means, len(subset)]
                )
    save_csv(
        metrics_dir / "4dgs_bbox_reference_summary.csv",
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


def save_metric_figures(
    figures_dir: Path, results: list[dict[str, Any]]
) -> None:
    metrics = (
        ("PSNR", "PSNR (dB)"),
        ("SSIM", "SSIM"),
        ("LPIPS", "LPIPS distance"),
    )
    x = np.arange(len(GAPS))
    width = 0.18
    groups = (
        ("training_cameras", "full", "#4C78A8"),
        ("training_cameras", "bbox", "#F58518"),
        ("heldout_camera", "full", "#72B7B2"),
        ("heldout_camera", "bbox", "#E45756"),
    )
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
                label=f"{setting.replace('_', ' ')} / {scope}",
                color=color,
            )
        axis.set_xticks(x, [f"k={gap}" for gap in GAPS])
        axis.set_title(metric)
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=7)
    fig.suptitle("Endpoint-flow bbox vs full-image metrics")
    fig.savefig(figures_dir / "bbox_reference_vs_full.png", dpi=180)
    plt.close(fig)


def write_readme(output_dir: Path, metadata: dict[str, Any]) -> None:
    reference = metadata["reference_pair"]
    text = f"""# Endpoint-pair optical-flow bounding-box evaluation

This correction is implemented independently of the earlier
`dynamic_bbox_evaluation.py` and pixel-mask pipeline. Existing outputs are not
read as masks, modified, or overwritten. It reuses existing 4DGS midpoint
prediction PNGs without rendering or training.

## Exact per-sample procedure

For each 4DGS midpoint frame t and temporal gap k, load only the corresponding
ground-truth endpoint pair `(t-k/2, t+k/2)`. Calculate OpenCV Farneback flow
between that pair, compute `sqrt(flow_x**2 + flow_y**2)`, keep exactly pixels
whose magnitude is strictly greater than 0.5 pixels, then find min/max x/y of
those active pixels. Each pair gets an independent box; there is no temporal
aggregation or cross-frame union. Max x/y coordinates are inclusive. No
morphology or other active-pixel filtering is applied.

Farneback parameters: `{json.dumps(FARNEBACK, sort_keys=True)}`.
The box coordinates are shown over the ground-truth midpoint image in the
same camera coordinate system.

For the requested reference sample, cam00 GT pair 20->22 maps to midpoint 21:

- Active pixels: {reference['active_pixel_count']}
- Box: x=[{reference['x_min']},{reference['x_max_inclusive']}], \
y=[{reference['y_min']},{reference['y_max_inclusive']}]
- Dimensions: {reference['bbox_width']}x{reference['bbox_height']} pixels
- Box area: {reference['bbox_area_percentage']:.2f}% of the image

These are the exact results with the stated threshold and Farneback parameters.
Scattered active pixels are included in the min/max box by definition. They
were not removed to force a tighter box. Refer to
`diagnostics/cam00_gap2_pair20_to22.png`, which shows the flow magnitude, raw
threshold mask, and resulting overlay.

## Evaluation

The script evaluates all existing 4DGS gap 2, 4, and 8 midpoint predictions,
for 17 training cameras and held-out cam00 (756 samples expected). PSNR uses
all RGB values inside the inclusive box. SSIM is computed as a full-image
spatial map and averaged at box pixels; LPIPS uses its spatial VGG distance map
at the same coordinates. Full-image PSNR, SSIM, and LPIPS are computed
independently for comparison. The script validates expected files, source vs
saved GT equality, and sample counts before writing CSVs.

## Outputs

- `boxes/flow_boxes.csv`: one independent box for each sample endpoint pair.
- `boxes/debug_log.txt`: active counts and inclusive coordinates for every box.
- `diagnostics/cam00_gap2_pair20_to22.png`: magnitude, active mask, and overlay.
- `diagnostics/bbox_overlays/`: GT midpoint overlay for every camera/sample.
- `metrics/4dgs_bbox_reference_metrics.csv` and
  `metrics/4dgs_bbox_reference_summary.csv`: per-sample and grouped metrics.

Run this correction with a fresh output directory:

```bash
python dynamic_bbox_reference_evaluation.py --stage all
```

Default output directory: `output/dynamic_bbox_reference_v1`. Existing
directories are never overwritten.
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

    if args.stage in ("all", "boxes"):
        ensure_absent([args.output_dir])
        args.output_dir.mkdir(parents=True)
        records = generate_boxes(args.dataset_root, args.output_dir)
        if args.stage == "boxes":
            return
    else:
        records = load_boxes(args.output_dir / "boxes" / "flow_boxes.csv")

    results = evaluate_predictions(
        args.project_root,
        args.dataset_root,
        args.output_dir,
        records,
    )
    print(f"Evaluated {len(results)} midpoint samples.")
    print(f"Metrics saved to {args.output_dir / 'metrics'}")


if __name__ == "__main__":
    main()
