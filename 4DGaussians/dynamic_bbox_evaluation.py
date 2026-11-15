#!/usr/bin/env python3
"""Ground-truth optical-flow bounding boxes and 4DGS midpoint evaluation."""

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
FLOW_MAGNITUDE_THRESHOLD = 0.5
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-camera/frame ground-truth flow bounding boxes and "
            "evaluate existing 4DGS midpoint predictions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT.parent / "coffee_martini",
        help="Neural 3D Video coffee_martini dataset directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing existing 4DGS midpoint predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "dynamic_bbox_evaluation_v2",
        help="New output directory, separate from pixel-mask evaluation.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "boxes", "evaluate"),
        default="all",
        help="Generate all pair boxes, evaluate predictions, or both.",
    )
    return parser.parse_args()


def ensure_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def load_gray_frames(dataset_root: Path, camera: str) -> np.ndarray:
    image_dir = dataset_root / camera / "images"
    paths = [image_dir / f"{index:04d}.png" for index in range(FRAME_COUNT)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{camera} is missing {len(missing)} expected frames: "
            + ", ".join(missing[:5])
        )

    frames: list[np.ndarray] = []
    expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise OSError(f"Could not read ground-truth image: {path}")
        if frame.shape != expected_shape:
            raise ValueError(
                f"Unexpected dimensions for {path}: {frame.shape}, "
                f"expected {expected_shape}"
            )
        frames.append(frame)
    return np.stack(frames, axis=0)


def calculate_flow_box(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    flow = cv2.calcOpticalFlowFarneback(
        first,
        second,
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
    active_y, active_x = np.nonzero(magnitude > FLOW_MAGNITUDE_THRESHOLD)
    active_count = int(active_x.size)
    if not active_count:
        raise ValueError(
            "No pixels exceed the fixed flow threshold "
            f"{FLOW_MAGNITUDE_THRESHOLD} px for this frame pair"
        )
    x_min = int(active_x.min())
    y_min = int(active_y.min())
    x_max = int(active_x.max())
    y_max = int(active_y.max())
    return magnitude, (x_min, y_min, x_max, y_max), active_count


def save_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def draw_box_overlay(
    dataset_root: Path,
    camera: str,
    source_frame: int,
    box: tuple[int, int, int, int],
    path: Path,
) -> None:
    image_path = (
        dataset_root
        / camera
        / "images"
        / f"{source_frame:04d}.png"
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    x_min, y_min, x_max, y_max = box
    draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 0, 0), width=3)
    draw.text(
        (8, 8),
        f"{camera}: GT {source_frame:04d}->{source_frame + 1:04d}; "
        f"flow>{FLOW_MAGNITUDE_THRESHOLD:g}px",
        fill=(255, 255, 0),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    image.save(path)


def generate_boxes(
    dataset_root: Path, output_dir: Path
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    boxes_dir = output_dir / "boxes"
    overlays_dir = output_dir / "diagnostics" / "bbox_overlays"
    examples_dir = output_dir / "diagnostics" / "bbox_examples"
    ensure_absent(
        [
            boxes_dir,
            overlays_dir,
            examples_dir,
            output_dir / "diagnostics" / "bbox_area_by_camera.png",
            output_dir / "diagnostics" / "bbox_montage.png",
            output_dir / "README.md",
        ]
    )
    boxes_dir.mkdir()
    overlays_dir.mkdir(parents=True)
    examples_dir.mkdir()

    boxes: dict[tuple[str, int], dict[str, Any]] = {}
    rows: list[list[Any]] = []
    pair_motion: dict[str, list[tuple[int, float, int]]] = {}
    for camera in tqdm(CAMERAS, desc="Ground-truth flow boxes"):
        frames = load_gray_frames(dataset_root, camera)
        camera_dir = overlays_dir / camera
        camera_dir.mkdir()
        camera_pairs: list[tuple[int, float, int]] = []
        for source_frame in range(FRAME_COUNT - 1):
            magnitude, box, active_count = calculate_flow_box(
                frames[source_frame], frames[source_frame + 1]
            )
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            area = width * height
            record = {
                "camera": camera,
                "source_frame": source_frame,
                "target_frame": source_frame + 1,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": width,
                "height": height,
                "bbox_area_pixels": area,
                "bbox_area_percentage": 100.0
                * area
                / (IMAGE_WIDTH * IMAGE_HEIGHT),
                "active_pixel_count": active_count,
                "active_pixel_percentage": 100.0
                * active_count
                / (IMAGE_WIDTH * IMAGE_HEIGHT),
                "flow_magnitude_mean_px": float(magnitude.mean()),
                "flow_magnitude_max_px": float(magnitude.max()),
            }
            boxes[(camera, source_frame)] = record
            camera_pairs.append(
                (source_frame, float(magnitude.mean()), active_count)
            )
            rows.append([record[column] for column in BOX_COLUMNS])
            draw_box_overlay(
                dataset_root,
                camera,
                source_frame,
                box,
                camera_dir
                / f"frame_{source_frame:04d}_to_{source_frame + 1:04d}.png",
            )
        pair_motion[camera] = camera_pairs

    save_csv(boxes_dir / "flow_boxes.csv", BOX_COLUMNS, rows)
    save_box_diagnostics(boxes, pair_motion, output_dir, dataset_root)
    metadata = {
        "dataset": "Neural 3D Video / coffee_martini",
        "cameras": list(CAMERAS),
        "frames_per_camera": FRAME_COUNT,
        "pairs_per_camera": FRAME_COUNT - 1,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "flow_algorithm": "OpenCV Farneback dense optical flow",
        "flow_parameters": FARNEBACK,
        "flow_magnitude": "sqrt(flow_x^2 + flow_y^2), pixels/frame",
        "active_pixel_rule": (
            f"flow magnitude > {FLOW_MAGNITUDE_THRESHOLD} pixels/frame"
        ),
        "bounding_box": (
            "smallest axis-aligned box containing all active pixels; "
            "x_max and y_max inclusive"
        ),
        "empty_active_set": "raise an error; no pair was silently skipped",
        "midpoint_box_mapping": (
            "for midpoint prediction/GT frame t, use source-frame box from "
            "ground-truth pair t -> t+1"
        ),
        "postprocessing": "none",
        "generated_pair_count": len(boxes),
        "boxes_file": "boxes/flow_boxes.csv",
        "bbox_overlay_files": f"diagnostics/bbox_overlays/<camera>/{FRAME_COUNT - 1} PNGs",
    }
    with (boxes_dir / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    return boxes, metadata


BOX_COLUMNS = [
    "camera",
    "source_frame",
    "target_frame",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
    "bbox_area_pixels",
    "bbox_area_percentage",
    "active_pixel_count",
    "active_pixel_percentage",
    "flow_magnitude_mean_px",
    "flow_magnitude_max_px",
]


def save_box_diagnostics(
    boxes: dict[tuple[str, int], dict[str, Any]],
    pair_motion: dict[str, list[tuple[int, float, int]]],
    output_dir: Path,
    dataset_root: Path,
) -> None:
    diagnostics = output_dir / "diagnostics"
    area_values = [
        boxes[(camera, frame)]["bbox_area_percentage"]
        for camera in CAMERAS
        for frame in range(FRAME_COUNT - 1)
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    for camera in CAMERAS:
        values = [
            boxes[(camera, frame)]["bbox_area_percentage"]
            for frame in range(FRAME_COUNT - 1)
        ]
        ax.plot(range(FRAME_COUNT - 1), values, linewidth=1, label=camera)
    ax.set(
        title="Ground-truth flow bounding-box area by camera and frame pair",
        xlabel="Source frame index (pair t to t+1)",
        ylabel="Bounding-box area (% of image)",
    )
    ax.legend(ncol=6, fontsize=7)
    fig.tight_layout()
    fig.savefig(diagnostics / "bbox_area_by_camera.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 6, figsize=(18, 10), constrained_layout=True)
    for ax, camera in zip(axes.flat, CAMERAS):
        source_frame = max(
            pair_motion[camera],
            key=lambda row: row[2],
        )[0]
        record = boxes[(camera, source_frame)]
        image_path = (
            dataset_root
            / camera
            / "images"
            / f"{source_frame:04d}.png"
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        x_min, y_min = record["x_min"], record["y_min"]
        x_max, y_max = record["x_max"], record["y_max"]
        ImageDraw.Draw(image).rectangle(
            (x_min, y_min, x_max, y_max), outline=(255, 0, 0), width=3
        )
        ax.imshow(image)
        ax.set_title(
            f"{camera}, pair {source_frame:04d}->{source_frame + 1:04d}\n"
            f"active={record['active_pixel_count']}, "
            f"box={record['bbox_area_percentage']:.1f}%"
        )
        ax.axis("off")
        draw_box_overlay(
            dataset_root,
            camera,
            source_frame,
            (x_min, y_min, x_max, y_max),
            diagnostics
            / "bbox_examples"
            / f"{camera}_max_active_pair_{source_frame:04d}.png",
        )
    fig.suptitle("GT flow boxes on each camera's highest-activity pair")
    fig.savefig(diagnostics / "bbox_montage.png", dpi=180)
    plt.close(fig)

    if len(area_values) != len(CAMERAS) * (FRAME_COUNT - 1):
        raise RuntimeError("Not all per-camera flow boxes were generated")


def load_boxes(boxes_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not boxes_path.is_file():
        raise FileNotFoundError(f"Missing flow boxes CSV: {boxes_path}")
    boxes: dict[tuple[str, int], dict[str, Any]] = {}
    with boxes_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            camera = row["camera"]
            source_frame = int(row["source_frame"])
            if camera not in CAMERAS or not 0 <= source_frame < FRAME_COUNT - 1:
                raise ValueError(f"Invalid flow-box CSV row: {row}")
            record = {
                column: (
                    camera
                    if column == "camera"
                    else int(row[column])
                    if column
                    in {
                        "source_frame",
                        "target_frame",
                        "x_min",
                        "y_min",
                        "x_max",
                        "y_max",
                        "width",
                        "height",
                        "bbox_area_pixels",
                        "active_pixel_count",
                    }
                    else float(row[column])
                )
                for column in BOX_COLUMNS
            }
            key = (camera, source_frame)
            if key in boxes:
                raise ValueError(f"Duplicate flow-box row: {key}")
            boxes[key] = record
    expected = len(CAMERAS) * (FRAME_COUNT - 1)
    if len(boxes) != expected:
        raise RuntimeError(
            f"Loaded {len(boxes)} flow boxes, expected {expected}"
        )
    for (camera, source_frame), box in boxes.items():
        if box["target_frame"] != source_frame + 1:
            raise ValueError(f"Non-consecutive flow pair: {(camera, box)}")
        if not (
            0 <= box["x_min"] <= box["x_max"] < IMAGE_WIDTH
            and 0 <= box["y_min"] <= box["y_max"] < IMAGE_HEIGHT
        ):
            raise ValueError(f"Out-of-image bounding box: {(camera, box)}")
    return boxes


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
        frames = list(range(gap // 2, 49, gap))
        observed = {"training_cameras": 0, "heldout_camera": 0}
        for camera in TRAIN_CAMERAS:
            for frame_index in frames:
                paths = prediction_paths(project_root, gap, camera, frame_index)
                source_gt = (
                    dataset_root / camera / "images" / f"{frame_index:04d}.png"
                )
                for path in (*paths, source_gt):
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Missing required gap-{gap} sample file: {path}"
                        )
                observed["training_cameras"] += 1
        for frame_index in frames:
            paths = prediction_paths(project_root, gap, "cam00", frame_index)
            source_gt = (
                dataset_root
                / "cam00"
                / "images"
                / f"{frame_index:04d}.png"
            )
            for path in (*paths, source_gt):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing required gap-{gap} sample file: {path}"
                    )
            observed["heldout_camera"] += 1
        if observed != EXPECTED_COUNTS[gap]:
            raise RuntimeError(
                f"Gap {gap}: found counts {observed}; "
                f"expected {EXPECTED_COUNTS[gap]}"
            )
        total += sum(observed.values())
        print(
            f"Preflight k={gap}: {observed['training_cameras']} training, "
            f"{observed['heldout_camera']} held-out"
        )
    if total != 756:
        raise RuntimeError(f"Found {total} expected samples, required 756")
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
    prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    channels = prediction.shape[1]
    window = gaussian_window(
        window_size, channels, prediction.device, prediction.dtype
    )
    padding = window_size // 2
    mu_prediction = F.conv2d(
        prediction, window, padding=padding, groups=channels
    )
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    variance_prediction = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mu_prediction_sq
    )
    variance_target = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mu_target_sq
    )
    covariance = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mu_product
    )
    c1, c2 = 0.01**2, 0.03**2
    channel_map = ((2 * mu_product + c1) * (2 * covariance + c2)) / (
        (mu_prediction_sq + mu_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return channel_map.mean(dim=1, keepdim=True)


def bbox_and_full_metrics(
    prediction_rgb: np.ndarray,
    target_rgb: np.ndarray,
    box: dict[str, Any],
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    if prediction_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: "
            f"{prediction_rgb.shape} vs {target_rgb.shape}"
        )
    if prediction_rgb.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Unexpected image dimensions: {prediction_rgb.shape}")

    x_min, x_max = int(box["x_min"]), int(box["x_max"])
    y_min, y_max = int(box["y_min"]), int(box["y_max"])
    bbox_pixels = (x_max - x_min + 1) * (y_max - y_min + 1)
    if bbox_pixels <= 0:
        raise ValueError(f"Invalid empty bbox: {box}")

    prediction = (
        torch.from_numpy(prediction_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    target = (
        torch.from_numpy(target_rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    difference = prediction - target
    full_mse = difference.square().mean()
    bbox_difference = difference[
        :, :, y_min : y_max + 1, x_min : x_max + 1
    ]
    bbox_mse = bbox_difference.square().mean()

    image_ssim = ssim_map(prediction, target)[0, 0]
    bbox_ssim = image_ssim[y_min : y_max + 1, x_min : x_max + 1].mean()

    spatial_lpips = lpips_model(
        prediction.mul(2).sub(1),
        target.mul(2).sub(1),
    )
    if spatial_lpips.ndim != 4 or spatial_lpips.shape[:2] != (1, 1):
        raise ValueError(f"Unexpected spatial LPIPS output: {spatial_lpips.shape}")
    if spatial_lpips.shape[-2:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(
            "LPIPS spatial output is not aligned with the image: "
            f"{spatial_lpips.shape[-2:]}"
        )
    bbox_lpips = spatial_lpips[
        0, 0, y_min : y_max + 1, x_min : x_max + 1
    ].mean()

    def psnr(mse: torch.Tensor) -> float:
        value = float(mse.item())
        return math.inf if value == 0 else -10.0 * math.log10(value)

    return {
        "PSNR_bbox": psnr(bbox_mse),
        "SSIM_bbox": float(bbox_ssim.item()),
        "LPIPS_bbox": float(bbox_lpips.item()),
        "PSNR_full": psnr(full_mse),
        "SSIM_full": float(image_ssim.mean().item()),
        "LPIPS_full": float(spatial_lpips.mean().item()),
    }


def evaluate_predictions(
    project_root: Path,
    dataset_root: Path,
    output_dir: Path,
    boxes: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    import lpips

    expected_total = preflight_predictions(project_root, dataset_root)
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
        frame_indices = list(range(gap // 2, 49, gap))
        gap_expected = sum(EXPECTED_COUNTS[gap].values())
        progress = tqdm(total=gap_expected, desc=f"Evaluating k={gap}")
        for setting, cameras in (
            ("training_cameras", TRAIN_CAMERAS),
            ("heldout_camera", ("cam00",)),
        ):
            for camera in cameras:
                for frame_index in frame_indices:
                    key = (camera, frame_index)
                    if key not in boxes:
                        raise KeyError(
                            f"No GT flow bbox for evaluation frame {key}; "
                            f"expected pair {frame_index}->{frame_index + 1}"
                        )
                    box = boxes[key]
                    pred_path, eval_gt_path, _ = prediction_paths(
                        project_root, gap, camera, frame_index
                    )
                    source_gt_path = (
                        dataset_root
                        / camera
                        / "images"
                        / f"{frame_index:04d}.png"
                    )
                    prediction = load_rgb(pred_path)
                    target = load_rgb(source_gt_path)
                    saved_gt = load_rgb(eval_gt_path)
                    if not np.array_equal(target, saved_gt):
                        raise ValueError(
                            f"Saved midpoint GT differs from source: "
                            f"{eval_gt_path}"
                        )
                    metrics = bbox_and_full_metrics(
                        prediction, target, box, lpips_model, device
                    )
                    bbox_pixels = int(box["bbox_area_pixels"])
                    results.append(
                        {
                            "method": "4DGS",
                            "temporal_gap": gap,
                            "evaluation_setting": setting,
                            "camera": camera,
                            "frame_index": frame_index,
                            "flow_pair_source_frame": frame_index,
                            "flow_pair_target_frame": frame_index + 1,
                            "flow_threshold_pixels": FLOW_MAGNITUDE_THRESHOLD,
                            "bbox_x_min": box["x_min"],
                            "bbox_y_min": box["y_min"],
                            "bbox_x_max_inclusive": box["x_max"],
                            "bbox_y_max_inclusive": box["y_max"],
                            "bbox_width": box["width"],
                            "bbox_height": box["height"],
                            "bbox_pixel_count": bbox_pixels,
                            "bbox_pixel_percentage": box["bbox_area_percentage"],
                            "flow_active_pixel_count": box["active_pixel_count"],
                            **metrics,
                        }
                    )
                    progress.update(1)
        progress.close()
        actual = sum(row["temporal_gap"] == gap for row in results)
        if actual != gap_expected:
            raise RuntimeError(
                f"k={gap}: evaluated {actual}, expected {gap_expected}"
            )

    if len(results) != expected_total:
        raise RuntimeError(
            f"Evaluated {len(results)} samples; expected {expected_total}"
        )
    write_metric_csvs(metrics_dir, results)
    write_metric_figures(figures_dir, results)
    return results


def write_metric_csvs(metrics_dir: Path, results: list[dict[str, Any]]) -> None:
    columns = [
        "method",
        "temporal_gap",
        "evaluation_setting",
        "camera",
        "frame_index",
        "flow_pair_source_frame",
        "flow_pair_target_frame",
        "flow_threshold_pixels",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_x_max_inclusive",
        "bbox_y_max_inclusive",
        "bbox_width",
        "bbox_height",
        "bbox_pixel_count",
        "bbox_pixel_percentage",
        "flow_active_pixel_count",
        "PSNR_bbox",
        "SSIM_bbox",
        "LPIPS_bbox",
        "PSNR_full",
        "SSIM_full",
        "LPIPS_full",
    ]
    save_csv(
        metrics_dir / "4dgs_bbox_metrics.csv",
        columns,
        [[row[column] for column in columns] for row in results],
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
                    f"Summary group {(gap, setting)} has {len(subset)} "
                    f"samples; expected {expected}"
                )
            for scope in ("bbox", "full"):
                names = (
                    f"PSNR_{scope}",
                    f"SSIM_{scope}",
                    f"LPIPS_{scope}",
                )
                means = [
                    float(np.mean([row[name] for row in subset]))
                    for name in names
                ]
                summary.append(
                    ["4DGS", gap, setting, scope, *means, len(subset)]
                )
    save_csv(
        metrics_dir / "4dgs_bbox_summary.csv",
        [
            "method",
            "temporal_gap",
            "evaluation_setting",
            "metric_scope",
            "mean_PSNR",
            "mean_SSIM",
            "mean_LPIPS",
            "num_evaluated_samples",
        ],
        summary,
    )


def write_metric_figures(
    figures_dir: Path, results: list[dict[str, Any]]
) -> None:
    metrics = (
        ("PSNR", "PSNR (dB)"),
        ("SSIM", "SSIM"),
        ("LPIPS", "LPIPS distance"),
    )
    x = np.arange(len(GAPS))
    width = 0.18
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    series = (
        ("training_cameras", "full", "#4C78A8"),
        ("training_cameras", "bbox", "#F58518"),
        ("heldout_camera", "full", "#72B7B2"),
        ("heldout_camera", "bbox", "#E45756"),
    )
    for axis, (metric, ylabel) in zip(axes, metrics):
        for index, (setting, scope, color) in enumerate(series):
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
    fig.suptitle("Bounding-box region and full-image evaluation")
    fig.savefig(figures_dir / "bbox_vs_full_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for setting, color, marker in (
            ("training_cameras", "#4C78A8", "o"),
            ("heldout_camera", "#E45756", "s"),
        ):
            values = []
            for gap in GAPS:
                sample_values = [
                    row[f"{metric}_bbox"]
                    for row in results
                    if row["temporal_gap"] == gap
                    and row["evaluation_setting"] == setting
                ]
                values.append(float(np.mean(sample_values)))
            axis.plot(
                GAPS,
                values,
                color=color,
                marker=marker,
                label=setting.replace("_", " "),
            )
        axis.set_xticks(GAPS, [f"k={gap}" for gap in GAPS])
        axis.set_title(metric)
        axis.set_ylabel(ylabel)
        axis.legend()
    fig.suptitle("Bounding-box metrics by temporal gap")
    fig.savefig(figures_dir / "bbox_gap_comparison.png", dpi=180)
    plt.close(fig)


def write_readme(
    output_dir: Path,
    boxes: dict[tuple[str, int], dict[str, Any]],
) -> None:
    active_percentages = np.array(
        [record["active_pixel_percentage"] for record in boxes.values()]
    )
    bbox_percentages = np.array(
        [record["bbox_area_percentage"] for record in boxes.values()]
    )
    text = f"""# Ground-truth flow bounding-box evaluation

This pipeline is independent of `dynamic_region_evaluation.py` and its
pixel-level masks, output files, and metrics. It reads the saved 4DGS midpoint
prediction PNGs and source ground-truth frames; it does not rerender or retrain.

## Reproducibility

Run in the project environment from the project directory, selecting a fresh
output directory each time:

```bash
python dynamic_bbox_evaluation.py \\
  --output-dir output/dynamic_bbox_evaluation_v3 \\
  --stage all
```

The script refuses to reuse an output directory that already exists.

Default dataset: `../coffee_martini`. Predictions are read from
`output/coffee_martini_k{{2,4,8}}/midpoint_evaluation_v4/gap_{{2,4,8}}`.
All 18 cameras and all 49 consecutive GT pairs per camera are processed.

## Bounding-box construction

- Grayscale source images: GT frames `0000.png` through `0049.png`, 676 x 507.
- Optical flow: OpenCV Farneback (`pyr_scale=0.5`, `levels=3`,
  `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, `flags=0`).
- Flow magnitude in pixels/frame: `sqrt(flow_x**2 + flow_y**2)`.
- Active pixel rule: magnitude strictly greater than {FLOW_MAGNITUDE_THRESHOLD}
  pixels/frame.
- For each camera and pair t -> t+1, the reported box is the smallest
  axis-aligned rectangle enclosing every active pixel in the source-frame flow
  map. Max coordinates are inclusive. No morphology, filtering, or clipping
  beyond the image bounds is applied. A pair with no active pixels is an error,
  not silently skipped.
- A midpoint image at frame t uses the GT flow box from pair t -> t+1. This keeps
  the box in the midpoint GT image's source coordinate system and applies the
  same deterministic rule across all gaps.
- Boxes and active-pixel counts for all 18 x 49 pairs are stored in
  `boxes/flow_boxes.csv`; GT overlays for every pair are in
  `diagnostics/bbox_overlays/`.

Across all 882 pairs, active pixels range from
{active_percentages.min():.4f}% to {active_percentages.max():.4f}% of the image;
their enclosing box areas range from {bbox_percentages.min():.4f}% to
{bbox_percentages.max():.4f}%.

## Metrics

- Before evaluation, every expected prediction, evaluation GT, source GT, and
  existing metrics CSV is checked. Saved evaluation GT must match the source
  frame pixel-for-pixel. Expected counts are k=2: 408 training + 24 held-out,
  k=4: 204 + 12, k=8: 102 + 6 (756 total).
- RGB images are converted to float in [0,1]. Bounding-box PSNR is calculated
  from mean squared error over all RGB values inside the inclusive box.
- SSIM uses the repository-compatible 11x11 Gaussian window (sigma 1.5),
  C1=0.01^2 and C2=0.03^2. The spatial SSIM map is computed on the full image;
  its pixel values whose centers lie inside the box are averaged. Image pixels
  outside the box are not zeroed.
- LPIPS uses pretrained VGG LPIPS with spatial output. The package's spatial
  distance map is upsampled to the original image resolution, and values inside
  the same inclusive box are averaged. Full LPIPS is the mean over the full map.
- Full PSNR, SSIM, and LPIPS are computed independently over the full image.
  Summary metrics are arithmetic means of per-sample values.

## Limitations

An axis-aligned box includes static pixels between and around active pixels, and
can become large when flow noise or camera motion spreads threshold crossings.
The fixed 0.5-pixel threshold defines motion activity, not semantic foreground.
The boxes are GT-derived and method-independent. No second-method predictions
are included in this run. Per-sample metrics and aggregated metrics are stored
in `metrics/4dgs_bbox_metrics.csv` and `metrics/4dgs_bbox_summary.csv` after
the evaluation stage completes.
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

    if args.stage in ("boxes", "all"):
        ensure_absent([args.output_dir])
        args.output_dir.mkdir(parents=True)
        boxes, _ = generate_boxes(args.dataset_root, args.output_dir)
        if args.stage == "boxes":
            write_readme(args.output_dir, boxes)
        print(
            f"Saved {len(boxes)} camera/frame-pair boxes to "
            f"{args.output_dir / 'boxes' / 'flow_boxes.csv'}"
        )
        if args.stage == "boxes":
            return
    else:
        boxes = load_boxes(args.output_dir / "boxes" / "flow_boxes.csv")

    results = evaluate_predictions(
        args.project_root,
        args.dataset_root,
        args.output_dir,
        boxes,
    )
    write_readme(args.output_dir, boxes)
    print(f"Evaluated {len(results)} samples.")
    print(f"Metrics: {args.output_dir / 'metrics'}")


if __name__ == "__main__":
    main()
