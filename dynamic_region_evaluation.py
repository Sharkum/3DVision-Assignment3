#!/usr/bin/env python3
"""Ground-truth-only dynamic masks and dynamic-region evaluation for 4DGS."""

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
IMAGE_WIDTH = 676
IMAGE_HEIGHT = 507
FRAME_COUNT = 50
OTSU_HISTOGRAM_BINS = 4096
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
            "Generate a single ground-truth-derived dynamic mask per camera "
            "and evaluate existing 4DGS midpoint predictions."
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
        help="4DGaussians project directory containing output/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "dynamic_region_evaluation_v1",
        help="Fresh result directory; existing masks/metrics are never replaced.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "masks", "evaluate"),
        default="all",
        help="Run mask generation, evaluation, or both in sequence.",
    )
    return parser.parse_args()


def ensure_new_paths(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def load_gray_frames(dataset_root: Path, camera: str) -> np.ndarray:
    image_dir = dataset_root / camera / "images"
    expected_names = [f"{index:04d}.png" for index in range(FRAME_COUNT)]
    paths = [image_dir / name for name in expected_names]
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
            raise OSError(f"OpenCV could not read ground-truth image: {path}")
        if frame.shape != expected_shape:
            raise ValueError(
                f"Unexpected image size for {path}: {frame.shape}, "
                f"expected {expected_shape}"
            )
        frames.append(frame)
    return np.stack(frames, axis=0)


def calculate_motion(
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion_sum = np.zeros(frames.shape[1:], dtype=np.float64)
    pair_means = np.empty(FRAME_COUNT - 1, dtype=np.float64)
    for index in range(FRAME_COUNT - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[index],
            frames[index + 1],
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
        motion_sum += magnitude
        pair_means[index] = float(magnitude.mean())

    motion_score = (motion_sum / (FRAME_COUNT - 1)).astype(np.float32)
    intensity_variation = frames.astype(np.float32).std(axis=0, ddof=0)
    return motion_score, intensity_variation, pair_means


def otsu_threshold(
    score_maps: dict[str, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    maximum = max(float(score.max()) for score in score_maps.values())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError(f"Invalid global motion-score maximum: {maximum}")

    histogram = np.zeros(OTSU_HISTOGRAM_BINS, dtype=np.int64)
    for score in score_maps.values():
        counts, _ = np.histogram(
            score,
            bins=OTSU_HISTOGRAM_BINS,
            range=(0.0, maximum),
        )
        histogram += counts

    edges = np.linspace(0.0, maximum, OTSU_HISTOGRAM_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    counts = histogram.astype(np.float64)
    cumulative_count = np.cumsum(counts)
    cumulative_sum = np.cumsum(counts * centers)
    total_count = cumulative_count[-1]
    total_sum = cumulative_sum[-1]

    left_count = cumulative_count[:-1]
    right_count = total_count - left_count
    valid = (left_count > 0) & (right_count > 0)
    between_class_variance = np.full(OTSU_HISTOGRAM_BINS - 1, -np.inf)
    left_mean = np.zeros_like(left_count)
    right_mean = np.zeros_like(right_count)
    left_mean[valid] = cumulative_sum[:-1][valid] / left_count[valid]
    right_mean[valid] = (
        total_sum - cumulative_sum[:-1][valid]
    ) / right_count[valid]
    between_class_variance[valid] = (
        left_count[valid]
        * right_count[valid]
        * (left_mean[valid] - right_mean[valid]) ** 2
    )

    threshold_index = int(np.argmax(between_class_variance))
    threshold = float(centers[threshold_index])
    return threshold, histogram, edges


def save_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def colorize_motion(score: np.ndarray, vmax: float | None = None) -> np.ndarray:
    if vmax is None:
        vmax = float(np.percentile(score, 99.5))
    normalized = np.clip(score / max(vmax, 1e-8), 0.0, 1.0)
    color = matplotlib.colormaps["turbo"](normalized)[..., :3]
    return (color * 255).astype(np.uint8)


def save_mask_diagnostics(
    dataset_root: Path,
    output_dir: Path,
    score_maps: dict[str, np.ndarray],
    intensity_maps: dict[str, np.ndarray],
    pair_means: dict[str, np.ndarray],
    threshold: float,
    histogram: np.ndarray,
    edges: np.ndarray,
) -> dict[str, float]:
    masks_root = output_dir / "masks"
    diagnostics_root = output_dir / "diagnostics"
    ensure_new_paths(
        [
            masks_root,
            diagnostics_root / "motion_score_distribution.png",
            diagnostics_root / "dynamic_pixel_percentage.png",
            diagnostics_root / "mask_overlays",
            diagnostics_root / "all_camera_masks.png",
            diagnostics_root / "motion_score_distribution.csv",
            diagnostics_root / "camera_motion_statistics.csv",
        ]
    )
    masks_root.mkdir()
    (diagnostics_root / "mask_overlays").mkdir(parents=True)

    camera_stats: dict[str, float] = {}
    statistics_rows: list[list[Any]] = []
    percentages: list[float] = []
    all_scores = np.concatenate(
        [score_maps[camera].reshape(-1) for camera in CAMERAS]
    )

    for camera in CAMERAS:
        score = score_maps[camera]
        mask = score > threshold
        dynamic_count = int(mask.sum())
        dynamic_percentage = 100.0 * dynamic_count / mask.size
        camera_stats[camera] = dynamic_percentage
        percentages.append(dynamic_percentage)
        camera_dir = masks_root / camera
        camera_dir.mkdir()
        np.save(camera_dir / "motion_score.npy", score, allow_pickle=False)
        np.save(
            camera_dir / "intensity_variation.npy",
            intensity_maps[camera],
            allow_pickle=False,
        )
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            camera_dir / "dynamic_mask.png"
        )

        frame_index = int(np.argmax(pair_means[camera])) + 1
        rgb = np.asarray(
            Image.open(
                dataset_root
                / camera
                / "images"
                / f"{frame_index:04d}.png"
            ).convert("RGB")
        )
        overlay = rgb.astype(np.float32)
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * red[mask]
        overlay_path = diagnostics_root / "mask_overlays" / f"{camera}.png"
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
            overlay_path
        )

        statistics_rows.append(
            [
                camera,
                float(score.min()),
                float(np.percentile(score, 50)),
                float(np.percentile(score, 90)),
                float(np.percentile(score, 95)),
                float(np.percentile(score, 99)),
                float(score.max()),
                float(score.mean()),
                dynamic_count,
                dynamic_percentage,
                frame_index,
            ]
        )

    save_csv(
        diagnostics_root / "camera_motion_statistics.csv",
        [
            "camera",
            "motion_min_px_per_frame",
            "motion_p50_px_per_frame",
            "motion_p90_px_per_frame",
            "motion_p95_px_per_frame",
            "motion_p99_px_per_frame",
            "motion_max_px_per_frame",
            "motion_mean_px_per_frame",
            "dynamic_pixel_count",
            "dynamic_pixel_percentage",
            "overlay_frame_index",
        ],
        statistics_rows,
    )
    save_csv(
        diagnostics_root / "motion_score_distribution.csv",
        ["bin_left_px_per_frame", "bin_right_px_per_frame", "pixel_count"],
        [
            [float(edges[index]), float(edges[index + 1]), int(histogram[index])]
            for index in range(len(histogram))
        ],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_scores, bins=512, log=True, color="#4472C4")
    ax.axvline(
        threshold,
        color="#D62728",
        linestyle="--",
        label=f"Global Otsu threshold = {threshold:.6f} px/frame",
    )
    ax.set(
        title="Pooled mean optical-flow magnitude (all 18 cameras)",
        xlabel="Mean flow magnitude (pixels/frame)",
        ylabel="Pixel count (log scale)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(diagnostics_root / "motion_score_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(CAMERAS, percentages, color="#4472C4")
    ax.set(
        title="Dynamic pixels per camera using the shared global threshold",
        ylabel="Dynamic pixels (%)",
        ylim=(0, 100),
    )
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(diagnostics_root / "dynamic_pixel_percentage.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 6, figsize=(15, 8), constrained_layout=True)
    for ax, camera in zip(axes.flat, CAMERAS):
        mask = score_maps[camera] > threshold
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(camera)
        ax.axis("off")
    fig.suptitle("Per-camera masks (shared Otsu threshold; no morphology)")
    fig.savefig(diagnostics_root / "all_camera_masks.png", dpi=180)
    plt.close(fig)

    save_flow_examples(
        dataset_root,
        diagnostics_root / "optical_flow_examples",
        score_maps["cam00"],
        pair_means["cam00"],
    )

    metadata = {
        "dataset": "Neural 3D Video / coffee_martini",
        "camera_names": list(CAMERAS),
        "training_camera_names": list(TRAIN_CAMERAS),
        "heldout_camera": "cam00",
        "frame_count": FRAME_COUNT,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "flow_algorithm": "OpenCV Farneback dense optical flow",
        "flow_parameters": FARNEBACK,
        "flow_magnitude": "sqrt(flow_x^2 + flow_y^2), pixels/frame",
        "motion_score": "arithmetic mean over all 49 consecutive frame pairs",
        "threshold_method": "pooled Otsu on a 4096-bin global motion-score histogram",
        "threshold_value_px_per_frame": threshold,
        "mask_definition": "motion_score > threshold",
        "postprocessing": "none",
        "intensity_variation": (
            "population standard deviation (ddof=0) of 8-bit grayscale "
            "ground-truth frames; diagnostic only"
        ),
        "dynamic_pixel_percentage_by_camera": camera_stats,
        "pooled_motion_score_percentiles_px_per_frame": {
            str(percentile): float(np.percentile(all_scores, percentile))
            for percentile in (0, 25, 50, 75, 90, 95, 99, 99.9, 100)
        },
        "mask_source": "ground-truth frames only; model prediction files not read",
    }
    with (masks_root / "threshold_metadata.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    return camera_stats


def save_flow_examples(
    dataset_root: Path,
    output_dir: Path,
    cam00_score: np.ndarray,
    pair_means: np.ndarray,
) -> None:
    output_dir.mkdir(exist_ok=True)
    ensure_new_paths(
        [
            output_dir / "cam00_top_motion_pairs_global.png",
            output_dir / "cam00_mean_motion_score.png",
        ]
    )
    frames = load_gray_frames(dataset_root, "cam00")
    indices = np.argsort(pair_means)[-3:][::-1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, index in enumerate(indices):
        axes[0, column].imshow(frames[index], cmap="gray")
        axes[0, column].set_title(f"GT cam00 frame {index:04d}")
        flow = cv2.calcOpticalFlowFarneback(
            frames[index],
            frames[index + 1],
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
        vmax = max(float(np.percentile(magnitude, 99.5)), 1e-8)
        axes[1, column].imshow(magnitude, cmap="turbo", vmin=0, vmax=vmax)
        axes[1, column].set_title(
            f"{index:04d} to {index + 1:04d}; "
            f"mean={pair_means[index]:.3f} px"
        )
        for axis in (axes[0, column], axes[1, column]):
            axis.axis("off")
    fig.savefig(output_dir / "cam00_top_motion_pairs_global.png", dpi=180)
    plt.close(fig)

    score_vmax = max(float(np.percentile(cam00_score, 99.5)), 1e-8)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(cam00_score, cmap="magma", vmin=0, vmax=score_vmax)
    ax.set_title("cam00 mean motion score over 49 pairs")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "cam00_mean_motion_score.png", dpi=180)
    plt.close(fig)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def prediction_paths(
    project_root: Path, gap: int, camera: str, frame_index: int
) -> tuple[Path, Path, Path]:
    evaluation_root = (
        project_root
        / "output"
        / f"coffee_martini_k{gap}"
        / "midpoint_evaluation_v4"
        / f"gap_{gap}"
    )
    if camera == "cam00":
        stem = f"heldout_cam00_{frame_index:04d}"
    else:
        stem = f"train_{camera}_{frame_index:04d}"
    return (
        evaluation_root / f"{stem}_pred.png",
        evaluation_root / f"{stem}_gt.png",
        evaluation_root / "metrics.csv",
    )


def preflight_predictions(project_root: Path, dataset_root: Path) -> int:
    expected_total = 0
    for gap in GAPS:
        frames = list(range(gap // 2, 49, gap))
        expected_by_setting = EXPECTED_COUNTS[gap]
        setting_counts = {"training_cameras": 0, "heldout_camera": 0}
        for camera in TRAIN_CAMERAS:
            for frame_index in frames:
                pred_path, eval_gt_path, metrics_path = prediction_paths(
                    project_root, gap, camera, frame_index
                )
                gt_path = (
                    dataset_root
                    / camera
                    / "images"
                    / f"{frame_index:04d}.png"
                )
                for path in (pred_path, eval_gt_path, metrics_path, gt_path):
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Missing required gap-{gap} training sample file: "
                            f"{path}"
                        )
                setting_counts["training_cameras"] += 1
        for frame_index in frames:
            pred_path, eval_gt_path, metrics_path = prediction_paths(
                project_root, gap, "cam00", frame_index
            )
            gt_path = (
                dataset_root
                / "cam00"
                / "images"
                / f"{frame_index:04d}.png"
            )
            for path in (pred_path, eval_gt_path, metrics_path, gt_path):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing required gap-{gap} held-out sample file: "
                        f"{path}"
                    )
            setting_counts["heldout_camera"] += 1

        if setting_counts != expected_by_setting:
            raise RuntimeError(
                f"Gap {gap} sample count mismatch: {setting_counts}; "
                f"expected {expected_by_setting}"
            )
        expected_total += sum(setting_counts.values())
        print(
            f"Preflight gap {gap}: "
            f"{setting_counts['training_cameras']} training + "
            f"{setting_counts['heldout_camera']} held-out samples"
        )

    if expected_total != 756:
        raise RuntimeError(f"Expected 756 total samples, got {expected_total}")
    return expected_total


def gaussian_window(
    window_size: int, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coordinates = torch.arange(window_size, dtype=dtype, device=device)
    coordinates -= window_size // 2
    one_dimensional = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    one_dimensional /= one_dimensional.sum()
    two_dimensional = one_dimensional[:, None] @ one_dimensional[None, :]
    return two_dimensional[None, None].expand(
        channels, 1, window_size, window_size
    ).contiguous()


def ssim_map(
    prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    channels = prediction.shape[1]
    window = gaussian_window(
        window_size, channels, prediction.device, prediction.dtype
    )
    padding = window_size // 2
    mean_prediction = F.conv2d(
        prediction, window, padding=padding, groups=channels
    )
    mean_target = F.conv2d(target, window, padding=padding, groups=channels)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_product = mean_prediction * mean_target
    variance_prediction = (
        F.conv2d(prediction * prediction, window, padding=padding, groups=channels)
        - mean_prediction_sq
    )
    variance_target = (
        F.conv2d(target * target, window, padding=padding, groups=channels)
        - mean_target_sq
    )
    covariance = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mean_product
    )
    c1 = 0.01**2
    c2 = 0.03**2
    values = ((2 * mean_product + c1) * (2 * covariance + c2)) / (
        (mean_prediction_sq + mean_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return values.mean(dim=1, keepdim=True)


def masked_and_full_metrics(
    prediction_rgb: np.ndarray,
    target_rgb: np.ndarray,
    mask_array: np.ndarray,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    if prediction_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"Prediction/ground-truth shape mismatch: "
            f"{prediction_rgb.shape} vs {target_rgb.shape}"
        )
    if mask_array.shape != prediction_rgb.shape[:2]:
        raise ValueError(
            f"Mask/image shape mismatch: {mask_array.shape} vs "
            f"{prediction_rgb.shape[:2]}"
        )
    if not mask_array.any():
        raise ValueError("Dynamic mask contains no selected pixels")

    prediction = (
        torch.from_numpy(prediction_rgb).permute(2, 0, 1).unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    target = (
        torch.from_numpy(target_rgb).permute(2, 0, 1).unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    mask = torch.from_numpy(mask_array).to(device=device, dtype=torch.bool)

    difference = prediction - target
    full_mse = difference.square().mean()
    dynamic_mse = difference[0, :, mask].square().mean()

    full_ssim_map = ssim_map(prediction, target)[0, 0]
    dynamic_ssim = full_ssim_map[mask].mean()

    prediction_lpips = prediction.mul(2).sub(1)
    target_lpips = target.mul(2).sub(1)
    lpips_map = lpips_model(prediction_lpips, target_lpips)
    if lpips_map.ndim != 4 or lpips_map.shape[:2] != (1, 1):
        raise ValueError(f"Unexpected spatial LPIPS output: {lpips_map.shape}")
    if lpips_map.shape[-2:] != mask.shape:
        mask_for_lpips = F.interpolate(
            mask[None, None].float(),
            size=lpips_map.shape[-2:],
            mode="nearest",
        )[0, 0].bool()
    else:
        mask_for_lpips = mask
    if not mask_for_lpips.any():
        raise ValueError("Resized dynamic mask contains no LPIPS feature pixels")
    full_lpips = lpips_map.mean()
    dynamic_lpips = lpips_map[0, 0][mask_for_lpips].mean()

    def psnr_from_mse(mse: torch.Tensor) -> float:
        value = float(mse.item())
        return math.inf if value == 0 else -10.0 * math.log10(value)

    return {
        "PSNR_dynamic": psnr_from_mse(dynamic_mse),
        "SSIM_dynamic": float(dynamic_ssim.item()),
        "LPIPS_dynamic": float(dynamic_lpips.item()),
        "PSNR_full": psnr_from_mse(full_mse),
        "SSIM_full": float(full_ssim_map.mean().item()),
        "LPIPS_full": float(full_lpips.item()),
    }


def evaluate_predictions(
    project_root: Path,
    dataset_root: Path,
    output_dir: Path,
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    import lpips

    expected_total = preflight_predictions(project_root, dataset_root)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    ensure_new_paths([metrics_dir, figures_dir])
    metrics_dir.mkdir()
    figures_dir.mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Metric device: {device}")
    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
    results: list[dict[str, Any]] = []

    first_validation: dict[str, float] | None = None
    for gap in GAPS:
        frame_indices = list(range(gap // 2, 49, gap))
        expected_gap_count = sum(EXPECTED_COUNTS[gap].values())
        progress = tqdm(total=expected_gap_count, desc=f"Evaluating gap {gap}")
        for setting, cameras in (
            ("training_cameras", TRAIN_CAMERAS),
            ("heldout_camera", ("cam00",)),
        ):
            for camera in cameras:
                mask = masks[camera]
                if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                    raise ValueError(
                        f"{camera} mask has wrong shape: {mask.shape}"
                    )
                for frame_index in frame_indices:
                    pred_path, eval_gt_path, _ = prediction_paths(
                        project_root, gap, camera, frame_index
                    )
                    source_gt_path = (
                        dataset_root
                        / camera
                        / "images"
                        / f"{frame_index:04d}.png"
                    )
                    prediction_rgb = load_rgb(pred_path)
                    target_rgb = load_rgb(source_gt_path)
                    saved_gt_rgb = load_rgb(eval_gt_path)
                    if not np.array_equal(target_rgb, saved_gt_rgb):
                        raise ValueError(
                            "Saved evaluation GT differs from source GT: "
                            f"{eval_gt_path}"
                        )
                    if prediction_rgb.shape != target_rgb.shape:
                        raise ValueError(
                            f"Prediction shape differs from GT: {pred_path}"
                        )

                    values = masked_and_full_metrics(
                        prediction_rgb,
                        target_rgb,
                        mask,
                        lpips_model,
                        device,
                    )
                    dynamic_count = int(mask.sum())
                    dynamic_percentage = 100.0 * dynamic_count / mask.size
                    results.append(
                        {
                            "method": "4DGS",
                            "temporal_gap": gap,
                            "evaluation_setting": setting,
                            "camera": camera,
                            "frame_index": frame_index,
                            "dynamic_pixel_count": dynamic_count,
                            "dynamic_pixel_percentage": dynamic_percentage,
                            **values,
                        }
                    )
                    if first_validation is None:
                        first_validation = values
                        print(
                            "Masked-metric validation sample "
                            f"{camera} frame {frame_index} gap {gap}: "
                            f"dynamic pixels={dynamic_count}/"
                            f"{mask.size} ({dynamic_percentage:.2f}%), "
                            f"PSNR dynamic/full="
                            f"{values['PSNR_dynamic']:.4f}/"
                            f"{values['PSNR_full']:.4f}, "
                            f"SSIM dynamic/full="
                            f"{values['SSIM_dynamic']:.4f}/"
                            f"{values['SSIM_full']:.4f}, "
                            f"LPIPS dynamic/full="
                            f"{values['LPIPS_dynamic']:.4f}/"
                            f"{values['LPIPS_full']:.4f}"
                        )
                    progress.update(1)
        progress.close()

        actual_gap_count = sum(
            result["temporal_gap"] == gap for result in results
        )
        if actual_gap_count != expected_gap_count:
            raise RuntimeError(
                f"Gap {gap}: evaluated {actual_gap_count}, "
                f"expected {expected_gap_count}"
            )

    if len(results) != expected_total:
        raise RuntimeError(
            f"Evaluated {len(results)} samples; expected {expected_total}"
        )
    write_metric_csvs(metrics_dir, results)
    write_metric_figures(figures_dir, results)
    return results


def write_metric_csvs(metrics_dir: Path, results: list[dict[str, Any]]) -> None:
    per_sample_columns = [
        "method",
        "temporal_gap",
        "evaluation_setting",
        "camera",
        "frame_index",
        "dynamic_pixel_count",
        "dynamic_pixel_percentage",
        "PSNR_dynamic",
        "SSIM_dynamic",
        "LPIPS_dynamic",
        "PSNR_full",
        "SSIM_full",
        "LPIPS_full",
    ]
    save_csv(
        metrics_dir / "4dgs_dynamic_metrics.csv",
        per_sample_columns,
        [[result[column] for column in per_sample_columns] for result in results],
    )

    summary_rows: list[list[Any]] = []
    for gap in GAPS:
        for setting in ("training_cameras", "heldout_camera"):
            subset = [
                result
                for result in results
                if result["temporal_gap"] == gap
                and result["evaluation_setting"] == setting
            ]
            expected = EXPECTED_COUNTS[gap][setting]
            if len(subset) != expected:
                raise RuntimeError(
                    f"Summary group {(gap, setting)} has {len(subset)} "
                    f"samples; expected {expected}"
                )
            for scope in ("dynamic", "full"):
                metric_columns = (
                    f"PSNR_{scope}",
                    f"SSIM_{scope}",
                    f"LPIPS_{scope}",
                )
                means = [
                    float(np.mean([row[column] for row in subset]))
                    for column in metric_columns
                ]
                summary_rows.append(
                    ["4DGS", gap, setting, scope, *means, len(subset)]
                )
    save_csv(
        metrics_dir / "4dgs_dynamic_summary.csv",
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
        summary_rows,
    )


def write_metric_figures(
    figures_dir: Path, results: list[dict[str, Any]]
) -> None:
    settings = ("training_cameras", "heldout_camera")
    metric_specs = (
        ("PSNR", "PSNR (dB)", True),
        ("SSIM", "SSIM", True),
        ("LPIPS", "LPIPS distance", False),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    x = np.arange(len(GAPS))
    width = 0.18
    for axis, (metric, ylabel, _) in zip(axes, metric_specs):
        for index, (setting, scope, color) in enumerate(
            (
                ("training_cameras", "full", "#4C78A8"),
                ("training_cameras", "dynamic", "#F58518"),
                ("heldout_camera", "full", "#72B7B2"),
                ("heldout_camera", "dynamic", "#E45756"),
            )
        ):
            means = []
            for gap in GAPS:
                column = f"{metric}_{scope}"
                subset = [
                    row[column]
                    for row in results
                    if row["temporal_gap"] == gap
                    and row["evaluation_setting"] == setting
                ]
                means.append(float(np.mean(subset)))
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
    fig.suptitle("Full-image and dynamic-region evaluation")
    fig.savefig(
        figures_dir / "dynamic_region_comparison.png", dpi=180
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (metric, ylabel, _) in zip(axes, metric_specs):
        for setting, color, marker in (
            ("training_cameras", "#4C78A8", "o"),
            ("heldout_camera", "#E45756", "s"),
        ):
            means = []
            for gap in GAPS:
                subset = [
                    row[f"{metric}_dynamic"]
                    for row in results
                    if row["temporal_gap"] == gap
                    and row["evaluation_setting"] == setting
                ]
                means.append(float(np.mean(subset)))
            axis.plot(
                GAPS,
                means,
                marker=marker,
                color=color,
                label=setting.replace("_", " "),
            )
        axis.set_xticks(GAPS, [f"k={gap}" for gap in GAPS])
        axis.set_title(metric)
        axis.set_ylabel(ylabel)
        axis.legend()
    fig.suptitle("Dynamic-region metrics across temporal gaps")
    fig.savefig(figures_dir / "gap_comparison.png", dpi=180)
    plt.close(fig)


def write_methodology(
    output_dir: Path,
    threshold: float,
    camera_stats: dict[str, float],
    results: list[dict[str, Any]] | None,
) -> None:
    percentages = ", ".join(
        f"{camera}: {value:.2f}%" for camera, value in camera_stats.items()
    )
    evaluation_summary = (
        "Evaluation has not been run yet."
        if results is None
        else f"Evaluated {len(results)} samples across gaps 2, 4, and 8."
    )
    text = f"""# Dynamic-region evaluation (4DGS)

## Reproducibility

Run from the 4DGaussians project directory with the project's Python environment:

```bash
python dynamic_region_evaluation.py --stage masks
python dynamic_region_evaluation.py --stage evaluate
```

Default dataset: `../coffee_martini`. Existing midpoint predictions are read from
`output/coffee_martini_k{{2,4,8}}/midpoint_evaluation_v4/gap_{{2,4,8}}`.
No model is rerendered. This result directory is created separately from all
existing model outputs.

## Ground-truth-only mask definition

- Cameras: cam00 (held out) and the 17 listed training cameras.
- Frames: the 50 consecutive grayscale ground-truth images `0000.png` through
  `0049.png`, 676 x 507 pixels.
- Flow: OpenCV Farneback dense optical flow with
  `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`,
  `poly_sigma=1.2`, and `flags=0`.
- For each of 49 consecutive pairs, flow magnitude is
  `sqrt(flow_x**2 + flow_y**2)` in pixels/frame. The per-camera motion score
  is the arithmetic mean of those 49 maps.
- A single pooled Otsu threshold is calculated over all pixels in all 18 motion
  maps using a 4096-bin histogram spanning zero to the global maximum. The
  threshold is {threshold:.9f} pixels/frame; a pixel is dynamic iff its motion
  score is strictly greater than this threshold.
- Mask post-processing: none (no dilation, erosion, or component filtering).
- Temporal grayscale intensity standard deviation is saved for diagnostics only;
  it does not participate in mask creation.

Dynamic-pixel percentage by camera: {percentages}.

Masks are finalized before prediction files are opened and the same camera mask
is reused for all three gaps. Mask arrays, motion scores, intensity maps, pooled
histogram, per-camera statistics, and visual diagnostics are retained in this
directory.

## Metrics

- Images are the existing saved 8-bit midpoint prediction PNGs and their matching
  source ground-truth PNGs. The saved evaluation GT is checked pixel-for-pixel
  against the source before use. Values are converted to RGB float in [0,1].
- PSNR is computed from RGB mean squared error over selected dynamic pixels and
  all three channels. Full-image PSNR is calculated independently over every
  image pixel.
- SSIM uses the repository's 11x11 Gaussian window (sigma 1.5) and constants
  C1=0.01^2, C2=0.03^2. The SSIM map is computed on the original image pair, then
  averaged at dynamic-mask center pixels. Neighborhoods use the unmasked source
  images; pixels outside the dynamic mask are not zeroed.
- LPIPS uses the installed pretrained VGG LPIPS spatial output. The package
  upsamples each feature-layer distance map to the input image resolution;
  dynamic LPIPS is the mean map value at dynamic-mask pixels. If an implementation
  returns a different resolution, the binary mask is resized with nearest-neighbor
  interpolation. Full LPIPS is the mean spatial-map value over the whole image.
- Group means are arithmetic means of per-sample metrics, reported separately by
  method, temporal gap, evaluation setting, and full/dynamic scope.

## Validation and limitations

The pipeline checks all expected predictions, source GT images, saved GT images,
camera masks, dimensions, and sample counts (756 total). It reports a first-sample
full-vs-dynamic metric comparison. The optical-flow/motion maps are diagnostics,
not object segmentation: camera motion, occlusion boundaries, texture, and flow
estimation errors can also receive high scores. Visual inspection found some
static-looking background structures (notably blinds and window/shelf edges)
inside the masks. These are retained rather than silently filtered: the result
is a high-temporal-flow mask, not a semantic person/object segmentation. Otsu is
an unsupervised global threshold, not a semantic human/object mask. No
second-method predictions were present in this experiment, so only 4DGS is
evaluated; the generated camera masks are method-independent and are retained
for reuse when those predictions are available.

{evaluation_summary}
"""
    path = output_dir / "README.md"
    with path.open("x", encoding="utf-8") as file:
        file.write(text)


def run_mask_stage(args: argparse.Namespace) -> dict[str, float]:
    output_dir = args.output_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    if args.stage == "all":
        ensure_new_paths([output_dir])
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_root = output_dir / "masks"
    ensure_new_paths([masks_root])

    score_maps: dict[str, np.ndarray] = {}
    intensity_maps: dict[str, np.ndarray] = {}
    pair_means: dict[str, np.ndarray] = {}
    for camera in tqdm(CAMERAS, desc="Ground-truth flow/motion"):
        frames = load_gray_frames(dataset_root, camera)
        score, intensity, means = calculate_motion(frames)
        score_maps[camera] = score
        intensity_maps[camera] = intensity
        pair_means[camera] = means

    threshold, histogram, edges = otsu_threshold(score_maps)
    print(f"Global pooled Otsu threshold: {threshold:.9f} px/frame")
    camera_stats = save_mask_diagnostics(
        dataset_root,
        output_dir,
        score_maps,
        intensity_maps,
        pair_means,
        threshold,
        histogram,
        edges,
    )
    for camera, percentage in camera_stats.items():
        print(f"{camera}: {percentage:.2f}% dynamic pixels")
    print(f"Ground-truth-only masks saved to {output_dir / 'masks'}")
    return camera_stats


def load_masks(output_dir: Path) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        mask_path = output_dir / "masks" / camera / "dynamic_mask.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing camera mask: {mask_path}")
        with Image.open(mask_path) as image:
            array = np.asarray(image.convert("L"))
        if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"Wrong mask size for {camera}: {array.shape}")
        values = np.unique(array)
        if not set(values.tolist()).issubset({0, 255}):
            raise ValueError(
                f"Mask for {camera} is not binary: values={values.tolist()}"
            )
        masks[camera] = array == 255
    return masks


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser()
    args.project_root = args.project_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser()

    camera_stats: dict[str, float] | None = None
    if args.stage in ("masks", "all"):
        camera_stats = run_mask_stage(args)
    elif not (args.output_dir / "masks" / "threshold_metadata.json").is_file():
        raise FileNotFoundError(
            "Run --stage masks first; no fixed threshold metadata was found."
        )

    if args.stage in ("evaluate", "all"):
        output_dir = args.output_dir.resolve()
        masks = load_masks(output_dir)
        results = evaluate_predictions(
            args.project_root,
            args.dataset_root.resolve(),
            output_dir,
            masks,
        )
        metadata_path = output_dir / "masks" / "threshold_metadata.json"
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        stored_stats = metadata["dynamic_pixel_percentage_by_camera"]
        if camera_stats is None:
            camera_stats = {
                camera: float(stored_stats[camera]) for camera in CAMERAS
            }
        write_methodology(
            output_dir,
            float(metadata["threshold_value_px_per_frame"]),
            camera_stats,
            results,
        )
        print(f"Evaluated {len(results)} samples.")
        print(f"Metrics saved to {output_dir / 'metrics'}")


if __name__ == "__main__":
    main()
