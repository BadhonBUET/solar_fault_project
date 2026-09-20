"""
Solar Panel Thermal Fault Detection Pipeline
=============================================
Detects and classifies PV panel faults (Hotspot / Line Fault / Partial Shading)
from thermal images using classical computer vision (tilt correction, texture-
based panel segmentation, adaptive local thresholding).

Optimizations applied vs. the original version:
  1. Tilt search runs on a downsampled image (132 warps on a small image +
     1 full-resolution warp, instead of 132 full-resolution warps).
  2. Panel ROI building uses cv2.connectedComponentsWithStats instead of
     findContours + a fresh full-image mask allocation per contour.
  3. Large-kernel median blur (background estimate) is approximated on a
     downsampled copy and resized back up.
  4. gray -> float32 conversion happens once per image and is passed down,
     instead of being repeated in two functions.
  5. Morphological structuring elements are built once at import time
     instead of being recreated on every function call.
  6. All tunable thresholds live in one CONFIG dict (was scattered magic
     numbers) - also doubles as the parameter reference table for reporting.
  7. Added a non-blocking batch mode that processes a folder of images and
     writes annotated outputs + a results CSV (fault counts, confidence,
     per-block runtime) - needed for accuracy/performance analysis across
     a test set, which the original single-image/GUI-blocking script could
     not do.
"""

import argparse
import csv
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# CONFIG: every tunable threshold in one place. Change values here, not in
# the function bodies below. This dict is also your slide-34 "master
# threshold table" - each key maps directly to a row in that table.
# ==============================================================================
CONFIG: Dict[str, Dict[str, Any]] = {
    "tilt": {
        "coarse_range_deg": 45.0,   # search from -45 to +45 degrees
        "coarse_steps": 91,         # ~1 degree resolution
        "fine_window_deg": 2.0,     # +/- 2 degrees around the coarse best
        "fine_steps": 41,           # ~0.1 degree resolution
        "search_scale": 0.25,       # downsample factor used only for angle search
        "min_search_dim": 200,      # never downsample below this many px
    },
    "texture_mask": {
        "kernel_size": (15, 15),    # box filter window for local mean/variance
        "std_threshold": 45,        # local-std cutoff for "smooth/cell-like"
        "min_valid_gray": 25,       # ignore near-black background pixels
    },
    "roi": {
        "max_join_dist": 35,        # horizontal closing kernel width (bridges frame lines)
        "border_padding": 1,        # px of padding added around final panel boxes
        "min_blob_area": 15,        # minimum pixel area for a raw texture blob
        "min_panel_area_ratio": 0.003,  # panel must cover >= 0.3% of image area
        "median_var_threshold": 65.0,   # median LTI variance must be below this
    },
    "fault_detection": {
        "bg_kernel_ratio": 0.05,        # local background kernel = 5% of min(h, w)
        "k_factor_base": 2.0,
        "k_factor_contrast_weight": 0.5,
        "k_factor_min": 2.2,
        "k_factor_max": 4.0,
        "threshold_min": 5.0,
        "threshold_max": 25.0,
        "min_hotspot_area": 15,
        "use_fast_background": True,    # approximate large-kernel median blur
    },
    "classification": {
        "partial_shading_range": (65.0, 80.0),  # % of max panel brightness
        "line_fault_min_brightness": 80.0,
        "hotspot_min_brightness": 75.0,
        "long_aspect_ratio_high": 3.0,
        "long_aspect_ratio_low": 0.33,
        "long_width_ratio": 0.45,
        "long_height_ratio": 0.45,
        "long_area_ratio": 0.03,
        "severe_aspect_ratio_high": 3.2,
        "severe_aspect_ratio_low": 0.31,
        "severe_width_ratio": 0.60,
        "severe_height_ratio": 0.60,
    },
}

# ==============================================================================
# Structuring elements built once at import time (previously recreated on
# every function call - wasteful once you're batch-processing many images).
# ==============================================================================
_roi_cfg = CONFIG["roi"]
HORIZ_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (_roi_cfg["max_join_dist"], 3))
VERT_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
PAD_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_RECT, (2 * _roi_cfg["border_padding"] + 1, 2 * _roi_cfg["border_padding"] + 1)
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pv_inspection")


# ==============================================================================
# BLOCK 2: IMAGE SELECTION (unchanged)
# ==============================================================================
def select_image() -> str:
    """Open a native file browser and return the chosen image path (or '').
    tkinter is imported here (not at module level) so batch/headless mode
    doesn't require a display or a tkinter install at all."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select Solar Thermal Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
    )
    root.destroy()
    return file_path


# ==============================================================================
# BLOCK 3: TILT ALIGNMENT (optimized)
# Angle search now runs on a downsampled copy of the image; the full-
# resolution image is only warped once, with the winning angle.
# ==============================================================================
def _score_angle(img: np.ndarray, angle: float, cx: float, cy: float) -> float:
    """Projection-profile score: higher when rows/cols are well aligned."""
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(
        img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
    )
    return float(np.var(np.mean(rot, axis=1)) + np.var(np.mean(rot, axis=0)))


def detect_and_align_tilt(gray: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Find the rotation angle that best aligns the panel grid (rows/columns of
    cells produce maximum brightness variance when properly aligned), then
    apply that single rotation at full resolution.

    Optimization: the 91 coarse + 41 fine candidate angles are scored on a
    downsampled image (cheap warps); only the final, winning angle is applied
    to the full-resolution image with a single high-quality warp.
    """
    cfg = CONFIG["tilt"]
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    scale = cfg["search_scale"]
    small_w = max(int(w * scale), cfg["min_search_dim"])
    small_h = max(int(h * scale), cfg["min_search_dim"])

    if small_w >= w or small_h >= h:
        # Image is already small; searching on it directly is fine.
        small, scx, scy = gray, cx, cy
    else:
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        scy, scx = small_h // 2, small_w // 2

    coarse_angles = np.linspace(-cfg["coarse_range_deg"], cfg["coarse_range_deg"], cfg["coarse_steps"])
    best_angle, max_score = 0.0, -1.0
    for a in coarse_angles:
        score = _score_angle(small, a, scx, scy)
        if score > max_score:
            max_score, best_angle = score, a

    fine_angles = np.linspace(
        best_angle - cfg["fine_window_deg"], best_angle + cfg["fine_window_deg"], cfg["fine_steps"]
    )
    for a in fine_angles:
        score = _score_angle(small, a, scx, scy)
        if score > max_score:
            max_score, best_angle = score, a

    # The ONLY full-resolution warp in tilt alignment.
    rot_mat = cv2.getRotationMatrix2D((cx, cy), best_angle, 1.0)
    gray_aligned = cv2.warpAffine(
        gray, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return best_angle, gray_aligned, rot_mat


# ==============================================================================
# BLOCK 4: CELL TEXTURE MASK
# ==============================================================================
def generate_cell_texture_mask(
    gray_aligned: np.ndarray, gray_f: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Flag pixels that are locally smooth/uniform - typical of solar cells."""
    cfg = CONFIG["texture_mask"]
    if gray_f is None:
        gray_f = gray_aligned.astype(np.float32)
    k = cfg["kernel_size"]

    mean = cv2.blur(gray_f, k)
    mean_sq = cv2.blur(gray_f ** 2, k)
    var = np.maximum(mean_sq - mean ** 2, 0)
    std_map = cv2.normalize(np.sqrt(var), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, low_var_mask = cv2.threshold(std_map, cfg["std_threshold"], 255, cv2.THRESH_BINARY_INV)
    valid_gray = (gray_aligned > cfg["min_valid_gray"]).astype(np.uint8) * 255
    low_var_mask = cv2.bitwise_and(low_var_mask, valid_gray)

    return std_map, low_var_mask


# ==============================================================================
# BLOCK 5: PANEL ROI BUILDER (optimized)
# Replaced findContours + per-contour full-image mask allocation with
# cv2.connectedComponentsWithStats, so the median-variance filter reads
# directly from the label map instead of redrawing a mask per candidate.
# ==============================================================================
def build_rect_joined_roi_with_padding(
    low_var_mask: np.ndarray, std_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join fragmented low-variance blobs into full-panel bounding boxes,
    keeping only regions whose median local-texture variance is low enough
    to be a real panel (median, not mean, so one bright frame line inside
    an otherwise-uniform panel doesn't disqualify it)."""
    cfg = CONFIG["roi"]
    h, w = low_var_mask.shape

    # Step 1: vectorized bounding boxes of raw low-variance blobs.
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(low_var_mask, connectivity=8)
    rect_boxes_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area > cfg["min_blob_area"]:
            rect_boxes_mask[y : y + bh, x : x + bw] = 255

    # Step 2: morphologically close gaps (frame lines / cell borders).
    closed_mask = cv2.morphologyEx(rect_boxes_mask, cv2.MORPH_CLOSE, HORIZ_CLOSE_KERNEL)
    closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, VERT_CLOSE_KERNEL)

    # Step 3: keep joined regions only if median texture variance is low enough.
    panel_n_labels, panel_labels, panel_stats, _ = cv2.connectedComponentsWithStats(
        closed_mask, connectivity=8
    )
    raw_roi_mask = np.zeros((h, w), dtype=np.uint8)
    min_panel_area = cfg["min_panel_area_ratio"] * h * w

    for i in range(1, panel_n_labels):
        x, y, bw, bh, area = panel_stats[i]
        if area < min_panel_area:
            continue
        region_std_vals = std_map[panel_labels == i]
        if region_std_vals.size == 0:
            continue
        if np.median(region_std_vals) < cfg["median_var_threshold"]:
            raw_roi_mask[y : y + bh, x : x + bw] = 255

    # Step 4: pad panel boxes slightly.
    final_padded_roi = cv2.dilate(raw_roi_mask, PAD_KERNEL, iterations=1)

    border_visual = cv2.cvtColor(low_var_mask, cv2.COLOR_GRAY2BGR)
    roi_cnts, _ = cv2.findContours(final_padded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(border_visual, roi_cnts, -1, (0, 255, 0), 2)

    return rect_boxes_mask, border_visual, final_padded_roi


# ==============================================================================
# BLOCK 6: THERMAL FAULT DETECTION & CLASSIFICATION (optimized)
# ==============================================================================
def _fast_local_median_background(
    gray_aligned: np.ndarray, k_size: int, downsample_factor: float = 0.5
) -> np.ndarray:
    """Approximate a large-kernel median blur (local background estimate) by
    running it on a downsampled copy and resizing back up. Median blur cost
    scales with kernel size, so for the ~5%-of-image-dimension kernels used
    here this is substantially faster, with minimal accuracy loss since the
    background is meant to be a *smooth* estimate anyway."""
    if k_size <= 15:
        return cv2.medianBlur(gray_aligned, k_size)
    h, w = gray_aligned.shape
    small_w = max(3, int(w * downsample_factor))
    small_h = max(3, int(h * downsample_factor))
    small = cv2.resize(gray_aligned, (small_w, small_h), interpolation=cv2.INTER_AREA)
    k_small = max(3, int(k_size * downsample_factor) | 1)
    small_bg = cv2.medianBlur(small, k_small)
    return cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)


def detect_faults_inside_roi(
    gray_aligned: np.ndarray, panel_roi_mask: np.ndarray, gray_f: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], float]:
    """Find pixels significantly hotter than their local background inside
    the panel ROI, then classify each surviving blob as Hotspot / Line Fault
    / Partial Shading based on brightness percentile and shape."""
    cfg = CONFIG["fault_detection"]
    cls_cfg = CONFIG["classification"]
    h, w = gray_aligned.shape
    if gray_f is None:
        gray_f = gray_aligned.astype(np.float32)

    k_size = int(min(h, w) * cfg["bg_kernel_ratio"]) | 1
    if cfg["use_fast_background"]:
        local_bg = _fast_local_median_background(gray_aligned, k_size).astype(np.float32)
    else:
        local_bg = cv2.medianBlur(gray_aligned, k_size).astype(np.float32)

    thermal_delta = np.maximum(gray_f - local_bg, 0.0)

    panel_raw_pixels = gray_aligned[panel_roi_mask == 255]
    max_panel_brightness = float(np.max(panel_raw_pixels)) if len(panel_raw_pixels) > 0 else 255.0

    local_mean = cv2.blur(thermal_delta, (k_size, k_size))
    local_sq_mean = cv2.blur(thermal_delta ** 2, (k_size, k_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.0))

    roi_deltas = thermal_delta[panel_roi_mask == 255]
    contrast_ratio = np.std(roi_deltas) / (np.mean(roi_deltas) + 1e-5) if len(roi_deltas) > 0 else 1.0
    k_factor = float(
        np.clip(
            cfg["k_factor_base"] + cfg["k_factor_contrast_weight"] * contrast_ratio,
            cfg["k_factor_min"],
            cfg["k_factor_max"],
        )
    )

    threshold_map = local_mean + np.clip(k_factor * local_std, cfg["threshold_min"], cfg["threshold_max"])
    raw_candidates = ((thermal_delta >= threshold_map) & (panel_roi_mask == 255)).astype(np.uint8) * 255
    clean_anomaly_mask = cv2.morphologyEx(raw_candidates, cv2.MORPH_OPEN, OPEN_KERNEL)

    panel_cnts, _ = cv2.findContours(panel_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel_boxes = [cv2.boundingRect(c) for c in panel_cnts if cv2.contourArea(c) > 0]
    contours, _ = cv2.findContours(clean_anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_faults: List[Dict[str, Any]] = []
    filtered_mask = np.zeros_like(clean_anomaly_mask)
    cnt_mask = np.zeros_like(clean_anomaly_mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg["min_hotspot_area"]:
            continue

        cnt_mask.fill(0)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        peak_anomaly_brightness = float(np.max(gray_aligned[cnt_mask == 255]))
        brightness_pct = (peak_anomaly_brightness / (max_panel_brightness + 1e-5)) * 100.0

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / bh if bh > 0 else 1.0
        confidence_pct = int(round(brightness_pct))

        cx_f, cy_f = x + bw / 2.0, y + bh / 2.0
        p_match = next(
            (pb for pb in panel_boxes if pb[0] <= cx_f <= pb[0] + pb[2] and pb[1] <= cy_f <= pb[1] + pb[3]),
            None,
        )
        panel_w, panel_h = (float(p_match[2]), float(p_match[3])) if p_match else (float(w), float(h))

        is_severe_line = (
            aspect_ratio >= cls_cfg["severe_aspect_ratio_high"] and bw >= cls_cfg["severe_width_ratio"] * panel_w
        ) or (
            aspect_ratio <= cls_cfg["severe_aspect_ratio_low"] and bh >= cls_cfg["severe_height_ratio"] * panel_h
        )
        is_really_long = (
            (aspect_ratio >= cls_cfg["long_aspect_ratio_high"] and bw >= cls_cfg["long_width_ratio"] * panel_w)
            or (aspect_ratio <= cls_cfg["long_aspect_ratio_low"] and bh >= cls_cfg["long_height_ratio"] * panel_h)
            or (area >= cls_cfg["long_area_ratio"] * panel_w * panel_h)
        )

        lo, hi = cls_cfg["partial_shading_range"]
        if lo <= brightness_pct <= hi and is_really_long:
            f_type, color = "Partial Shading", (0, 165, 255)
        elif brightness_pct > cls_cfg["line_fault_min_brightness"] and is_really_long:
            f_type, color = "Line Fault", (0, 255, 255)
        elif brightness_pct > cls_cfg["hotspot_min_brightness"]:
            f_type, color = ("Line Fault", (0, 255, 255)) if is_severe_line else ("Hotspot", (0, 0, 255))
        else:
            continue

        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        detected_faults.append(
            {"box": (x, y, bw, bh), "type": f_type, "color": color, "confidence": confidence_pct}
        )

    percentage_map = np.zeros_like(thermal_delta, dtype=np.float32)
    valid_indices = (filtered_mask > 0) & (panel_roi_mask == 255)
    if np.any(valid_indices):
        percentage_map[valid_indices] = (
            gray_aligned[valid_indices].astype(np.float32) / (max_panel_brightness + 1e-5)
        ) * 100.0

    return thermal_delta, percentage_map, detected_faults, max_panel_brightness


# ==============================================================================
# BLOCK 7: MAIN PIPELINE & VISUALIZATION
# ==============================================================================
def _show_diagnostic_figure(
    gray_raw, std_map, low_var_mask, border_overlay, thermal_delta, percentage_map, annotated, counts, faults
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(gray_raw, cmap="gray")
    axes[0, 0].set_title("1. Raw Thermal Input", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(std_map, cmap="viridis")
    axes[0, 1].set_title("2. LTI Variance Map", fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(low_var_mask, cmap="gray")
    axes[0, 2].set_title("3. Cell Texture Mask", fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(border_overlay)
    axes[0, 3].set_title("4. Rect-Joined +1px ROI", fontweight="bold")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(thermal_delta, cmap="magma")
    axes[1, 0].set_title("5. Spatial Thermal Delta", fontweight="bold")
    axes[1, 0].axis("off")

    im_perc = axes[1, 1].imshow(percentage_map, cmap="inferno", vmin=0, vmax=100)
    axes[1, 1].set_title("6. Intensity Map (Verified Faults)", fontweight="bold")
    axes[1, 1].axis("off")
    fig.colorbar(im_perc, ax=axes[1, 1], fraction=0.046, pad=0.04, label="% Panel Max Brightness")

    axes[1, 2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(
        f'7. Output (H:{counts["Hotspot"]} L:{counts["Line Fault"]} S:{counts["Partial Shading"]})',
        fontweight="bold",
    )
    axes[1, 2].axis("off")

    axes[1, 3].axis("off")
    axes[1, 3].set_title("8. Fault Summary List", fontweight="bold")
    summary_text = [
        f"Total Detected: {len(faults)}",
        f"• Hotspots: {counts['Hotspot']}",
        f"• Line Faults: {counts['Line Fault']}",
        f"• Partial Shading: {counts['Partial Shading']}",
        "",
    ]
    if faults:
        summary_text.append("Top Anomalies:")
        for idx, f in enumerate(faults[:8]):
            summary_text.append(f"{idx + 1}. {f['type']} - {f['confidence']}%")
    else:
        summary_text.append("No faults detected.")

    axes[1, 3].text(
        0.05,
        0.95,
        "\n".join(summary_text),
        transform=axes[1, 3].transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.show()


def run_pv_inspection_pipeline(
    image_path: str, save_dir: Optional[str] = None, show_plot: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run the full fault-detection pipeline on a single thermal image.

    Returns a results dict (fault counts, confidences, per-block runtime) so
    it can be logged to CSV in batch mode, in addition to optionally
    displaying and/or saving the annotated output.
    """
    timings: Dict[str, float] = {}
    t_start = time.perf_counter()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logger.error("Could not open thermal image: %s", image_path)
        return None

    gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray_raw.shape

    t0 = time.perf_counter()
    angle, gray_aligned, rot_mat = detect_and_align_tilt(gray_raw)
    timings["tilt_alignment_s"] = time.perf_counter() - t0

    gray_f = gray_aligned.astype(np.float32)  # computed once, reused below

    t0 = time.perf_counter()
    std_map, low_var_mask = generate_cell_texture_mask(gray_aligned, gray_f)
    timings["texture_mask_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rect_boxes_mask, border_overlay, panel_roi = build_rect_joined_roi_with_padding(low_var_mask, std_map)
    timings["roi_build_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    thermal_delta, percentage_map, faults, max_p_bright = detect_faults_inside_roi(
        gray_aligned, panel_roi, gray_f
    )
    timings["fault_detection_s"] = time.perf_counter() - t0

    inv_rot_mat = cv2.invertAffineTransform(rot_mat)
    panel_roi_orig = cv2.warpAffine(panel_roi, inv_rot_mat, (w, h), flags=cv2.INTER_NEAREST)

    annotated = img_bgr.copy()
    panel_cnts, _ = cv2.findContours(panel_roi_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, panel_cnts, -1, (0, 255, 0), 2)

    counts = {"Hotspot": 0, "Line Fault": 0, "Partial Shading": 0}
    for fault in faults:
        x, y, bw, bh = fault["box"]
        f_type, color, conf = fault["type"], fault["color"], fault["confidence"]
        counts[f_type] += 1

        pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
        pts_homo = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
        pts_orig = np.dot(pts_homo, inv_rot_mat.T).astype(np.int32)

        cv2.polylines(annotated, [pts_orig], isClosed=True, color=color, thickness=2)
        top_left = pts_orig[0]
        cv2.putText(
            annotated,
            f"{f_type} {conf}%",
            (top_left[0] - 2, max(top_left[1] - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
        )

    timings["total_s"] = time.perf_counter() - t_start

    logger.info(
        "Corrected Tilt Angle: %.2f | Panel Max Intensity: %.1f | "
        "Verified Anomalies: %d Hotspots, %d Line Faults, %d Partial Shadings | Total: %.3fs",
        angle,
        max_p_bright,
        counts["Hotspot"],
        counts["Line Fault"],
        counts["Partial Shading"],
        timings["total_s"],
    )

    results: Dict[str, Any] = {
        "image_path": image_path,
        "tilt_angle_deg": round(angle, 3),
        "max_panel_brightness": round(max_p_bright, 1),
        "n_hotspot": counts["Hotspot"],
        "n_line_fault": counts["Line Fault"],
        "n_partial_shading": counts["Partial Shading"],
        "n_total_faults": len(faults),
        "fault_details": faults,
        **{k: round(v, 4) for k, v in timings.items()},
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(save_dir, f"{base}_annotated.png")
        cv2.imwrite(out_path, annotated)
        results["annotated_path"] = out_path

    if show_plot:
        _show_diagnostic_figure(
            gray_raw, std_map, low_var_mask, border_overlay, thermal_delta, percentage_map, annotated, counts, faults
        )

    return results


def run_batch(
    image_dir: str, save_dir: str = "batch_results", csv_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process every supported image in image_dir without opening a GUI, saving
    annotated images plus a CSV summary (fault counts, confidence-relevant
    fields, and per-block runtime). Use this to run your 20-image test set
    for the accuracy/performance-analysis slides.
    """
    os.makedirs(save_dir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(save_dir, "results.csv")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_paths = sorted(
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(exts)
    )

    if not image_paths:
        logger.warning("No supported images found in %s", image_dir)
        return []

    all_results: List[Dict[str, Any]] = []
    for path in image_paths:
        logger.info("Processing %s", path)
        res = run_pv_inspection_pipeline(path, save_dir=save_dir, show_plot=False)
        if res is not None:
            all_results.append(res)

    fieldnames = [
        "image_path",
        "tilt_angle_deg",
        "max_panel_brightness",
        "n_hotspot",
        "n_line_fault",
        "n_partial_shading",
        "n_total_faults",
        "tilt_alignment_s",
        "texture_mask_s",
        "roi_build_s",
        "fault_detection_s",
        "total_s",
        "annotated_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    logger.info("Batch complete: %d/%d images processed. Results saved to %s",
                len(all_results), len(image_paths), csv_path)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PV thermal fault detection pipeline")
    parser.add_argument(
        "--batch", type=str, default=None, help="Path to a folder of thermal images to batch-process"
    )
    parser.add_argument(
        "--save-dir", type=str, default="batch_results", help="Where to save annotated images/CSV in batch mode"
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch, save_dir=args.save_dir)
    else:
        logger.info("Select image to run pipeline...")
        selected_path = select_image()
        if selected_path:
            run_pv_inspection_pipeline(selected_path, show_plot=True)