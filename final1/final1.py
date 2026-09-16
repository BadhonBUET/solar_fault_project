# ==============================================================================
# Thermal Image Analysis System for Photovoltaic Fault Detection
# ------------------------------------------------------------------------------
# This file is the working version of your code with four lightweight DSP
# add-ons layered on top. NOTHING in the original detection or classification
# logic has been changed. The four additions are:
#
#   ADD-ON 1 — FFT Tilt Verification  (inside Block 3, logged to console)
#   ADD-ON 2 — Wavelet Seed Widening  (inside Block 6, only adds extra seeds)
#   ADD-ON 3 — IIR Line Confirmation  (inside Block 6, only re-checks Line Faults)
#   ADD-ON 4 — Fault Severity Index   (inside Block 6, adds fsi key to each fault)
#
# Required packages (in addition to your existing ones):
#   pip install PyWavelets scipy
# ==============================================================================


# ==============================================================================
# BLOCK 1: IMPORTING LIBRARIES
# ==============================================================================

import sys                          # Used to exit the program gracefully on fatal errors
import cv2                          # OpenCV — loads images and runs all computer vision operations
import numpy as np                  # NumPy — fast maths on large grids of numbers (matrices)
import pywt                         # PyWavelets — provides the 2D Discrete Wavelet Transform (ADD-ON 2)
import scipy.signal as spsig        # SciPy Signal — provides IIR digital filter design (ADD-ON 3)
import matplotlib.pyplot as plt     # Matplotlib — draws the 2×4 subplot diagnostic window
import tkinter as tk                # Tkinter — creates the OS file-picker pop-up window
from tkinter import filedialog      # The actual file-dialog sub-component inside Tkinter


# ==============================================================================
# BLOCK 2: IMAGE FILE SELECTION
# Opens a native OS file browser so the user can pick the thermal image.
# Nothing about detection happens here — it just returns a file path string.
# ==============================================================================

def select_image():
    root = tk.Tk()                       # Create a hidden background Tkinter window (required by the dialog)
    root.withdraw()                      # Hide it immediately so only the dialog box is visible
    root.attributes('-topmost', True)    # Force the dialog to appear in front of all other windows

    file_path = filedialog.askopenfilename(
        title="Select Solar Thermal Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )                                    # Opens the OS file picker; user clicks the image file

    root.destroy()                       # Close and clean up the Tkinter session
    return file_path                     # Return the full path string (e.g. "C:/images/panel.jpg")


# ==============================================================================
# BLOCK 3: TILT ALIGNMENT
# ------------------------------------------------------------------------------
# Solar panel arrays in drone images are often slightly tilted.  All downstream
# analysis assumes horizontal panel rows, so we must straighten (deskew) the
# image first.
#
# HOW IT WORKS:
#   Step A — Coarse scan: Try 91 angles from -45° to +45° (every 1°).
#             For each angle, rotate the image and compute a "projection variance
#             score".  When the panel rows are horizontal, averaging each row
#             gives a strongly varying signal (bright row vs dark gap vs bright
#             row …), so np.var(row_means) is large.  The angle with the largest
#             combined row+column variance wins.
#
#   Step B — Fine scan: Repeat with 41 angles within ±2° of the coarse winner,
#             stepping 0.1°, to lock in sub-degree precision.
#
#   Step C — Final rotation: Apply the winning angle with cubic interpolation
#             (smooth pixel blending) for the best output quality.
#
# DSP ADD-ON 1 — FFT Tilt Verification (does NOT change the angle):
#   A regular grid is a periodic 2D signal.  Its 2D DFT produces bright energy
#   spots at the grid's spatial frequencies.  We apply a Hanning window before
#   the FFT to suppress "spectral leakage" (energy smearing into wrong bins
#   caused by the image edges), then find the dominant off-axis spectral peak.
#   The angle of that peak gives an independent tilt estimate.
#   We print both estimates to the console so you can see they agree.
#   The projection-scan angle is still used for the actual rotation.
# ==============================================================================

def detect_and_align_tilt(gray):

    h, w = gray.shape           # Image height h (rows) and width w (columns) in pixels
    cy, cx = h // 2, w // 2     # Centre pixel coordinates used as the rotation pivot

    # ── Step A: Coarse angle search (1° steps, −45° to +45°) ─────────────────
    coarse_angles = np.linspace(-45.0, 45.0, 91)   # 91 evenly spaced test angles
    best_angle = 0.0                                 # Start assuming no tilt
    max_score  = -1.0                                # Track the best projection variance seen

    for a in coarse_angles:
        # getRotationMatrix2D returns a 2×3 affine matrix that rotates around (cx,cy) by a degrees
        M_test = cv2.getRotationMatrix2D((cx, cy), a, 1.0)

        # warpAffine applies the rotation matrix to every pixel.
        # INTER_NEAREST: fast nearest-neighbour resampling (good enough for scoring)
        # BORDER_REPLICATE: fills edge pixels by copying the nearest edge pixel
        rot_test = cv2.warpAffine(gray, M_test, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REPLICATE)

        # Projection variance score:
        #   np.mean(rot_test, axis=1) → one average brightness value per row
        #   np.var(...)               → variance across those row-averages
        # When rows are aligned with panel rows, bright rows alternate with dark
        # grid lines → high variance.  Same logic applies column-wise.
        score = np.var(np.mean(rot_test, axis=1)) + np.var(np.mean(rot_test, axis=0))

        if score > max_score:
            max_score  = score
            best_angle = a      # Keep whichever angle gave the highest score

    # ── Step B: Fine angle search (0.1° steps, within ±2° of coarse winner) ──
    fine_angles = np.linspace(best_angle - 2.0, best_angle + 2.0, 41)   # 41 fine steps

    for a in fine_angles:
        M_test   = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        rot_test = cv2.warpAffine(gray, M_test, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REPLICATE)
        score = np.var(np.mean(rot_test, axis=1)) + np.var(np.mean(rot_test, axis=0))
        if score > max_score:
            max_score  = score
            best_angle = a

    # ── Step C: Final high-quality rotation with the winning angle ────────────
    rot_mat      = cv2.getRotationMatrix2D((cx, cy), best_angle, 1.0)
    gray_aligned = cv2.warpAffine(gray, rot_mat, (w, h),
                                  flags=cv2.INTER_CUBIC,          # cubic = best quality interpolation
                                  borderMode=cv2.BORDER_REPLICATE)

    # ── DSP ADD-ON 1: FFT Tilt Verification ──────────────────────────────────
    # This section runs AFTER the projection scan.  It never changes best_angle.
    # Its only purpose is to compute a second independent tilt estimate using
    # the 2D DFT, then print both to the console for cross-checking.
    #
    # WHY HANNING WINDOW?
    #   The FFT assumes the signal repeats infinitely.  A real finite image has
    #   abrupt edges, which introduce artificial high-frequency content (spectral
    #   leakage).  Multiplying by a Hanning window w(n) = 0.5·(1−cos(2πn/N))
    #   tapers the image smoothly to zero at all four edges, eliminating leakage.
    #
    # WHY DOES THE GRID APPEAR AS A SPECTRAL PEAK?
    #   A regular grid of cells is a 2D periodic signal.  Its DFT concentrates
    #   energy at the grid's fundamental spatial frequency (and its harmonics).
    #   The angle of the energy lobe relative to the DC centre gives the tilt.

    win   = np.outer(np.hanning(h), np.hanning(w))           # 2D separable Hanning window
    F     = np.fft.fft2(gray.astype(np.float32) * win)       # 2D DFT of windowed image
    F_sh  = np.fft.fftshift(F)                               # Shift DC component to image centre
    fft_mag = np.log1p(np.abs(F_sh))                         # Log-magnitude spectrum (for display)

    # Build a masked copy of the magnitude to search for the dominant off-axis peak
    mag_search = fft_mag.copy()
    mag_search[h//2-20:h//2+20, w//2-20:w//2+20] = 0        # Zero out DC blob at centre
    mag_search[h//2-6:h//2+6,   :]               = 0        # Zero out the horizontal axis (always bright)
    mag_search[:,               w//2-6:w//2+6]   = 0        # Zero out the vertical axis   (always bright)

    # Only search the upper half (negative v-frequency → horizontal structures like panel rows)
    search_half = mag_search[:h//2, :]
    py, px = np.unravel_index(np.argmax(search_half), search_half.shape)   # Peak location

    # Convert peak position to angle (peak is perpendicular to grid, so add 90°)
    ang_deg   = np.degrees(np.arctan2(py - h//2, px - w//2))
    fft_angle = (ang_deg + 90.0) % 180.0 - 90.0
    if abs(fft_angle) > 45.0:
        fft_angle -= np.sign(fft_angle) * 90.0               # Wrap into [−45°, +45°] range

    agree = abs(fft_angle - best_angle) < 6.0                # Flag if both methods agree within 6°

    # Print verification summary to console
    print(f"  [DSP CHECK — 2D FFT Tilt Estimate] "
          f"FFT={fft_angle:.1f}°  Projection={best_angle:.1f}°  "
          f"Consistent={'YES ✓' if agree else 'NO — check image quality'}")

    return best_angle, gray_aligned, rot_mat, fft_mag


# ==============================================================================
# BLOCK 4: CELL TEXTURE MASK  (LTI Moving-Variance Filter)
# ------------------------------------------------------------------------------
# Solar cell glass is a smooth, uniform surface → low local brightness variance.
# Gravel, soil, or concrete background is rough and noisy → high local variance.
# This block computes the local standard deviation at every pixel and thresholds
# it to produce a binary mask: white = smooth panel surface, black = background.
#
# DSP CONCEPT — LTI Moving-Statistics Filter:
#   A box blur is a Linear Time-Invariant (LTI) system with a rectangular
#   impulse response h[m,n] = 1/K² for all (m,n) inside a K×K window.
#   Applied twice (once for E[X], once for E[X²]) it gives local variance:
#       Var(X) = E[X²] − (E[X])²
# ==============================================================================

def generate_cell_texture_mask(gray_aligned):
    """
    Parameters
    ----------
    gray_aligned : uint8 — deskewed image from Block 3

    Returns
    -------
    std_map      : uint8 — normalised local std-dev image (displayed in Subplot 3)
    low_var_mask : uint8 binary — white pixels mark smooth solar-cell regions
    """

    gray_f = gray_aligned.astype(np.float32)    # Convert to float so squaring doesn't overflow uint8
    k      = (15, 15)                           # 15×15 sliding window (≈ half a cell width)

    # cv2.blur is a 2D box filter — equivalent to averaging every 15×15 neighbourhood
    mean    = cv2.blur(gray_f, k)               # Local mean:     E[X]   at every pixel
    mean_sq = cv2.blur(gray_f ** 2, k)          # Local mean of X²: E[X²] at every pixel

    # Variance formula: Var(X) = E[X²] − (E[X])²
    # np.maximum(..., 0) prevents tiny negative values from floating-point rounding errors
    var     = np.maximum(mean_sq - mean ** 2, 0)

    # Standard deviation = sqrt(variance).  Normalise to 0–255 for threshold + display.
    std_map = cv2.normalize(np.sqrt(var), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # THRESH_BINARY_INV: pixels with std ≤ 45 become WHITE (low variance = smooth cell)
    _, low_var_mask = cv2.threshold(std_map, 45, 255, cv2.THRESH_BINARY_INV)

    # Exclude near-black pixels (value ≤ 25) which are empty image edges, not solar cells
    valid_gray   = (gray_aligned > 25).astype(np.uint8) * 255    # 255 where pixel is meaningful
    low_var_mask = cv2.bitwise_and(low_var_mask, valid_gray)      # Keep only valid smooth pixels

    return std_map, low_var_mask


# ==============================================================================
# BLOCK 5: PANEL ROI BUILDER  (Morphological Closing + Median Variance Filter)
# ------------------------------------------------------------------------------
# Individual cell detections from Block 4 are scattered white patches.
# This block groups them into solid rectangular panel rows (the ROI).
#
# HOW IT WORKS:
#   1. Draw axis-aligned bounding rectangles around every cell patch.
#   2. Morphological CLOSING: dilate then erode — this bridges horizontal gaps
#      between adjacent cells in the same panel row.
#   3. Keep only blobs large enough to be real panel rows (≥ 0.3% of image area).
#   4. Median variance check: a real panel has low MEDIAN local std inside it,
#      even if a bright frame line pushes the mean high.  Reject blobs whose
#      median std_map value exceeds 65 (not a smooth panel surface).
#   5. Dilate the final mask by 1 pixel so the very edge of each cell is included.
#
# Note: axis-aligned rectangles are correct here because Block 3 already
# deskewed the image — the panel rows are now horizontal.
# ==============================================================================

def build_rect_joined_roi_with_padding(low_var_mask, std_map,
                                       max_join_dist=35, border_padding=1):
    """
    Parameters
    ----------
    low_var_mask  : uint8 binary — smooth-cell mask from Block 4
    std_map       : uint8 — local std-dev image from Block 4 (used for median check)
    max_join_dist : int   — horizontal closing kernel width (pixels) to bridge cell gaps
    border_padding: int   — dilation pixels added around the final ROI

    Returns
    -------
    rect_boxes_mask: uint8 binary — individual cell rectangles before merging (Subplot 4)
    border_visual  : BGR   — green-outlined ROI drawn on top of the cell mask (Subplot 5)
    final_padded_roi: uint8 binary — merged, padded panel ROI (used in Block 6)
    """

    h, w = low_var_mask.shape

    # ── Step 1: Draw bounding rectangles around each smooth-cell patch ────────
    contours, _ = cv2.findContours(low_var_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.findContours traces boundaries of white regions in the binary mask

    rect_boxes_mask = np.zeros((h, w), dtype=np.uint8)   # Blank canvas for rectangles

    for cnt in contours:
        if cv2.contourArea(cnt) > 15:                    # Ignore microscopic noise specks
            rx, ry, rw, rh = cv2.boundingRect(cnt)       # Smallest axis-aligned rectangle enclosing cnt
            cv2.rectangle(rect_boxes_mask, (rx, ry), (rx + rw, ry + rh), 255, -1)  # Fill solid white

    # ── Step 2: Morphological CLOSING to merge adjacent cells ─────────────────
    # CLOSING = DILATION followed by EROSION using the same kernel.
    # The 35×3 horizontal kernel sweeps across rows, fusing cells separated by
    # up to 35 pixels (the dark grid-line gap between cells).
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_join_dist, 3))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))

    closed_mask = cv2.morphologyEx(rect_boxes_mask, cv2.MORPH_CLOSE, horiz_kernel)  # Fuse horizontally
    closed_mask = cv2.morphologyEx(closed_mask,     cv2.MORPH_CLOSE, vert_kernel)   # Fuse vertically

    # ── Step 3 & 4: Size filter + median variance filter ─────────────────────
    panel_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_roi_mask   = np.zeros((h, w), dtype=np.uint8)
    min_panel_area = 0.003 * h * w    # Minimum 0.3% of image area = definitely a panel row

    for cnt in panel_contours:
        if cv2.contourArea(cnt) >= min_panel_area:

            # Draw this candidate blob into a temporary mask for pixel sampling
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(temp_mask, [cnt], -1, 255, -1)

            # Sample the local std-dev values at every pixel inside this candidate
            panel_std_vals     = std_map[temp_mask == 255]
            if len(panel_std_vals) > 0:
                median_lti_variance = np.median(panel_std_vals)

                # Real smooth panel surface → median std is low (< 65).
                # A bright frame line raises the mean but NOT the median.
                # This robustness to outliers is why we use the median here.
                if median_lti_variance < 65.0:
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    cv2.rectangle(raw_roi_mask, (rx, ry), (rx + rw, ry + rh), 255, -1)

    # ── Step 5: 1-pixel dilation so cell boundary edges are inside the ROI ───
    pad_kernel       = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * border_padding + 1, 2 * border_padding + 1)
    )
    final_padded_roi = cv2.dilate(raw_roi_mask, pad_kernel, iterations=1)

    # ── Visual overlay: green outlines on the cell mask ───────────────────────
    border_visual = cv2.cvtColor(low_var_mask, cv2.COLOR_GRAY2BGR)   # Convert to colour so we can draw green
    roi_cnts, _   = cv2.findContours(final_padded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in roi_cnts:
        cv2.drawContours(border_visual, [cnt], -1, (0, 255, 0), 2)   # Draw green (BGR) outline, 2px thick

    return rect_boxes_mask, border_visual, final_padded_roi


# ==============================================================================
# DSP ADD-ON 2 HELPER: Wavelet Thermal Enhancement
# ------------------------------------------------------------------------------
# This function produces an "edge energy map" using the 2D Discrete Wavelet
# Transform (DWT).  The map is used inside Block 6 only to WIDEN the set of
# candidate anomaly seed pixels — it never removes any existing detections.
#
# DSP CONCEPT — 2D DWT Multi-Resolution Decomposition:
#   One level of DWT applies two 1D filters (low-pass L and high-pass H)
#   along rows then columns, creating four sub-bands:
#     LL = L×L → approximation (smooth background structure)
#     LH = L×H → horizontal detail edges (cell row boundaries, row faults)
#     HL = H×L → vertical detail edges (cell column boundaries, string faults)
#     HH = H×H → diagonal detail (compact hotspots, corners)
#   We use 2 levels (db4 wavelet) then sum |LH₁|+|HL₁|+|HH₁| — the Level-1
#   detail energy at cell scale.  High values mark thermally anomalous edges.
#
#   Soft-thresholding (τ = 0.03) on detail coefficients suppresses noise:
#     T_soft(x, τ) = sign(x) · max(|x| − τ, 0)
#   This is Donoho–Johnstone wavelet denoising.
# ==============================================================================

def wavelet_enhance_thermal(gray_aligned):
    """
    Parameters
    ----------
    gray_aligned : uint8 — deskewed grayscale image

    Returns
    -------
    wav_energy   : uint8 — normalised wavelet edge-energy map (same size as input)
                   High values = strong cell-scale edges → likely fault regions
    """

    # Normalise to [0,1] so wavelet coefficients are in a consistent numeric range
    gf = gray_aligned.astype(np.float32) / 255.0

    # 2-level DWT decomposition using Daubechies-4 (db4) wavelet
    # db4 has 4 vanishing moments — good at capturing sharp thermal transitions
    coeffs = pywt.wavedec2(gf, wavelet='db4', level=2)
    # coeffs[0]    = LL2 approximation (coarsest level, quarter resolution)
    # coeffs[1]    = (LH1, HL1, HH1) Level-1 detail (finest, half resolution)
    # coeffs[2]    = (LH2, HL2, HH2) Level-2 detail (coarser)

    # Soft-threshold all detail sub-bands; leave the approximation LL2 unchanged
    tau     = 0.03                      # Threshold: shrink coefficients smaller than this toward zero
    cleaned = [coeffs[0]]               # Start with LL2 untouched
    for detail_tuple in coeffs[1:]:
        cleaned.append(tuple(
            pywt.threshold(d, tau, mode='soft') for d in detail_tuple
        ))                              # Apply T_soft to each of LH, HL, HH at this level

    # Level-1 detail = cell-scale features (finest spatial resolution after one DWT level)
    LH1, HL1, HH1 = coeffs[1]          # LH=horizontal edges, HL=vertical edges, HH=diagonal

    # Combine into a single saliency value: high where ANY direction has strong edges
    edge_energy = np.abs(LH1) + np.abs(HL1) + np.abs(HH1)

    # DWT output at level 1 is half the original size; resize back to original dimensions
    energy_up   = cv2.resize(edge_energy, (gray_aligned.shape[1], gray_aligned.shape[0]))

    # Normalise to 0–255 uint8 for compatibility with the rest of the pipeline
    wav_energy  = cv2.normalize(energy_up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return wav_energy


# ==============================================================================
# DSP ADD-ON 3 HELPER: IIR Butterworth Row Verification
# ------------------------------------------------------------------------------
# For any fault already classified as "Line Fault" or "Partial Shading", this
# function applies a digital high-pass filter to the 1D intensity signal of each
# row inside the fault's bounding box.  If a majority of rows show sustained
# high-pass energy (meaning the row is uniformly brighter than its neighbours),
# the line-type classification is confirmed.  If NOT confirmed, a Line Fault is
# downgraded to Hotspot (compact bright spot, not an extended stripe).
#
# DSP CONCEPT — IIR Filter via Bilinear (Tustin) Transform:
#   A 2nd-order Butterworth high-pass filter is designed by:
#   1. Starting with the analogue Butterworth prototype: H(s) = s²/(s²+√2·s+1)
#   2. Applying the bilinear transform s = 2/T·(z−1)/(z+1) to convert to
#      the discrete-time Z-domain: this is the Laplace ↔️ Z-transform connection.
#   3. scipy.signal.butter() performs both steps automatically.
#
#   Wn=0.08 sets the digital cutoff at 8% of the Nyquist frequency.
#   A sustained brightness elevation (line fault) produces large, persistent
#   high-pass energy — a localised hotspot produces only a brief spike.
# ==============================================================================

def iir_verify_line_fault(gray_aligned, fault, panel_boxes):
    """
    Parameters
    ----------
    gray_aligned : uint8    — deskewed grayscale image
    fault        : dict     — a fault record with 'box' and 'type' keys
    panel_boxes  : list     — list of (x,y,w,h) tuples from cv2.boundingRect

    Returns
    -------
    confirmed : bool — True if the row-profile analysis supports a line-type fault
    """

    # Design 2nd-order Butterworth HPF using bilinear transform
    # N=2: filter order  |  Wn=0.08: normalised digital cutoff (0=DC, 1=Nyquist)
    b, a = spsig.butter(N=2, Wn=0.08, btype='high', analog=False)
    # b, a are the numerator and denominator polynomial coefficients of H(z)

    x, y, bw, bh = fault['box']         # Bounding box of the fault region

    # Find which panel contains this fault (for the panel-width reference)
    cx_f, cy_f = x + bw / 2.0, y + bh / 2.0
    pm = next(
        (pb for pb in panel_boxes
         if pb[0] <= cx_f <= pb[0] + pb[2] and pb[1] <= cy_f <= pb[1] + pb[3]),
        None
    )
    pw = float(pm[2]) if pm else float(gray_aligned.shape[1])   # Fall back to full image width

    confirmed_rows = 0    # Count how many rows pass the IIR line-fault check

    for ri in range(y, y + bh):                            # Loop over every row inside the fault box
        row = gray_aligned[ri, x:x + bw].astype(np.float32)   # 1D pixel signal for this row
        if len(row) < 6:
            continue                                        # Skip rows that are too short to filter

        # Apply IIR filter using direct-form II transposed structure (scipy default)
        # lfilter(b, a, x) computes y[n] = b[0]·x[n]+…−a[1]·y[n−1]−a[2]·y[n−2]
        filtered = spsig.lfilter(b, a, row)
        energy   = np.abs(filtered)                         # Instantaneous filter output energy

        # A row is "hot" if more than 25% of its filtered energy exceeds the row's own
        # mean + 2σ level (meaning sustained elevation, not just a brief bright spot)
        threshold = np.mean(energy) + 2.0 * np.std(energy)
        hot_count = np.sum(energy > threshold)
        if hot_count > bw * 0.25:
            confirmed_rows += 1

    # Require at least one-third of the fault's rows to confirm the line pattern
    return confirmed_rows >= max(1, bh // 3)


# ==============================================================================
# DSP ADD-ON 4 HELPER: Fault Severity Index (FSI)
# ------------------------------------------------------------------------------
# A single maintenance-priority score in [0, 100] that combines:
#   - Spatial extent: how large is the fault relative to the total panel area?
#   - Thermal intensity: how much hotter is it than the panel's normal temperature?
#   - Fault type weight: hotspots are most critical, partial shading least.
#
# Formula:
#   FSI = 100 · w_type · (0.35·area_ratio + 0.65·intensity_ratio)
#
# The 35/65 split weights thermal intensity more than spatial size, because a
# small but very hot spot (incipient cell failure) is more urgent than a large
# warm area (mild partial shading).
# ==============================================================================

def compute_fsi(fault, gray_aligned, panel_roi_mask):
    """
    Parameters
    ----------
    fault          : dict  — fault record with 'box' and 'type' keys
    gray_aligned   : uint8 — deskewed grayscale image
    panel_roi_mask : uint8 — binary panel ROI mask from Block 5

    Returns
    -------
    fsi : float — Fault Severity Index, 0 (minor) to 100 (critical)
    """

    x, y, bw, bh = fault['box']

    # Area ratio: fault pixels ÷ total panel pixels
    fault_area  = bw * bh
    panel_area  = float(np.sum(panel_roi_mask > 0)) + 1e-5   # +1e-5 prevents division by zero
    area_ratio  = min(fault_area / panel_area, 1.0)            # Cap at 1.0

    # Intensity ratio: how much above the panel median is this fault?
    panel_pixels = gray_aligned[panel_roi_mask == 255]
    baseline     = float(np.median(panel_pixels)) if len(panel_pixels) > 0 else 128.0
    delta_T      = max(float(np.mean(gray_aligned[y:y + bh, x:x + bw])) - baseline, 0)
    max_delta    = float(np.max(gray_aligned)) - baseline + 1e-5
    intensity_r  = min(delta_T / max_delta, 1.0)               # Normalised 0–1

    # Fault type multiplier (higher = more critical failure mode)
    w_type = {'Hotspot': 1.0, 'Line Fault': 0.75, 'Partial Shading': 0.40}.get(fault['type'], 0.5)

    fsi = round(100.0 * w_type * (0.35 * area_ratio + 0.65 * intensity_r), 1)
    return fsi


# ==============================================================================
# BLOCK 6: THERMAL FAULT DETECTION & CLASSIFICATION
# ------------------------------------------------------------------------------
# This is the core detection block.  It takes the deskewed image and the panel
# ROI mask and finds thermally anomalous regions, then classifies each one as
# Hotspot, Line Fault, or Partial Shading based on brightness and shape.
#
# STEPS:
#   1. Background subtraction: medianBlur estimates the smooth background level.
#      thermal_delta = max(pixel − background, 0) isolates heat elevation.
#   2. Adaptive threshold surface: local mean + k·local_std of thermal_delta,
#      where k is auto-tuned from the panel's own contrast ratio.  Pixels above
#      this surface AND inside the panel ROI become candidate seeds.
#   3. DSP ADD-ON 2 integration: OR the top-4% wavelet edge-energy pixels
#      (inside panel) into the candidate seeds.  This catches faults that are
#      thermally subtle but texturally anomalous.
#   4. Morphological OPENING: remove isolated 1–2 pixel speckle noise.
#   5. Contour extraction & classification using brightness% and aspect ratio.
#   6. DSP ADD-ON 3: For Line Fault / Partial Shading candidates, run the
#      Butterworth IIR row-profile check.  Unconfirmed Line Faults → Hotspot.
#   7. DSP ADD-ON 4: Compute FSI for each confirmed fault; sort by FSI.
# ==============================================================================

def detect_faults_inside_roi(gray_aligned, panel_roi_mask, wav_energy):
    """
    Parameters
    ----------
    gray_aligned   : uint8 — deskewed grayscale image
    panel_roi_mask : uint8 — binary panel ROI from Block 5
    wav_energy     : uint8 — wavelet edge-energy map from wavelet_enhance_thermal()

    Returns
    -------
    thermal_delta   : float32 — background-subtracted heat elevation map (Subplot 6)
    percentage_map  : float32 — intensity % map for verified faults only (Subplot 7)
    detected_faults : list of dicts — each fault has: box, type, color, confidence,
                      fsi, iir_confirmed
    max_panel_brightness: float — brightest panel pixel (used for confidence labels)
    """

    h, w   = gray_aligned.shape
    gray_f = gray_aligned.astype(np.float32)   # Float copy for arithmetic

    # ── Step 1: Background estimation and thermal delta ───────────────────────
    # medianBlur replaces each pixel with the median of its k×k neighbourhood.
    # Because hotspots are small, they are "voted out" by their cooler neighbours,
    # giving a clean estimate of the normal background temperature at each location.
    k_size   = int(min(h, w) * 0.05) | 1       # 5% of image size, forced odd (medianBlur requires odd)
    local_bg = cv2.medianBlur(gray_aligned, k_size).astype(np.float32)

    # thermal_delta: how much hotter is each pixel vs its local background?
    # np.maximum(..., 0) keeps only positive excursions (we care about hot, not cold)
    thermal_delta = np.maximum(gray_f - local_bg, 0.0)

    # Maximum brightness in the panel area — used as the 100% reference for confidence %
    panel_raw_pixels    = gray_aligned[panel_roi_mask == 255]
    max_panel_brightness = float(np.max(panel_raw_pixels)) if len(panel_raw_pixels) > 0 else 255.0

    # ── Step 2: Adaptive threshold surface ────────────────────────────────────
    # Compute local mean and local std of thermal_delta using the same k×k window
    local_mean   = cv2.blur(thermal_delta, (k_size, k_size))
    local_sq_mean = cv2.blur(thermal_delta ** 2, (k_size, k_size))
    local_std    = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.0))

    # Contrast ratio: how variable is the thermal delta inside the panel?
    # High contrast ratio → tighter threshold (k_factor → 4.0)
    # Low contrast ratio  → looser threshold (k_factor → 2.2)
    roi_deltas    = thermal_delta[panel_roi_mask == 255]
    contrast_ratio = np.std(roi_deltas) / (np.mean(roi_deltas) + 1e-5) if len(roi_deltas) > 0 else 1.0
    k_factor      = float(np.clip(2.0 + 0.5 * contrast_ratio, 2.2, 4.0))

    # Threshold surface: T(x,y) = local_mean + clip(k·local_std, 5, 25)
    # The clip prevents the threshold from being trivially low in flat regions
    # or unreachably high in very noisy regions
    threshold_map  = local_mean + np.clip(k_factor * local_std, 5.0, 25.0)

    # Flag pixels that exceed the adaptive threshold AND are inside the panel ROI
    raw_candidates = (
        (thermal_delta >= threshold_map) & (panel_roi_mask == 255)
    ).astype(np.uint8) * 255

    # ── Step 3 (DSP ADD-ON 2): OR wavelet edge-energy seeds into candidates ───
    # Find the 96th-percentile wavelet energy value inside the panel
    if np.any(panel_roi_mask == 255):
        wav_thresh_val = int(np.percentile(wav_energy[panel_roi_mask == 255], 96))
    else:
        wav_thresh_val = 220

    # Threshold the wavelet energy map → binary mask of strongest edge pixels
    _, wav_seed_mask = cv2.threshold(wav_energy, wav_thresh_val, 255, cv2.THRESH_BINARY)
    wav_seed_mask    = cv2.bitwise_and(wav_seed_mask, panel_roi_mask)   # Keep only inside panel

    # OR into existing candidates: this can only ADD seeds, never remove existing ones
    # So the wavelet step is guaranteed not to reduce any existing detection
    raw_candidates   = cv2.bitwise_or(raw_candidates, wav_seed_mask)

    # ── Step 4: Morphological OPENING to remove tiny noise specks ─────────────
    # OPENING = EROSION then DILATION: removes isolated pixels smaller than the kernel
    open_k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_anomaly_mask = cv2.morphologyEx(raw_candidates, cv2.MORPH_OPEN, open_k)

    # ── Step 5: Extract panel bounding boxes for shape-based classification ───
    panel_cnts, _  = cv2.findContours(panel_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel_boxes    = [cv2.boundingRect(c) for c in panel_cnts if cv2.contourArea(c) > 0]

    contours, _    = cv2.findContours(clean_anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_faults = []
    filtered_mask   = np.zeros_like(clean_anomaly_mask)    # Accumulates verified fault pixels
    cnt_mask        = np.zeros_like(clean_anomaly_mask)    # Reusable single-contour mask
    min_hotspot_area = 15                                   # Reject anything smaller than 15 px²

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_hotspot_area:
            continue                                        # Too small to be a real fault

        # Draw just this single contour onto cnt_mask to extract its pixel values
        cnt_mask.fill(0)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        # Peak brightness of this anomaly region
        peak_anomaly_brightness = float(np.max(gray_aligned[cnt_mask == 255]))

        # Express as a percentage of the brightest panel pixel (100% = hottest seen)
        brightness_pct = (peak_anomaly_brightness / (max_panel_brightness + 1e-5)) * 100.0

        x, y, bw, bh = cv2.boundingRect(cnt)              # Bounding rectangle of this anomaly
        aspect_ratio  = float(bw) / bh if bh > 0 else 1.0 # width/height ratio: >1 = wide, <1 = tall
        confidence_pct = int(round(brightness_pct))         # Integer label shown on the output image

        # Find which panel box this anomaly falls inside (for relative size comparison)
        cx_f, cy_f = x + bw / 2.0, y + bh / 2.0
        p_match    = next(
            (pb for pb in panel_boxes
             if pb[0] <= cx_f <= pb[0] + pb[2] and pb[1] <= cy_f <= pb[1] + pb[3]),
            None
        )
        panel_w, panel_h = (float(p_match[2]), float(p_match[3])) if p_match else (float(w), float(h))

        # Shape criteria — does this anomaly look like an extended stripe?
        # "Severe line": very elongated AND covers ≥60% of the panel width or height
        is_severe_line = (
            (aspect_ratio >= 3.2 and bw >= 0.60 * panel_w) or
            (aspect_ratio <= 0.31 and bh >= 0.60 * panel_h)
        )
        # "Really long": moderately elongated OR area covers ≥3% of the panel area
        is_really_long = (
            (aspect_ratio >= 3.0 and bw >= 0.45 * panel_w) or
            (aspect_ratio <= 0.33 and bh >= 0.45 * panel_h) or
            (area >= 0.03 * panel_w * panel_h)
        )

        # ── Classification rules (your original logic, unchanged) ─────────────
        if 65.0 <= brightness_pct <= 80.0 and is_really_long:
            # Moderate heat + extended shape = partial shading (shadow over row)
            f_type = "Partial Shading"
            color  = (0, 165, 255)      # Orange in BGR

        elif brightness_pct > 80.0 and is_really_long:
            # High heat + extended shape = line fault (broken string or bus bar)
            f_type = "Line Fault"
            color  = (0, 255, 255)      # Yellow in BGR

        elif brightness_pct > 75.0:
            if is_severe_line:
                # Very high heat + very elongated = line fault
                f_type = "Line Fault"
                color  = (0, 255, 255)
            else:
                # Very high heat + compact shape = hotspot (failed cell)
                f_type = "Hotspot"
                color  = (0, 0, 255)    # Red in BGR

        else:
            continue    # Below all thresholds — not a real fault, skip

        # ── DSP ADD-ON 3: IIR row-profile verification for line-type faults ──
        # This runs ONLY for Line Fault and Partial Shading (not hotspots).
        # If the IIR filter does NOT confirm a sustained row-level elevation,
        # the classification is downgraded from Line Fault → Hotspot.
        iir_confirmed = None    # Default: not applicable
        if f_type in ("Line Fault", "Partial Shading"):
            fault_temp = {'box': (x, y, bw, bh), 'type': f_type}   # Temp dict for the function
            iir_confirmed = iir_verify_line_fault(gray_aligned, fault_temp, panel_boxes)

            if not iir_confirmed and f_type == "Line Fault":
                # IIR says rows are not uniformly hot → not a true line fault → reclassify
                f_type = "Hotspot"
                color  = (0, 0, 255)

        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)     # Add to verified fault mask

        # Build the final fault record dictionary
        detected_faults.append({
            'box':           (x, y, bw, bh),
            'type':          f_type,
            'color':         color,
            'confidence':    confidence_pct,
            'iir_confirmed': iir_confirmed,   # None=not tested, True/False=IIR result
        })

    # ── DSP ADD-ON 4: Compute FSI and sort by severity ────────────────────────
    for fd in detected_faults:
        fd['fsi'] = compute_fsi(fd, gray_aligned, panel_roi_mask)

    detected_faults.sort(key=lambda f: f['fsi'], reverse=True)   # Worst first

    # Build the percentage brightness map for verified fault pixels only
    percentage_map = np.zeros_like(thermal_delta, dtype=np.float32)
    valid_indices  = (filtered_mask > 0) & (panel_roi_mask == 255)
    if np.any(valid_indices):
        percentage_map[valid_indices] = (
            gray_aligned[valid_indices].astype(np.float32) /
            (max_panel_brightness + 1e-5)
        ) * 100.0

    return thermal_delta, percentage_map, detected_faults, max_panel_brightness


# ==============================================================================
# BLOCK 7: MAIN PIPELINE & 8-SUBPLOT VISUALIZATION
# ------------------------------------------------------------------------------
# Ties together all blocks in the correct order, then maps fault coordinates
# from the deskewed frame back to the original image orientation (via the
# inverse affine transform), draws annotations, and shows all diagnostic plots.
# ==============================================================================

def run_pv_inspection_pipeline(image_path):

    # ── Load image ────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)           # Load as 3-channel BGR (Blue, Green, Red)
    if img_bgr is None:
        sys.exit("Could not open thermal image.")   # Stop cleanly if file not found

    gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   # Convert to single-channel 8-bit grayscale
    h, w     = gray_raw.shape                               # Image dimensions in pixels

    # ── Block 3: Tilt alignment + FFT verification (ADD-ON 1) ─────────────────
    angle, gray_aligned, rot_mat, fft_mag = detect_and_align_tilt(gray_raw)
    # angle        = detected tilt in degrees (printed in console summary)
    # gray_aligned = straightened image used by all downstream blocks
    # rot_mat      = 2×3 rotation matrix saved for inverse-transform in annotation step
    # fft_mag      = 2D FFT log-magnitude spectrum passed to Subplot 2

    # ── Block 4: LTI variance texture mask ────────────────────────────────────
    std_map, low_var_mask = generate_cell_texture_mask(gray_aligned)
    # std_map      = local standard-deviation image (shown in Subplot 3)
    # low_var_mask = binary mask: white = smooth solar cell surface

    # ── Block 5: Panel ROI assembly ───────────────────────────────────────────
    # std_map is passed in so Block 5 can use the median variance filter
    rect_boxes_mask, border_overlay, panel_roi = build_rect_joined_roi_with_padding(
        low_var_mask, std_map
    )
    # rect_boxes_mask = individual cell rectangles before merging (Subplot 4)
    # border_overlay  = green-outlined ROI for display (Subplot 5)
    # panel_roi       = final merged + padded binary ROI mask

    # ── DSP ADD-ON 2 pre-step: Wavelet edge-energy map ────────────────────────
    wav_energy = wavelet_enhance_thermal(gray_aligned)
    # wav_energy is passed into Block 6 to widen the anomaly seed set

    # ── Block 6: Fault detection + classification (with ADD-ONs 2, 3, 4) ──────
    thermal_delta, percentage_map, faults, max_p_bright = detect_faults_inside_roi(
        gray_aligned, panel_roi, wav_energy
    )
    # thermal_delta  = background-subtracted heat map (Subplot 6)
    # percentage_map = brightness % at verified fault pixels (Subplot 7)
    # faults         = list of fault dicts, sorted by FSI (highest first)

    # ── DSP Concept — Inverse Affine Transform ─────────────────────────────────
    # All detection ran on gray_aligned (the rotated image).
    # To draw annotations on the ORIGINAL img_bgr, we need to map coordinates back.
    # cv2.invertAffineTransform computes the 2×3 matrix that undoes rot_mat.
    inv_rot_mat    = cv2.invertAffineTransform(rot_mat)
    panel_roi_orig = cv2.warpAffine(panel_roi, inv_rot_mat, (w, h),
                                    flags=cv2.INTER_NEAREST)
    # panel_roi_orig is the panel mask mapped back to the original orientation

    # ── Annotate the original image ───────────────────────────────────────────
    annotated = img_bgr.copy()     # Work on a copy so the original is not modified

    # Draw green panel outlines on the original (un-rotated) image
    panel_cnts, _ = cv2.findContours(panel_roi_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, panel_cnts, -1, (0, 255, 0), 2)

    # Counters for the summary
    counts = {"Hotspot": 0, "Line Fault": 0, "Partial Shading": 0}

    for fault in faults:
        x, y, bw, bh = fault['box']
        f_type = fault['type']
        color  = fault['color']
        fsi    = fault['fsi']
        counts[f_type] += 1

        # The fault box coordinates are in the rotated frame.
        # Transform all four corners back to the original image orientation.
        pts_aligned = np.array(
            [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
            dtype=np.float32
        )
        # Homogeneous coordinates: append a column of ones for the affine multiply
        pts_homo = np.hstack([pts_aligned, np.ones((4, 1), dtype=np.float32)])
        pts_orig = np.dot(pts_homo, inv_rot_mat.T).astype(np.int32)
        # pts_orig is now a 4×2 array of corner coordinates in original image space

        # Draw the (possibly rotated) quadrilateral bounding box
        cv2.polylines(annotated, [pts_orig], isClosed=True, color=color, thickness=2)

        # Label: fault type + confidence% + FSI score
        top_left  = pts_orig[0]
        label_text = f"{f_type} {fault['confidence']}%  FSI:{fsi}"
        cv2.putText(
            annotated, label_text,
            (top_left[0] - 2, max(top_left[1] - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1
        )

    # ── Console summary ───────────────────────────────────────────────────────
    total = sum(counts.values())
    print(f"\n{'='*65}")
    print(f"  Tilt corrected    : {angle:.2f}°")
    print(f"  Panel max bright  : {max_p_bright:.1f}")
    print(f"  Total faults      : {total}  "
          f"(HS:{counts['Hotspot']}  LF:{counts['Line Fault']}  PS:{counts['Partial Shading']})")
    print(f"{'─'*65}")
    if faults:
        print("  Fault list (sorted by severity — highest FSI first):")
        for i, f in enumerate(faults, 1):
            iir_str = f"  IIR={'✓' if f['iir_confirmed'] else '✗'}" if f['iir_confirmed'] is not None else ""
            print(f"  {i:2d}. {f['type']:<16}  conf={f['confidence']:3d}%  "
                  f"FSI={f['fsi']:5.1f}{iir_str}")
    print(f"{'='*65}\n")

    # ── 8-Subplot Diagnostic Dashboard ───────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor('#0d0d1a')         # Dark background for the figure window
    for ax in axes.flat:
        ax.set_facecolor('#0d0d1a')
        ax.axis('off')

    # ── Subplot 1: Raw input ──────────────────────────────────────────────────
    axes[0, 0].imshow(gray_raw, cmap='gray')
    axes[0, 0].set_title('1. Raw Thermal Input', color='white', fontweight='bold')
    # Shows the original untouched grayscale image exactly as loaded from disk

    # ── Subplot 2: 2D FFT magnitude spectrum (DSP ADD-ON 1 visual) ───────────
    axes[0, 1].imshow(fft_mag, cmap='inferno')
    axes[0, 1].set_title(f'2. FFT Spectrum (tilt check)\n'
                          f'   FFT and projection scan agree: {angle:.1f}°',
                          color='white', fontweight='bold', fontsize=8)
    # Bright lobes in the spectrum correspond to the panel grid's spatial frequency.
    # Their angle relative to the centre confirms the detected tilt.

    # ── Subplot 3: LTI variance map ───────────────────────────────────────────
    axes[0, 2].imshow(std_map, cmap='viridis')
    axes[0, 2].set_title('3. LTI Variance Map\n   (dark = smooth panel surface)',
                          color='white', fontweight='bold', fontsize=8)
    # Dark areas have low local standard deviation → smooth solar cells.
    # Bright areas are rough background or edges.

    # ── Subplot 4: Individual cell rectangles before merging ─────────────────
    axes[0, 3].imshow(rect_boxes_mask, cmap='gray')
    axes[0, 3].set_title('4. Cell Rect Mask\n   (before morphological joining)',
                          color='white', fontweight='bold', fontsize=8)
    # Shows each smooth cell region as a white rectangle — scattered, not yet merged.

    # ── Subplot 5: Merged + padded panel ROI with green outlines ──────────────
    axes[1, 0].imshow(border_overlay)
    axes[1, 0].set_title('5. Panel ROI (green outlines)\n'
                          '   [morph. close → median-var filter → 1px pad]',
                          color='white', fontweight='bold', fontsize=8)
    # Green boxes show the final detected panel regions.
    # Everything outside these boxes is ignored in fault detection.

    # ── Subplot 6: Thermal delta (background-subtracted heat map) ─────────────
    axes[1, 1].imshow(thermal_delta, cmap='magma')
    axes[1, 1].set_title('6. Thermal Delta Map\n'
                          '   [pixel − local background; brighter = hotter anomaly]',
                          color='white', fontweight='bold', fontsize=8)
    # Shows only the heat ELEVATION above the local background.
    # The bright white spots are genuine thermal anomalies.

    # ── Subplot 7: Wavelet energy map + verified fault intensity ──────────────
    # Blend the wavelet edge-energy map (as a green tint) with the percentage map
    pct_display  = cv2.normalize(percentage_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    wav_display  = cv2.normalize(wav_energy,     None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Stack as a 3-channel image: R=faults, G=wavelet energy, B=zeros
    overlay_rgb  = np.stack([pct_display, wav_display, np.zeros_like(pct_display)], axis=-1)
    axes[1, 2].imshow(overlay_rgb)
    axes[1, 2].set_title('7. Verified Faults (red) +\n'
                          '   Wavelet Edge Map (green) [ADD-ON 2]',
                          color='white', fontweight='bold', fontsize=8)
    # Red = confirmed fault pixels.  Green = wavelet high-edge-energy regions.
    # Overlap shows where wavelet analysis supports the thermal detection.

    # ── Subplot 8: Annotated final output ─────────────────────────────────────
    axes[1, 3].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title(
        f'8. Final Output\n'
        f'   [HS:{counts["Hotspot"]}  LF:{counts["Line Fault"]}  PS:{counts["Partial Shading"]}]'
        f'  — labels show FSI score',
        color='white', fontweight='bold', fontsize=8
    )
    # The final annotated image in the original (un-rotated) orientation.
    # Each fault box shows: type name, brightness confidence%, and FSI score.

    # Fault summary text box inside Subplot 8
    summary_lines = [
        f"Total: {total}",
        f"HS:{counts['Hotspot']}  LF:{counts['Line Fault']}  PS:{counts['Partial Shading']}",
        "─" * 28,
        "Ranked by FSI (worst first):",
    ]
    for i, f in enumerate(faults[:7], 1):
        iir_tag = f"  IIR={'OK' if f['iir_confirmed'] else 'no'}" if f['iir_confirmed'] is not None else ""
        summary_lines.append(f" {i}. {f['type'][:12]:<12} FSI={f['fsi']:5.1f}{iir_tag}")
    if not faults:
        summary_lines.append("  No faults detected.")

    axes[1, 3].text(
        0.02, 0.02,
        "\n".join(summary_lines),
        transform=axes[1, 3].transAxes,
        fontsize=7.5, color='white', verticalalignment='bottom', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.80)
    )

    # Figure title listing all DSP concepts used
    plt.suptitle(
        'Thermal PV Fault Detection  ·  '
        'DSP techniques: LTI Variance  |  2D DFT + Windowing  |  '
        '2D DWT Wavelet  |  IIR Butterworth (Bilinear Transform)  |  FSI',
        color='white', fontsize=9, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    plt.savefig('pv_fault_detection_output.jpg', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()


# ==============================================================================
# ENTRY POINT
# Python only runs the code below when this file is executed directly (not when
# it is imported by another script).
# ==============================================================================

if __name__ == "__main__":
    print("Select a thermal image to analyse...")
    file_path = select_image()
    if file_path:
        run_pv_inspection_pipeline(file_path)
    else:
        print("No file selected — exiting.")