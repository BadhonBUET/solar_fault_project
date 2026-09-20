# ==============================================================================
# Thermal Image Analysis System for Photovoltaic Fault Detection   (v3)
# ------------------------------------------------------------------------------
# PIPELINE
#
#   1. TILT          coarse-to-fine projection-variance scan  ->  2-D FFT verification
#   2. PANEL AREA    LTI moving-variance maps (coarse window = area, fine window = borders),
#                    cut automatically with Otsu on a LOG scale
#   3. PANEL EDGES   1-level 2-D DWT edge bands (horizontal / vertical borders)
#   4. PANEL ROI     variance blobs -> morphological closing -> INSIDE CHECK
#                    (grow / trim / split / verify every row) -> DWT border snap -> +1 px pad
#   5. NOISE + HEAT  3x3 median noise removal inside the panel, thermal-delta map
#                    (pixel - local panel background), static row structure removed
#   6. HARD WAVELET  hard-threshold DWT of the heat map inside the ROI; two outputs:
#                      - thermal anomaly       (de-noised heat above threshold)
#                      - abrupt brightness increase (surviving detail coefficients)
#                    BOTH are marked as anomalies
#   7. CLASSIFY      squarish / small  -> Hotspot
#                    long              -> Line Fault
#                    long + lighter    -> Partial Shading
#
# THRESHOLDS
#   No fixed grey-level numbers.  Every threshold is DERIVED from the image being
#   processed (noise sigma, brightness range, cell size, Otsu split, ...).  `Config`
#   holds only RELATIVE knobs (sigma multipliers, fractions, cell-size multiples).
#   The dashboard has TWO sliders, which are all you normally need:
#       Sensitivity       right = catch weaker anomalies, left = only strong ones
#       Panel ROI cutoff  right = looser (larger ROI),   left = stricter (tighter ROI)
#   The derived absolute values are printed to the console on every update.
#
# Required packages:   pip install opencv-python numpy scipy PyWavelets matplotlib
# Run:                 python pv_fault_detection.py [image_path] [--no-tuner]
# ==============================================================================


# ==============================================================================
# BLOCK 1: IMPORTING LIBRARIES
# ==============================================================================

import sys                                   # Exit gracefully on fatal errors
import copy                                  # Deep-copy the config for the live tuner
import argparse                              # Optional command-line image path
from dataclasses import dataclass, field     # Clean, editable configuration containers
from types import SimpleNamespace            # Lightweight "bag of results" objects

import cv2                                   # OpenCV - image I/O + all computer-vision operations
import numpy as np                           # NumPy - matrix maths
import pywt                                  # PyWavelets - 2-D Discrete Wavelet Transform
import scipy.signal as spsig                 # SciPy Signal - IIR (Butterworth) filter design
from scipy import ndimage as ndi             # SciPy ndimage - nearest-panel-pixel fill
import matplotlib.pyplot as plt              # Matplotlib - diagnostic dashboard
from matplotlib.patches import Polygon       # Draw rotated fault boxes on the plots
from matplotlib.widgets import Slider, Button   # Live threshold tuner

try:                                         # Tkinter is only needed for the file-picker pop-up
    import tkinter as tk
    from tkinter import filedialog
except ImportError:                          # (headless machines: pass the path on the command line)
    tk, filedialog = None, None


# ==============================================================================
# CONFIGURATION  -  the ONLY place you need to edit
# ------------------------------------------------------------------------------
# Every value below is RELATIVE (a multiplier, a fraction, a number of cells).
# The absolute threshold used on a given image is computed from that image.
# ==============================================================================

@dataclass
class TiltCfg:
    coarse_range_deg: float = 45.0      # coarse scan covers  -R ... +R degrees
    coarse_step_deg: float = 1.0        # coarse scan step
    fine_half_range_deg: float = 2.0    # fine scan covers  coarse_winner +/- this
    fine_step_deg: float = 0.1          # fine scan step
    analysis_max_side: int = 640        # angle search runs on a copy no larger than this (speed)
    fft_r_min: float = 0.012            # FFT ring: inner radius (cycles/pixel) - skips DC blob
    fft_r_max: float = 0.30             # FFT ring: outer radius (cycles/pixel)
    agree_tol_deg: float = 3.0          # FFT and projection estimates "agree" inside this


@dataclass
class RoiCfg:
    var_window_frac: float = 0.03       # coarse LTI window (panel AREA)  = this * min(H, W)  (0.03 -> 15 px on 512 px)
    fine_window_frac: float = 0.01      # fine LTI window (panel BORDERS / inside check) = this * min(H, W)
    var_thresh_scale: float = 1.0       # PANEL ROI CUTOFF: x the automatic (Otsu) cut.  <1 stricter, >1 looser
    auto_cutoff: bool = True            # if the ROI is empty / covers almost everything, retry other cutoffs
    inside_frac: float = 0.5            # a row/column is "panel" if it is >= this * the panel's own smoothness
    dark_frac: float = 0.10             # ignore pixels darker than this * (99.5th percentile brightness)
    wavelet: str = 'db4'                # mother wavelet for the panel-edge DWT
    edge_z: float = 4.0                 # a DWT border must exceed  edge_z * robust-noise  to be accepted
    snap_to_edges: bool = True          # extend ROI sides outward onto the DWT panel borders
    close_h_cells: float = 2.5          # horizontal closing kernel = this * cell width (bridges module gaps / faults)
    gap_split_cells: float = 1.2        # a non-panel column band wider than this * cell width splits a row in two
    min_panel_cells: float = 3.0        # smallest ROI blob kept, in cell footprints
    min_smooth_frac: float = 0.35       # a candidate must be at least this fraction smooth surface
    pad_px: int = 1                     # final padding ("1 px add-on")


@dataclass
class ThermalCfg:
    bg_window_cells: float = 4.0        # background median window = this * cell width
    margin_cells: float = 0.06          # ROI rim ignored for detection = this * cell width (min 2 px)
    sensitivity: float = 1.0            # global detection gain.  >1 more sensitive (thresholds / gain)
    z_hi: float = 5.0                   # seed threshold   = z_hi * noise sigma
    z_lo: float = 2.5                   # growth threshold = z_lo * noise sigma
    min_contrast_frac: float = 0.06     # ...but never below this fraction of the image dynamic range
    hyst_ratio: float = 0.5             # growth threshold is at least this * seed threshold
    wavelet: str = 'db4'                # wavelet for the hard-threshold enhancement
    wav_level: int = 2                  # decomposition depth
    wav_tau_scale: float = 0.7          # hard threshold = scale * sigma * sqrt(2 ln N)  (Donoho universal)
    wav_gain: float = 1.0               # >1 boosts the surviving (abrupt) detail coefficients
    abrupt_z: float = 2.0               # abrupt-increase map must exceed abrupt_z * sigma to count as anomaly
    denoise: bool = True                # 3x3 median inside the panel before the heat map (removes salt & pepper)
    frag_gap_cells: float = 0.5         # bridge gaps up to this * cell width ALONG a row (cell gaps cut a stripe)
    frag_gap_v_cells: float = 0.15      # ...but only this much across rows (keeps neighbouring faults apart)
    peak_radius_cells: float = 0.25     # growth is judged against the strongest pixel within this * cell width
    region_peak_frac: float = 0.4       # a fault keeps only pixels >= this * its own peak (stops halo bloat)
    min_region_cells: float = 0.01      # smallest fault kept, in cell footprints (min 6 px)


@dataclass
class ClassCfg:
    aspect_long: float = 3.0            # long side / short side needed to count as "long"
    min_len_h_cells: float = 1.5        # a horizontal line must be >= this many cell WIDTHS long
    min_len_v_cells: float = 0.5        # a vertical line must be >= this many cell HEIGHTS long
    ps_max_heat: float = 0.5            # long + heat below this = Partial Shading, else Line Fault
    ref_floor_frac: float = 0.4         # heat reference is at least this * image dynamic range


@dataclass
class FsiCfg:
    w_area: float = 0.35                # weight of spatial extent in the FSI
    w_intensity: float = 0.65           # weight of thermal intensity in the FSI
    full_area_cells: float = 2.0        # a fault covering this many cell footprints = full area score
    type_weight: dict = field(default_factory=lambda: {
        'Hotspot': 1.0, 'Line Fault': 0.75, 'Partial Shading': 0.40})


@dataclass
class IirCfg:
    enabled: bool = True                # run the Butterworth continuity check on long faults
    order: int = 2                      # Butterworth order
    cutoff: float = 0.20                # normalised digital cut-off (1.0 = Nyquist)
    min_cover: float = 0.90             # fraction of the fault length that must stay hot
    can_reclassify: bool = False        # True: an unconfirmed Line Fault is downgraded to Hotspot


@dataclass
class Config:
    tilt: TiltCfg = field(default_factory=TiltCfg)
    roi: RoiCfg = field(default_factory=RoiCfg)
    thermal: ThermalCfg = field(default_factory=ThermalCfg)
    cls: ClassCfg = field(default_factory=ClassCfg)
    fsi: FsiCfg = field(default_factory=FsiCfg)
    iir: IirCfg = field(default_factory=IirCfg)


# Fault colours (BGR for OpenCV) and short labels
FAULT_COLOR = {'Hotspot': (0, 0, 255), 'Line Fault': (0, 255, 255), 'Partial Shading': (0, 165, 255)}
FAULT_ABBR = {'Hotspot': 'HS', 'Line Fault': 'LF', 'Partial Shading': 'PS'}


def odd(n, lo=3):
    """Round to the nearest odd integer >= lo (kernel sizes must be odd)."""
    n = max(int(round(n)), lo)
    return n if n % 2 == 1 else n + 1


# ==============================================================================
# BLOCK 2: IMAGE FILE SELECTION
# ==============================================================================

def select_image():
    root = tk.Tk()                       # Hidden Tk window (required by the dialog)
    root.withdraw()
    root.attributes('-topmost', True)    # Dialog appears in front of other windows
    file_path = filedialog.askopenfilename(
        title="Select Solar Thermal Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return file_path


# ==============================================================================
# BLOCK 3: TILT ALIGNMENT   (coarse -> fine scan, then FFT verification)
# ------------------------------------------------------------------------------
# A) Coarse scan  : 91 angles, -45..+45 deg, 1 deg steps
# B) Fine scan    : 41 angles around the coarse winner, 0.1 deg steps
#    Score        : var(row means) + var(column means).  Horizontal panel rows
#                   make the row-mean signal swing strongly -> large variance.
# C) Final rotate : winning angle, cubic interpolation, on the FULL image.
# D) FFT verify   : Hanning-windowed 2-D DFT.  A tilted grid puts its spectral
#                   energy on two perpendicular streaks; the streak direction is
#                   the tilt.  Instead of trusting one brightest pixel, the
#                   log-magnitude is integrated over a ring at every angle
#                   (angular energy profile), folded modulo 90 deg, and the peak
#                   of that profile is the FFT tilt estimate.
#
# Improvements over v1: the angle search runs on a downscaled copy (same angle,
# much faster); the FFT estimate uses the whole angular profile (robust); a
# validity mask of the rotated frame is returned so replicated border pixels are
# never mistaken for "smooth panel surface" downstream.
# ==============================================================================

def _projection_score(img, angle, cx, cy):
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return float(np.var(np.mean(rot, axis=1)) + np.var(np.mean(rot, axis=0)))


def fft_tilt_estimate(gray, tcfg):
    """Independent tilt estimate from the Hanning-windowed 2-D DFT (returns angle, log-magnitude)."""
    h, w = gray.shape
    g = gray.astype(np.float32)
    g -= g.mean()                                              # remove DC so it cannot dominate
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)   # taper edges -> no spectral leakage
    F = np.fft.fftshift(np.fft.fft2(g * win))                  # 2-D DFT, DC moved to the centre
    mag = np.log1p(np.abs(F)).astype(np.float32)               # log-magnitude spectrum

    # Sample the spectrum on a polar grid: 720 angles (0.5 deg) x 96 radii (physical frequency units,
    # so non-square images keep true angles).
    n_ang = 720
    theta = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    rho = np.linspace(tcfg.fft_r_min, tcfg.fft_r_max, 96)
    mx = (w // 2 + np.outer(np.cos(theta), rho) * w).astype(np.float32)
    my = (h // 2 + np.outer(np.sin(theta), rho) * h).astype(np.float32)
    polar = cv2.remap(mag, mx, my, cv2.INTER_LINEAR)
    profile = polar.mean(axis=1)                               # angular energy profile (0.5 deg bins)

    fold = profile.reshape(4, n_ang // 4).sum(axis=0)          # a grid has energy every 90 deg -> fold
    ext = np.concatenate([fold[-4:], fold, fold[:4]])          # circular smoothing (9 bins)
    fold = np.convolve(ext, np.ones(9) / 9.0, mode='same')[4:-4]
    peak_deg = float(np.argmax(fold)) * 0.5                    # 0 .. 90 deg
    fft_angle = ((peak_deg + 45.0) % 90.0) - 45.0              # wrap into [-45, +45)
    return fft_angle, mag


def detect_and_align_tilt(gray, tcfg):
    """
    Returns SimpleNamespace(angle, fft_angle, agree, gray_aligned, valid, rot_mat, fft_mag)
      angle        : tilt (deg) chosen by the projection scan - this one is applied
      fft_angle    : independent FFT estimate (verification only)
      valid        : uint8 mask, 255 where the rotated frame holds real image data
    """
    h, w = gray.shape
    scale = min(1.0, tcfg.analysis_max_side / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(gray, (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
    sh, sw = small.shape
    scx, scy = sw // 2, sh // 2

    # ---- Step A: coarse scan ------------------------------------------------
    R = tcfg.coarse_range_deg
    coarse = np.linspace(-R, R, int(round(2 * R / tcfg.coarse_step_deg)) + 1)
    best_angle, max_score = 0.0, -1.0
    for a in coarse:
        s = _projection_score(small, a, scx, scy)
        if s > max_score:
            max_score, best_angle = s, float(a)

    # ---- Step B: fine scan around the coarse winner -------------------------
    fh = tcfg.fine_half_range_deg
    n_fine = int(round(2 * fh / tcfg.fine_step_deg)) + 1
    for a in np.linspace(best_angle - fh, best_angle + fh, n_fine):
        s = _projection_score(small, a, scx, scy)
        if s > max_score:
            max_score, best_angle = s, float(a)

    # ---- Step C: final rotation of the FULL image ---------------------------
    rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
    gray_aligned = cv2.warpAffine(gray, rot_mat, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), rot_mat, (w, h),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))          # 2 px safety rim

    # ---- Step D: FFT verification (does NOT change the angle) ---------------
    fft_angle, fft_mag = fft_tilt_estimate(small, tcfg)
    diff = ((fft_angle - best_angle + 45.0) % 90.0) - 45.0       # difference modulo 90 deg
    agree = abs(diff) <= tcfg.agree_tol_deg

    return SimpleNamespace(angle=best_angle, fft_angle=fft_angle, agree=agree,
                           gray_aligned=gray_aligned, valid=valid,
                           rot_mat=rot_mat, fft_mag=fft_mag)


# ==============================================================================
# BLOCK 4: PANEL AREA  -  LTI MOVING-VARIANCE MAP
# ------------------------------------------------------------------------------
# A box blur is an LTI system with a rectangular impulse response, so
#     Var(X) = E[X^2] - (E[X])^2      (two box filters)
# gives the local variance.  Solar-cell glass is smooth (low variance); gravel,
# soil and vegetation are rough (high variance).
#
# v1 normalised by the image MAXIMUM and cut at a fixed 45/255, so one bright
# edge pixel changed the meaning of the threshold for the whole image.
# v2 normalises by the 99.5th percentile and places the cut with Otsu's method
# (the split that best separates the two populations "smooth" / "rough"), so the
# threshold follows the image.  `var_thresh_scale` shifts it if you want.
# ==============================================================================

def _local_std(gf, k):
    """Local standard deviation with a k x k box window (two LTI box filters)."""
    mean = cv2.blur(gf, (k, k))                                # E[X]
    mean_sq = cv2.blur(gf * gf, (k, k))                        # E[X^2]
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))     # sqrt(E[X^2] - E[X]^2)


def _log_otsu(std, usable):
    """
    Map the std image to 0..255 on a LOG scale and cut it with Otsu's method.
    Log scale matters: smooth panel (std ~3-10) and rough gravel (std ~30-60) are
    far apart in log units, while a few hot edges with std ~100+ can no longer
    drag the cut into the middle of the gravel (which is what a linear scale did).
    """
    ls = np.log1p(std)
    v = ls[usable] if usable.any() else ls.ravel()
    lo, hi = np.percentile(v, [0.5, 99.5])
    hi = max(float(hi), float(lo) + 1e-3)
    m = np.clip((ls - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if usable.sum() > 100:
        T, _ = cv2.threshold(m[usable].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        T = 128.0
    return m, float(T)


def lti_variance_mask(gray, valid, rcfg):
    """
    Two LTI variance maps of the same image:
      coarse window -> where the panel AREA is       (low_var_mask, std_map)
      fine   window -> where the smooth surface ends (fine_smooth) - used to check
                       the inside of every ROI and to place its borders precisely
    Both are cut automatically (Otsu on log-variance); `var_thresh_scale` (the
    "Panel ROI cutoff" slider) moves both cuts together.
    """
    h, w = gray.shape
    k = odd(rcfg.var_window_frac * min(h, w), lo=5)            # coarse window follows image size
    kf = odd(rcfg.fine_window_frac * min(h, w), lo=3)          # fine window
    gf = gray.astype(np.float32)

    vb = valid > 0
    bright_ref = float(np.percentile(gray[vb], 99.5)) if vb.any() else 255.0
    usable = vb & (gray > rcfg.dark_frac * bright_ref)         # ignore black frame / empty corners

    std_map, T0 = _log_otsu(_local_std(gf, k), usable)
    T = float(np.clip(T0 * rcfg.var_thresh_scale, 1.0, 254.0))
    low_var_mask = (((std_map <= T) & usable).astype(np.uint8)) * 255

    fine_map, Tf0 = _log_otsu(_local_std(gf, kf), usable)
    Tf = float(np.clip(Tf0 * rcfg.var_thresh_scale, 1.0, 254.0))
    fine_smooth = (fine_map <= Tf) & usable
    b = kf // 2 + 1                                            # window reflects at the image border ->
    fine_smooth[:b, :] = False; fine_smooth[-b:, :] = False    # the outermost pixels are not trusted
    fine_smooth[:, :b] = False; fine_smooth[:, -b:] = False

    return SimpleNamespace(std_map=std_map, low_var_mask=low_var_mask, T=T, k=k,
                           usable=usable, fine_smooth=fine_smooth, kf=kf, Tf=Tf)


def _cell_pitch(gray, low_var_mask):
    """
    Horizontal cell pitch from the AUTOCORRELATION of the panel's column-mean profile
    (the dark cell gaps repeat every `pitch` pixels).  Works even when the gaps are too
    faint to split the variance mask into separate cells.  Returns None if no periodicity.
    """
    h, w = gray.shape
    rows = (low_var_mask > 0).mean(axis=1) >= 0.25                # rows that belong to a panel
    if rows.sum() < 8:
        return None
    prof = gray[rows].astype(np.float32).mean(axis=0)
    trend = cv2.blur(prof.reshape(1, -1), (max(9, w // 8) | 1, 1)).ravel()
    prof = prof - trend                                            # remove slow gradients
    prof -= prof.mean()
    if float(np.dot(prof, prof)) < 1e-6:
        return None
    f = np.fft.rfft(prof, n=2 * w)
    ac = np.fft.irfft(f * np.conj(f))[:w]                          # Wiener-Khinchin autocorrelation
    ac /= ac[0]
    lo, hi = 6, int(0.12 * w)
    for lag in range(lo, min(hi, w - 2)):                          # FIRST clear peak = one cell pitch
        if ac[lag] >= 0.15 and ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]:
            return float(lag)
    return None


def estimate_cell_geometry(low_var_mask, k, gray=None):
    """Cell footprint (w x h).  Every later size/kernel is a multiple of it."""
    h, w = low_var_mask.shape
    clean = cv2.morphologyEx(low_var_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    clean = cv2.dilate(clean, np.ones((k, k), np.uint8))       # undo the k/2 retraction of the variance window
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(0.0005 * h * w, 1.5 * k * k)               # ignore specks
    ws, hs = [], []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area and bw >= 3 and bh >= 3:
            ws.append(bw)
            hs.append(bh)
    if len(ws) >= 3:
        cw, ch = float(np.median(ws)), float(np.median(hs))
    else:                                                      # fallback when no cell pattern is visible
        cw, ch = 0.05 * w, 0.12 * h
    pitch = _cell_pitch(gray, low_var_mask) if gray is not None else None
    if pitch is not None:
        cw = pitch                                             # periodicity beats blob width
    cw = float(np.clip(cw, 0.015 * w, 0.12 * w))
    ch = float(np.clip(ch, 0.015 * h, 0.25 * h))
    return SimpleNamespace(cell_w=cw, cell_h=ch, n_blobs=len(ws), pitch=pitch)


# ==============================================================================
# BLOCK 5: PANEL EDGES (2-D DWT)  +  PANEL ROI
# ------------------------------------------------------------------------------
# The variance map tells us WHERE the smooth panel surface is, but the moving
# window makes it retract a few pixels from the true panel border.  The 2-D DWT
# tells us exactly where the straight borders are:
#
#   cH (horizontal-detail band) responds to horizontal edges (top/bottom borders)
#   cV (vertical-detail band)   responds to vertical edges   (left/right borders, cell gaps)
#
# Each band is rebuilt alone with the inverse DWT (so it is pixel-aligned with the
# image).  A straight border adds up coherently when the band is averaged along
# its length, random gravel texture cancels - so the border shows as a clear
# spike in a 1-D profile, judged against the profile's own robust noise level.
#
# ROI construction
#   1. bounding rectangle of every smooth-cell blob
#   2. morphological CLOSING (kernel from the cell size) fuses cells into rows
#   3. keep blobs that are big enough (in cells) and mostly smooth surface
#   4. INSIDE CHECK  (fine-window LTI smoothness profile of every row / column):
#        - grow onto rows/columns that are still smooth (the coarse window retracts)
#        - trim rows/columns that are NOT smooth (gravel, gaps, frame lines)
#        - split a candidate where a non-panel band cuts through it (two rows that
#          were merged, gravel strip between them)
#        - drop candidates that are mostly not smooth
#   5. each side is moved out to the nearest DWT border (small reach only)
#   6. +1 px padding
# The verified smooth interior ("core", before step 5-6) is what the thermal
# analysis uses, so frame rims / gaps can never become false anomalies.
# ==============================================================================

def dwt_edge_bands(gray, wavelet):
    """Horizontal-edge band Dh and vertical-edge band Dv of a 1-level 2-D DWT, at full resolution."""
    h, w = gray.shape
    gf = gray.astype(np.float32) / 255.0
    ph, pw = h % 2, w % 2
    if ph or pw:
        gf = np.pad(gf, ((0, ph), (0, pw)), mode='edge')       # DWT wants even sizes
    cA, (cH, cV, cD) = pywt.dwt2(gf, wavelet, mode='periodization')
    z = np.zeros_like(cA)
    Dh = pywt.idwt2((z, (cH, z, z)), wavelet, mode='periodization')[:h, :w].astype(np.float32)
    Dv = pywt.idwt2((z, (z, cV, z)), wavelet, mode='periodization')[:h, :w].astype(np.float32)
    return Dh, Dv


def _scan_outward(p, pos, step, reach, thr):
    """Walk outward from `pos`; return the peak of the first profile value above `thr` (else `pos`)."""
    n = len(p)
    for d in range(reach + 1):
        i = pos + step * d
        if i < 0 or i >= n:
            break
        if p[i] >= thr:
            while 0 <= i + step < n and p[i + step] > p[i] and abs(i + step - pos) <= reach + 2:
                i += step                                      # climb to the local maximum
            return i
    return pos


def _snap_rect(rect, Dh, Dv, reach, z):
    """Move each side of `rect` outward (max `reach` px) onto the nearest coherent DWT border."""
    x, y, bw, bh = rect
    h, w = Dh.shape
    x2, y2 = x + bw, y + bh
    ph = np.abs(Dh[:, x:x2].mean(axis=1))          # horizontal-border strength per row (signed mean -> coherent)
    pv = np.abs(Dv[y:y2, :].mean(axis=0))          # vertical-border strength per column
    thr_h = z * np.median(ph) / 0.6745 + 1e-9      # robust noise of the profile (half-normal median)
    thr_v = z * np.median(pv) / 0.6745 + 1e-9
    ny1 = _scan_outward(ph, y, -1, reach, thr_h)
    ny2 = _scan_outward(ph, y2 - 1, +1, reach, thr_h) + 1
    nx1 = _scan_outward(pv, x, -1, reach, thr_v)
    nx2 = _scan_outward(pv, x2 - 1, +1, reach, thr_v) + 1
    return nx1, ny1, nx2 - nx1, ny2 - ny1


def _runs(mask):
    """(start, end_exclusive) of every run of True in a 1-D boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def _refine_rect(rect, lti, geom, rcfg, reach):
    """
    INSIDE CHECK of one candidate rectangle using the fine-window smoothness map.
    Returns a list of verified rectangles (0, 1 or several).
    """
    fs = lti.fine_smooth.astype(np.float32)
    H, W = fs.shape
    x, y, bw, bh = rect
    x, y = max(int(x), 0), max(int(y), 0)
    x2, y2 = min(int(x + bw), W), min(int(y + bh), H)
    if x2 - x < 3 or y2 - y < 3:
        return []
    frac = rcfg.inside_frac

    # ---- rows: grow onto smooth rows, then cut away every non-panel band ----
    row = fs[:, x:x2].mean(axis=1)                       # smooth fraction of each row
    thr = frac * float(np.median(row[y:y2]))             # relative to THIS panel's own smoothness
    for _ in range(reach):
        if y > 0 and row[y - 1] >= thr:
            y -= 1
        else:
            break
    for _ in range(reach):
        if y2 < H and row[y2] >= thr:
            y2 += 1
        else:
            break
    runs = []
    for a, b in _runs(row[y:y2] >= thr):                 # runs of panel rows, tolerate 2-row glitches
        if runs and a - runs[-1][1] <= 2:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    min_h = max(6, int(0.25 * geom.cell_h))

    out = []
    for a, b in runs:
        ya, yb = y + a, y + b
        if yb - ya < min_h:
            continue
        # ---- columns of this band: grow the ends, then cut at wide non-panel gaps
        col = fs[ya:yb, :].mean(axis=0)
        thr_c = frac * float(np.median(col[x:x2]))
        xa, xb = x, x2
        for _ in range(reach):
            if xa > 0 and col[xa - 1] >= thr_c:
                xa -= 1
            else:
                break
        for _ in range(reach):
            if xb < W and col[xb] >= thr_c:
                xb += 1
            else:
                break
        cruns = []
        gap_split = max(int(rcfg.gap_split_cells * geom.cell_w), 2 * lti.k)
        for a2, b2 in _runs(col[xa:xb] >= thr_c):        # cell gaps / module gaps are narrow -> merged
            if cruns and a2 - cruns[-1][1] <= gap_split:
                cruns[-1] = (cruns[-1][0], b2)
            else:
                cruns.append((a2, b2))
        for a2, b2 in cruns:
            xs, xe = xa + a2, xa + b2
            if xe - xs >= max(6, int(0.5 * geom.cell_w)) and fs[ya:yb, xs:xe].mean() >= 0.4:
                out.append((xs, ya, xe - xs, yb - ya))   # verified: mostly smooth surface
    return out


def build_panel_roi(gray, lti, geom, rcfg):
    h, w = gray.shape
    cw, ch = geom.cell_w, geom.cell_h

    # ---- Step 1: solid rectangle around every smooth-cell blob -------------
    contours, _ = cv2.findContours(lti.low_var_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_rects = np.zeros((h, w), np.uint8)
    min_speck = max(15.0, 0.02 * cw * ch)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_speck:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            cv2.rectangle(cell_rects, (rx, ry), (rx + rw, ry + rh), 255, -1)

    # ---- Step 2: morphological closing (kernels scale with the cell size) ---
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (odd(rcfg.close_h_cells * cw), 3))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, odd(0.07 * ch, lo=5)))
    closed = cv2.morphologyEx(cell_rects, cv2.MORPH_CLOSE, kh)      # fuse cells along the row
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kv)          # fuse small vertical gaps

    # ---- Step 3: candidate blobs (size in cells + median variance) ----------
    Dh, Dv = dwt_edge_bands(gray, rcfg.wavelet)
    reach = int(lti.k // 2 + 3)                                     # coarse window retracts ~k/2
    reach_snap = int(lti.kf + 1)                                    # DWT border search: a few px only
    min_panel_area = max(rcfg.min_panel_cells * cw * ch, 0.0005 * h * w)
    panel_cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    core_rects, rects, blob = [], [], np.zeros((h, w), np.uint8)
    for cnt in panel_cnts:
        if cv2.contourArea(cnt) < min_panel_area:
            continue
        blob.fill(0)
        cv2.drawContours(blob, [cnt], -1, 255, -1)
        if lti.fine_smooth[blob == 255].mean() < rcfg.min_smooth_frac:
            continue                                                # mostly rough -> not a panel
        # ---- Step 4: inside check (grow / trim / split / verify) ----------
        for rect in _refine_rect(cv2.boundingRect(cnt), lti, geom, rcfg, reach):
            if rect[2] * rect[3] < 0.5 * min_panel_area:
                continue
            core_rects.append(rect)
            # ---- Step 5: move the sides onto the DWT panel borders --------
            rects.append(_snap_rect(rect, Dh, Dv, reach_snap, rcfg.edge_z) if rcfg.snap_to_edges else rect)

    # ---- Step 6: draw ROI + core, add the 1 px pad ---------------------------
    def draw(rr):
        m = np.zeros((h, w), np.uint8)
        for (rx, ry, rw, rh) in rr:
            cv2.rectangle(m, (max(rx, 0), max(ry, 0)), (min(rx + rw, w) - 1, min(ry + rh, h) - 1), 255, -1)
        return m
    roi = draw(rects)
    core = draw(core_rects)
    pad = 2 * rcfg.pad_px + 1
    roi = cv2.dilate(roi, np.ones((pad, pad), np.uint8))

    # ---- Evidence image for the dashboard: variance mask + DWT borders ------
    near = cv2.dilate(lti.low_var_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)) > 0
    eh = np.abs(Dh); ev = np.abs(Dv)
    th_h = rcfg.edge_z * np.median(eh) / 0.6745 * 2.0
    th_v = rcfg.edge_z * np.median(ev) / 0.6745 * 2.0
    evidence = np.zeros((h, w, 3), np.uint8)
    evidence[lti.low_var_mask > 0] = (200, 200, 200)                # grey  = LTI smooth-surface mask
    evidence[(eh > th_h) & near] = (0, 220, 255)                    # cyan  = DWT horizontal borders
    evidence[(ev > th_v) & near] = (255, 140, 0)                    # orange= DWT vertical borders
    return SimpleNamespace(roi=roi, core=core, rects=rects, core_rects=core_rects,
                           evidence=evidence, cell_rects=cell_rects)


# ==============================================================================
# BLOCK 6: THERMAL DELTA MAP  +  HARD WAVELET ENHANCEMENT   (inside the ROI only)
# ------------------------------------------------------------------------------
# NOISE REMOVAL
#   A 3x3 median filter (inside the panel only) removes salt-and-pepper pixels
#   before anything is measured.
#
# THERMAL DELTA
#   delta = pixel - local panel background.  The background is a median filter of
#   the image in which everything OUTSIDE the verified panel interior has been
#   replaced by the nearest panel pixel - so gravel and frame lines can never bias
#   the background near a panel border.  The median window is a few cell widths,
#   so faults smaller than the window are "voted out" of the background.  The warm band
#   that every module shows along its top / bottom edge repeats along the whole row, so
#   it is measured (median over x of each pixel row) and removed.
#
# NOISE LEVEL  (image quality)
#   sigma = 1.4826 * MAD(delta) measured on the smooth panel surface only (no cell
#   gaps, no faults).  Noisy image -> large sigma -> higher thresholds.
#
# HARD WAVELET ENHANCEMENT   (Donoho-Johnstone hard thresholding)
#   The positive heat map is decomposed with a 2-level DWT.  Detail coefficients
#   smaller than tau = scale * sigma * sqrt(2 ln N) are set to ZERO (noise); the
#   ones above tau are kept UNCHANGED (hard: real steps are not shrunk).  Two maps
#   are rebuilt from the surviving coefficients:
#     enhanced = approximation + kept details  -> de-noised heat map (thermal anomaly)
#     abrupt   = kept details ONLY             -> where brightness rises ABRUPTLY
#                (compact bright spots and sharp bright edges, even when their
#                 heat is too low to pass the plain thermal threshold: the wavelet
#                 gathers the energy of the whole spot, so its gain grows with size)
#   Both maps are marked as anomalies in Block 7.
# ==============================================================================

def thermal_delta_map(gray, core, valid, geom, lti, tcfg, rects=()):
    h, w = gray.shape
    inside = (core > 0) & (valid > 0)
    if not inside.any():
        return None

    g = cv2.medianBlur(gray, 3) if tcfg.denoise else gray          # noise removal

    # Fill outside-panel pixels with the nearest panel pixel, then median-filter -> panel-only background
    idx = ndi.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    filled = g[idx[0], idx[1]]
    k_bg = odd(np.clip(tcfg.bg_window_cells * geom.cell_w, 15, 0.5 * min(h, w)), lo=15)
    pd = k_bg // 2                                                  # mirror the borders (replicating a dark
    fp = cv2.copyMakeBorder(filled, pd, pd, pd, pd, cv2.BORDER_REFLECT_101)   # edge column would fake a cold background)
    bg = cv2.medianBlur(fp, k_bg)[pd:pd + h, pd:pd + w].astype(np.float32)
    delta = g.astype(np.float32) - bg

    # Remove the STATIC row structure: every module has the same warm band along its top / bottom edge,
    # so it repeats along the whole panel row.  The median over x of each pixel row captures it and is
    # subtracted (a fault would have to cover more than half of the row width to be absorbed).
    for (rx, ry, rw, rh) in rects:
        band = delta[ry:ry + rh, rx:rx + rw]
        band -= np.median(band, axis=1, keepdims=True)

    # Detection area = verified panel interior minus a thin rim
    margin = max(2, int(round(tcfg.margin_cells * geom.cell_w)))
    inner = cv2.erode(inside.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0          # image border = outside
    if not inner.any():
        inner = inside

    # Noise sigma from the smooth panel surface only (robust MAD)
    src = inner & lti.fine_smooth
    if src.sum() < 500:
        src = inner
    d_s = delta[src]
    sigma = float(1.4826 * np.median(np.abs(d_s - np.median(d_s))))
    sigma = max(sigma, 0.5)                                         # never below quantisation noise

    vb = valid > 0
    p_hi, p_lo = np.percentile(gray[vb], [99.5, 0.5])
    dyn = max(float(p_hi - p_lo), 16.0)                             # brightness range of the image

    # ---- derived thresholds (grey levels), all functions of THIS image ------
    sens = max(tcfg.sensitivity, 1e-3)
    T_hi = max(tcfg.z_hi * sigma, tcfg.min_contrast_frac * dyn) / sens
    T_lo = max(tcfg.z_lo * sigma, tcfg.hyst_ratio * max(tcfg.z_hi * sigma, tcfg.min_contrast_frac * dyn)) / sens
    T_ab = tcfg.abrupt_z * sigma / sens

    # ---- hard wavelet enhancement -----------------------------------------
    delta_pos = np.maximum(delta, 0.0) * inner
    m = 2 ** tcfg.wav_level
    ph, pw = (-h) % m, (-w) % m
    x = np.pad(delta_pos, ((0, ph), (0, pw)), mode='constant')
    level = max(1, min(tcfg.wav_level, pywt.dwtn_max_level(x.shape, tcfg.wavelet)))
    coeffs = pywt.wavedec2(x, tcfg.wavelet, mode='periodization', level=level)
    tau = tcfg.wav_tau_scale * sigma * float(np.sqrt(2.0 * np.log(max(int(inner.sum()), 2)))) / sens
    kept = [coeffs[0]]                                              # approximation untouched
    only = [np.zeros_like(coeffs[0])]                               # details-only copy (approximation = 0)
    for det in coeffs[1:]:
        hard = tuple(pywt.threshold(d, tau, mode='hard') * tcfg.wav_gain for d in det)
        kept.append(hard)
        only.append(hard)
    enhanced = pywt.waverec2(kept, tcfg.wavelet, mode='periodization')[:h, :w]
    abrupt = pywt.waverec2(only, tcfg.wavelet, mode='periodization')[:h, :w]
    enhanced = np.maximum(enhanced, 0.0).astype(np.float32) * inner
    abrupt = np.maximum(abrupt, 0.0).astype(np.float32) * inner     # brightness INCREASES only

    return SimpleNamespace(delta=delta, enhanced=enhanced, abrupt=abrupt, inner=inner, inside=inside,
                           sigma=sigma, dyn=dyn, T_hi=T_hi, T_lo=T_lo, T_ab=T_ab, tau=tau,
                           k_bg=k_bg, margin=margin)


# ==============================================================================
# BLOCK 7: FAULT DETECTION + CLASSIFICATION
# ------------------------------------------------------------------------------
# Regions come from two kinds of evidence, both marked as anomalies:
#   thermal anomaly    : seeds = enhanced heat >= T_hi, grown to the pixels that are
#                        >= a fraction of the strongest pixel NEARBY (and >= T_lo)
#   abrupt increase    : brightness rises abruptly (hard-wavelet detail map >= T_ab);
#                        catches compact bright spots that are too faint for T_hi
# A region is tightened to its own peak so blur / halo never inflates its shape.
#
# SHAPE  (rotation-invariant, from the minimum-area rectangle of the region)
#   aspect = long side / short side
#   long   = aspect >= aspect_long  AND  long side >= min length
#            (min length is measured in CELL widths / heights of this panel)
#
# HEAT   (how "light" the region is in the thermal map)
#   heat = median(delta in region) / T_ref
#   T_ref = max( strongest anomaly of THIS image , ref_floor_frac * dynamic range )
#   so "slightly lighter" always means "clearly weaker than the strongest thermal
#   feature / the brightness range of the image", never a fixed grey level.
#
# RULES
#   not long                         -> Hotspot
#   long  and heat >= ps_max_heat    -> Line Fault
#   long  and heat <  ps_max_heat    -> Partial Shading
#
# IIR CONTINUITY CHECK (DSP add-on)
#   The intensity profile along the fault's LONG axis is low-pass filtered with a
#   zero-phase Butterworth IIR filter (designed through the bilinear transform).
#   A real stripe stays above T_lo along (min_cover) of its length.  v1 filtered
#   along image rows only, so vertical faults could never be confirmed.
# ==============================================================================

def iir_verify_continuity(delta, box, horizontal, T_lo, icfg):
    x, y, bw, bh = box
    patch = delta[y:y + bh, x:x + bw]
    profile = patch.mean(axis=0) if horizontal else patch.mean(axis=1)    # profile along the long axis
    b, a = spsig.butter(N=icfg.order, Wn=icfg.cutoff, btype='low', analog=False)   # bilinear transform
    if len(profile) <= 3 * max(len(a), len(b)):                          # too short to filter
        return True
    smooth = spsig.filtfilt(b, a, profile.astype(np.float64))            # zero-phase IIR
    return float(np.mean(smooth >= T_lo)) >= icfg.min_cover


def compute_fsi(f, T_ref, geom, fcfg):
    """Fault Severity Index 0..100 = 100 * w_type * (w_area*area_ratio + w_int*intensity_ratio)."""
    area_ratio = min(f['area'] / (fcfg.full_area_cells * geom.cell_w * geom.cell_h), 1.0)
    intensity_ratio = min(f['peak_heat'], 1.0)
    w_type = fcfg.type_weight.get(f['type'], 0.5)
    return round(100.0 * w_type * (fcfg.w_area * area_ratio + fcfg.w_intensity * intensity_ratio), 1)


def detect_and_classify(gray, th, geom, cfg):
    h, w = gray.shape
    tcfg, ccfg = cfg.thermal, cfg.cls

    # Two kinds of evidence, both marked as anomalies:
    #   thermal anomaly  : de-noised heat above the seed / growth thresholds
    #   abrupt increase  : brightness rises abruptly (hard-wavelet detail map)
    ab = th.abrupt >= th.T_ab
    hi = ((th.enhanced >= th.T_hi) | ab) & th.inner

    # Growth is judged against the strongest pixel NEARBY (not a global number): the halo of a very
    # hot fault is dropped, while a weaker fault a few cells away keeps its own, lower level.
    r = max(3, int(round(tcfg.peak_radius_cells * geom.cell_w)))
    P = cv2.dilate(th.enhanced, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    lo = ((th.enhanced >= np.maximum(th.T_lo, tcfg.region_peak_frac * P)) | ab) & th.inner

    # bridge the cell-gap holes of a stripe (along the row), then remove speckle
    g_h = max(1, int(round(tcfg.frag_gap_cells * geom.cell_w)))
    g_v = max(1, int(round(tcfg.frag_gap_v_cells * geom.cell_w)))
    lo_u8 = lo.astype(np.uint8) * 255
    lo_u8 = cv2.morphologyEx(lo_u8, cv2.MORPH_CLOSE, np.ones((1, g_h), np.uint8))
    lo_u8 = cv2.morphologyEx(lo_u8, cv2.MORPH_CLOSE, np.ones((g_v, 1), np.uint8))
    lo_u8 = cv2.morphologyEx(lo_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    lo_u8[~th.inner] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(lo_u8, connectivity=8)
    seeded = set(np.unique(labels[hi & (lo_u8 > 0)]).tolist()) - {0}
    min_area = max(6.0, tcfg.min_region_cells * geom.cell_w * geom.cell_h)

    max_panel_bright = float(gray[th.inside].max())
    regions = []
    for lab in sorted(seeded):
        bx, by, bw, bh, area = stats[lab]
        if area < min_area:
            continue
        sl = (slice(by, by + bh), slice(bx, bx + bw))
        m = labels[sl] == lab
        Ec, ab_c = th.enhanced[sl], ab[sl]

        # Tighten the region: a thermal anomaly keeps pixels >= region_peak_frac * its own peak;
        # an abrupt-increase-only region keeps its abrupt pixels.  Keeps blur / halo out of the shape.
        if (Ec[m] >= th.T_hi).any():                                 # thermal anomaly
            peak = float(np.percentile(Ec[m], 99))
            keep = m & (Ec >= max(th.T_lo, tcfg.region_peak_frac * peak))
        else:                                                        # abrupt-increase only
            keep = m & ab_c
        if keep.sum() < min_area:
            continue
        ys, xs = np.nonzero(keep)
        bx, by = bx + int(xs.min()), by + int(ys.min())
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        sl = (slice(by, by + bh), slice(bx, bx + bw))
        m = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        area = int(m.sum())

        pts = cv2.findNonZero(m.astype(np.uint8))
        pts = pts + np.array([[[bx, by]]], dtype=pts.dtype)
        rect = cv2.minAreaRect(pts)                                  # rotation-invariant extent
        (rw, rh) = rect[1]
        L, Wd = max(rw, rh) + 1.0, max(min(rw, rh), 0.0) + 1.0       # +1: pixel centres -> extent
        box = cv2.boxPoints(rect)                                    # 4 corners (aligned frame)
        e0, e1 = box[1] - box[0], box[2] - box[1]
        long_vec = e0 if np.hypot(*e0) >= np.hypot(*e1) else e1
        horizontal = abs(long_vec[0]) >= abs(long_vec[1])
        vals = th.delta[sl][m]
        has_th = bool((th.enhanced[sl][m] >= th.T_hi).any())
        has_ab = bool(ab[sl][m].any())
        regions.append({
            'source': 'thermal+abrupt' if (has_th and has_ab) else ('thermal' if has_th else 'abrupt'),
            'box': (int(bx), int(by), int(bw), int(bh)), 'rect_pts': box, 'area': int(area),
            'L': float(L), 'W': float(Wd), 'aspect': float(L / Wd), 'horizontal': bool(horizontal),
            'med': float(np.median(vals)), 'p95': float(np.percentile(vals, 95)),
            'brightness_pct': int(round(100.0 * float(gray[sl][m].max()) / (max_panel_bright + 1e-5))),
        })

    # heat reference of THIS image
    strongest = max([r['p95'] for r in regions], default=0.0)
    T_ref = max(strongest, ccfg.ref_floor_frac * th.dyn, 1e-3)

    faults = []
    for r in regions:
        min_len = ccfg.min_len_h_cells * geom.cell_w if r['horizontal'] else ccfg.min_len_v_cells * geom.cell_h
        is_long = (r['aspect'] >= ccfg.aspect_long) and (r['L'] >= min_len)
        r['heat'] = min(r['med'] / T_ref, 1.0)
        r['peak_heat'] = min(r['p95'] / T_ref, 1.0)

        if not is_long:
            f_type = 'Hotspot'                       # squarish or small
        elif r['heat'] >= ccfg.ps_max_heat:
            f_type = 'Line Fault'                    # long and hot
        else:
            f_type = 'Partial Shading'               # long and only slightly lighter

        iir_ok = None
        if is_long and cfg.iir.enabled:
            iir_ok = iir_verify_continuity(th.delta, r['box'], r['horizontal'], th.T_lo, cfg.iir)
            if (not iir_ok) and f_type == 'Line Fault':
                f_type = 'Partial Shading'

        r.update(type=f_type, color=FAULT_COLOR[f_type], iir_confirmed=iir_ok, is_long=is_long)
        r['fsi'] = compute_fsi(r, T_ref, geom, cfg.fsi)
        faults.append(r)

    faults.sort(key=lambda f: f['fsi'], reverse=True)               # worst first
    return SimpleNamespace(faults=faults, T_ref=T_ref, hi=hi, lo=lo_u8 > 0)


# ==============================================================================
# BLOCK 8: MAIN ANALYSIS  (everything after the tilt step - re-run by the tuner)
# ==============================================================================

def build_roi_auto(ga, valid, cfg):
    """
    Panel ROI with a safety net.  The ROI is built with the user's cutoff first; only
    if that is a gross failure (no panel found, or "panel" covers almost the whole
    frame) are stricter / looser cutoffs tried automatically.
    Returns (lti, geom, roi_res, multiplier_used).
    """
    tries = (1.0, 0.8, 0.65, 1.25, 1.5) if cfg.roi.auto_cutoff else (1.0,)
    first = None
    for mult in tries:
        rc = copy.copy(cfg.roi)
        rc.var_thresh_scale = cfg.roi.var_thresh_scale * mult
        lti = lti_variance_mask(ga, valid, rc)                       # panel AREA (coarse) + inside check (fine)
        geom = estimate_cell_geometry(lti.low_var_mask, lti.k, ga)   # cell size -> all kernel sizes
        res = build_panel_roi(ga, lti, geom, rc)                     # panel EDGES + ROI
        cov = float((res.core > 0).sum()) / max(int((valid > 0).sum()), 1)
        res.coverage = cov
        if first is None:
            first = (lti, geom, res, mult)
        if res.rects and 0.02 <= cov <= 0.92:
            return lti, geom, res, mult
    return first


def analyse(gray_raw, img_bgr, tilt, cfg):
    h, w = gray_raw.shape
    ga, valid = tilt.gray_aligned, tilt.valid

    lti, geom, roi_res, roi_mult = build_roi_auto(ga, valid, cfg)
    th = thermal_delta_map(ga, roi_res.core, valid, geom, lti, cfg.thermal, roi_res.core_rects)

    R = SimpleNamespace(cfg=cfg, gray_raw=gray_raw, tilt=tilt, lti=lti, geom=geom, roi_res=roi_res,
                        roi_mult=roi_mult, th=th, det=None, faults=[],
                        counts={'Hotspot': 0, 'Line Fault': 0, 'Partial Shading': 0}, annotated=None)
    if th is not None:
        R.det = detect_and_classify(ga, th, geom, cfg)
        R.faults = R.det.faults
    for f in R.faults:
        R.counts[f['type']] += 1

    # ---- map results back onto the ORIGINAL (un-rotated) image --------------
    inv = cv2.invertAffineTransform(tilt.rot_mat)
    roi_orig = cv2.warpAffine(roi_res.roi, inv, (w, h), flags=cv2.INTER_NEAREST)
    annotated = img_bgr.copy()
    cnts, _ = cv2.findContours(roi_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, cnts, -1, (0, 255, 0), 2)
    fs = 0.38 * max(1.0, max(h, w) / 512.0)
    for f in R.faults:
        pts = np.hstack([f['rect_pts'].astype(np.float32), np.ones((4, 1), np.float32)])
        pts_o = (pts @ inv.T).astype(np.int32)
        cv2.polylines(annotated, [pts_o], True, f['color'], 2)
        tl = pts_o[np.argmin(pts_o[:, 1])]
        label = f"{FAULT_ABBR[f['type']]} {f['brightness_pct']}%  FSI:{f['fsi']}"
        pos = (int(tl[0]) - 2, max(int(tl[1]) - 5, 12))
        cv2.putText(annotated, label, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, label, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, f['color'], 1, cv2.LINE_AA)
    R.annotated = annotated
    return R


def print_summary(R):
    t = R.tilt
    print(f"\n{'=' * 70}")
    print(f"  Tilt corrected     : {t.angle:.2f} deg   (FFT check {t.fft_angle:.2f} deg -> "
          f"{'agree' if t.agree else 'DISAGREE - check image quality'})")
    print(f"  Cell footprint     : {R.geom.cell_w:.0f} x {R.geom.cell_h:.0f} px  ({R.geom.n_blobs} blobs)"
          f"   variance thr (Otsu): {R.lti.T:.0f}/255   variance window: {R.lti.k} px")
    print(f"  Panel ROI          : {len(R.roi_res.rects)} row(s), {100 * R.roi_res.coverage:.1f}% of the frame"
          + (f"   (cutoff auto-adjusted x{R.roi_mult:g})" if R.roi_mult != 1.0 else ""))
    if R.th is None:
        print("  No panel ROI found - nothing to analyse.")
        print(f"{'=' * 70}\n")
        return
    th = R.th
    print(f"  Noise sigma        : {th.sigma:.2f} grey levels     dynamic range: {th.dyn:.0f}")
    print(f"  Seed / grow thr    : {th.T_hi:.1f} / {th.T_lo:.1f} grey levels     wavelet tau: {th.tau:.1f}"
          f"     abrupt-increase thr: {th.T_ab:.1f}")
    print(f"  Heat reference     : {R.det.T_ref:.1f} grey levels   (Partial Shading if heat < {R.cfg.cls.ps_max_heat:.2f})")
    c = R.counts
    print(f"  Total faults       : {sum(c.values())}   (HS:{c['Hotspot']}  LF:{c['Line Fault']}  PS:{c['Partial Shading']})")
    print(f"{'-' * 70}")
    for i, f in enumerate(R.faults, 1):
        iir = '' if f['iir_confirmed'] is None else f"  IIR={'ok' if f['iir_confirmed'] else 'no'}"
        print(f"  {i:2d}. {f['type']:<16} FSI={f['fsi']:5.1f}  heat={f['heat']:.2f}  aspect={f['aspect']:4.1f}  "
              f"{'H' if f['horizontal'] else 'V'}  size={f['L']:.0f}x{f['W']:.0f}  [{f['source']}]{iir}")
    print(f"{'=' * 70}\n")


# ==============================================================================
# BLOCK 9: 8-SUBPLOT DASHBOARD  +  LIVE THRESHOLD TUNER
# ==============================================================================

def _rgb(bgr):
    return (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)


def render(fig, axes, R):
    for ax in axes.flat:
        ax.clear()
        ax.set_facecolor('#0d0d1a')
        ax.axis('off')
    tl = dict(color='white', fontweight='bold', fontsize=8)
    t, ga = R.tilt, R.tilt.gray_aligned

    axes[0, 0].imshow(R.gray_raw, cmap='gray')
    axes[0, 0].set_title('1. Raw Thermal Input', color='white', fontweight='bold')

    axes[0, 1].imshow(t.fft_mag, cmap='inferno')
    axes[0, 1].set_title(f"2. FFT Spectrum (tilt check)\nFFT {t.fft_angle:.1f}\u00b0   Projection {t.angle:.1f}\u00b0   "
                         f"{'agree ✓' if t.agree else 'DISAGREE ✗'}", **tl)

    axes[0, 2].imshow(R.lti.std_map, cmap='viridis')
    axes[0, 2].set_title(f"3. LTI Variance Map", **tl)

    axes[0, 3].imshow(R.roi_res.evidence)
    axes[0, 3].set_title("4. Panel evidence\ngrey = LTI area   cyan/orange = DWT horizontal/vertical borders", **tl)

    ov = cv2.cvtColor(ga, cv2.COLOR_GRAY2RGB)
    cnts, _ = cv2.findContours(R.roi_res.roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
    for (rx, ry, rw, rh) in R.roi_res.core_rects:                    # verified smooth interior (used for detection)
        cv2.rectangle(ov, (rx, ry), (rx + rw - 1, ry + rh - 1), (0, 200, 255), 1)
    axes[1, 0].imshow(ov)
    axes[1, 0].set_title("5. Panel ROI\n"
                         "[Morph closing \u2192 DWT border \u2192 1 px pad]", **tl)

    if R.th is not None:
        th = R.th
        show = np.where(th.inside, th.delta, 0.0)
        vmax = max(float(np.percentile(show[th.inside], 99.8)), 3.0 * th.T_hi)
        axes[1, 1].imshow(np.maximum(show, 0), cmap='magma', vmin=0, vmax=vmax)
        axes[1, 1].set_title(f"6. Thermal Delta map inside ROI )\nsigma={th.sigma:.1f}   seed {th.T_hi:.0f} / grow {th.T_lo:.0f}", **tl)

        comp = np.stack([np.clip(th.enhanced / (2.0 * th.T_hi), 0, 1),         # red   = thermal excess (de-noised)
                         np.clip(th.abrupt / (4.0 * th.T_ab), 0, 1),           # green = abrupt brightness increase
                         np.zeros_like(th.enhanced)], axis=-1)
        axes[1, 2].imshow(comp)
        for f in R.faults:
            axes[1, 2].add_patch(Polygon(f['rect_pts'], closed=True, fill=False,
                                         edgecolor=_rgb(f['color']), linewidth=1.3))
        axes[1, 2].set_title(f"7. Wavelet map:  boxes = accepted faults\n"
                             "red = thermal excess ;  green = abrupt brightness increase", **tl)
    else:
        axes[1, 1].set_title("6. Thermal Delta - no ROI found", **tl)
        axes[1, 2].set_title("7. Wavelet map - no ROI found", **tl)

    axes[1, 3].imshow(cv2.cvtColor(R.annotated, cv2.COLOR_BGR2RGB))
    c = R.counts
    axes[1, 3].set_title(f"8. Final Output\n[HS:{c['Hotspot']}  LF:{c['Line Fault']}  PS:{c['Partial Shading']}]"
                         f"  labels: brightness% and FSI", **tl)
    lines = [f"Total: {sum(c.values())}", f"HS:{c['Hotspot']}  LF:{c['Line Fault']}  PS:{c['Partial Shading']}",
             "-" * 30, "Ranked by FSI (worst first):"]
    for i, f in enumerate(R.faults[:7], 1):
        tag = '' if f['iir_confirmed'] is None else f"  IIR={'ok' if f['iir_confirmed'] else 'no'}"
        lines.append(f" {i}. {f['type'][:12]:<12} FSI={f['fsi']:5.1f}{tag}")
    if not R.faults:
        lines.append("  No faults detected.")
    axes[1, 3].text(0.02, 0.02, "\n".join(lines), transform=axes[1, 3].transAxes, fontsize=7.5,
                    color='white', va='bottom', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.80))
    fig.suptitle('Thermal PV Fault Detection  \u00b7  DSP: coarse-fine tilt + 2D FFT | LTI variance | '
                 '2D DWT edges + hard-threshold enhancement | IIR Butterworth | FSI',
                 color='white', fontsize=9, fontweight='bold')


# (label, config group, attribute, min, max)   -  only the two knobs that matter
TUNER_SLIDERS = [
    ("Anomaly sensitivity", 'thermal', 'sensitivity', 0.4, 3.0),         # right = detect weaker anomalies
    ("Panel ROI sensitivity", 'roi', 'var_thresh_scale', 0.6, 1.4),   # right = looser (bigger ROI), left = stricter
]


def launch_dashboard(gray_raw, img_bgr, base_cfg, tune=True, save_path='pv_fault_detection_output.jpg'):
    tilt = detect_and_align_tilt(gray_raw, base_cfg.tilt)            # the tilt never changes with the sliders
    R = analyse(gray_raw, img_bgr, tilt, base_cfg)
    print_summary(R)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11 if tune else 10))
    fig.patch.set_facecolor('#0d0d1a')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.14 if tune else 0.02, wspace=0.03, hspace=0.18)
    render(fig, axes, R)
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())

    if tune:
        widgets = []
        sliders = []

        def recompute():
            cfg = copy.deepcopy(base_cfg)
            for s, (_, grp, attr, _, _) in zip(sliders, TUNER_SLIDERS):
                setattr(getattr(cfg, grp), attr, float(s.val))
            Rn = analyse(gray_raw, img_bgr, tilt, cfg)
            render(fig, axes, Rn)
            print_summary(Rn)
            fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=250)                   # debounce while dragging
        timer.single_shot = True
        timer.add_callback(recompute)

        def on_change(_):
            timer.stop()
            timer.start()

        for i, (label, grp, attr, lo, hi) in enumerate(TUNER_SLIDERS):
            ax_s = fig.add_axes([0.14 + i * 0.38, 0.06, 0.20, 0.025], facecolor='#222244')
            s = Slider(ax_s, label, lo, hi, valinit=float(getattr(getattr(base_cfg, grp), attr)))
            s.label.set_color('white'); s.label.set_fontsize(10)
            s.valtext.set_color('white'); s.valtext.set_fontsize(10)
            s.on_changed(on_change)
            sliders.append(s)
            widgets.append(s)

        ax_r = fig.add_axes([0.90, 0.055, 0.045, 0.035])
        btn_reset = Button(ax_r, 'Reset', color='#222244', hovercolor='#33336a')
        btn_reset.label.set_color('white')
        btn_reset.on_clicked(lambda _e: [s.reset() for s in sliders])
        ax_v = fig.add_axes([0.95, 0.055, 0.04, 0.035])
        btn_save = Button(ax_v, 'Save', color='#222244', hovercolor='#33336a')
        btn_save.label.set_color('white')
        btn_save.on_clicked(lambda _e: fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor()))
        widgets += [btn_reset, btn_save]
        fig._pv_widgets = widgets                                    # keep references alive
        fig._pv_timer = timer
        fig.text(0.14, 0.105, "Anomaly sensitivity: move right to catch weaker anomalies, left to keep only strong ones.          "
                              "Panel ROI detection sensitivity.",
                 color='#aaaacc', fontsize=9)
        fig._pv_recompute = recompute

    plt.show()
    return R


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PV thermal fault detection")
    ap.add_argument('image', nargs='?', help="thermal image path (omit to use the file picker)")
    ap.add_argument('--no-tuner', action='store_true', help="hide the live threshold sliders")
    args = ap.parse_args()

    path = args.image
    if not path:
        if tk is None:
            sys.exit("Tkinter is not available - pass the image path on the command line.")
        print("Select a thermal image to analyse...")
        path = select_image()
    if not path:
        sys.exit("No file selected - exiting.")

    img = cv2.imread(path)
    if img is None:
        sys.exit("Could not open thermal image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    launch_dashboard(gray, img, Config(), tune=not args.no_tuner)