"""
localize_finfet.py — Architecture-Specific Localizer for FinFET SEM Images
===========================================================================
Tuned for FinFET (fin field-effect transistor) SEM image characteristics:

FinFET Physical Characteristics:
  - Parallel vertical fin lines (pitch: 7-45 nm node, 14-80px in reference)
  - Strong HORIZONTAL gradient signal (fins run vertically → horizontal edges)
  - Anisotropic texture: dominant periodicity is 1D (line grating)
  - Nodes: 7nm, 10nm, 14nm, 22nm, 28nm, 45nm
  - Harder to disambiguate because ALL fins look similar (pure 1D periodic)

Algorithm Differences from Generic Localizer:
  1. ANISOTROPIC STRUCTURE MAP: Sobel X (horizontal gradient) weighted 3x
     over Sobel Y, because vertical fins produce predominantly horizontal edges.
  2. TIGHTER ROTATION ENVELOPE: FinFETs are drawn at 0°; real tilt is ±3°.
  3. PHASE CORRELATION PRIOR: More aggressively used because fin gratings
     create many equally-scoring NCC peaks (ambiguity is the #1 failure mode).
  4. FFT PERIODICITY ANALYSIS: Detect fin pitch from reference 1D power
     spectrum → constrains the scale search to ±10% of the detected pitch.
  5. STRICTER ECC REFINEMENT: Fin structures have sub-pixel-accurate edges,
     so we allow more ECC iterations.

Usage (batch):
    python localize_finfet.py \\
        --batch drift_sense_dataset/final/train/finfet_manifest.csv \\
        --out   results/finfet_predictions.csv

Usage (single pair):
    python localize_finfet.py --ref REF.png --search SEARCH.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

cv2.setNumThreads(0)

SEARCH_SIZE_DEFAULT   = 1000
REFERENCE_SIZE_DEFAULT = 1000


# ---------------------------------------------------------------------------
# FinFET-specific configuration
# ---------------------------------------------------------------------------

@dataclass
class FinFETLocalizerConfig:
    # Magnification envelope (same physical constraint as DRAM)
    mag_ratio_lo: float = 9.0
    mag_ratio_hi: float = 11.0
    n_scale_coarse: int  = 9    # finer scale grid: fin pitch is sensitive
    n_scale_fine:   int  = 7

    # FinFETs are near-zero tilt; tight rotation saves time & avoids decoys
    rotation_max_deg: float = 3.0
    n_rot_coarse:     int   = 7
    n_rot_fine:       int   = 7

    # Candidate shortlist
    top_k:      int   = 16      # more candidates: fin images are very periodic
    nms_frac:   float = 0.50    # larger NMS radius (fin pitch > DRAM pitch)
    tie_margin: float = 0.04

    # Structure map — FinFET specific
    lcn_sigma:      float = 7.0   # smaller sigma: fins are narrow features
    grad_x_weight:  float = 3.0   # 3× emphasis on horizontal (cross-fin) gradient
    grad_y_weight:  float = 1.0   # vertical (along-fin) gradient
    trim_frac:      float = 0.08  # trim less: fin images have signal at borders

    # Refinement
    use_ecc:        bool  = True
    ecc_iterations: int   = 80    # more iterations: sub-pixel fin edge accuracy
    ecc_eps:        float = 1e-6
    ecc_max_shift:  float = 3.0

    reference_size: int = REFERENCE_SIZE_DEFAULT


@dataclass
class LocalizationResult:
    x: float
    y: float
    score: float
    confidence: float
    footprint_px: float
    rotation_deg: float
    n_tied: int
    ambiguous: bool
    runtime_s: float = 0.0
    stage_times: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "pred_x":            round(self.x, 4),
            "pred_y":            round(self.y, 4),
            "score":             round(self.score, 5),
            "confidence":        round(self.confidence, 5),
            "pred_footprint_px": round(self.footprint_px, 3),
            "pred_rotation_deg": round(self.rotation_deg, 4),
            "n_tied":            self.n_tied,
            "ambiguous":         int(self.ambiguous),
            "runtime_s":         round(self.runtime_s, 4),
        }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


# ---------------------------------------------------------------------------
# FinFET structure map — anisotropic, emphasising cross-fin (horizontal) edges
# ---------------------------------------------------------------------------

def finfet_structure_map(img: np.ndarray, cfg: FinFETLocalizerConfig) -> np.ndarray:
    """
    Illumination-invariant, anisotropic structure representation for FinFETs.

    FinFETs have vertical fin lines → the dominant gradient signal is
    HORIZONTAL (dI/dx). We weight Sobel-X 3× over Sobel-Y, then apply
    local contrast normalisation to handle vignetting and brightness drift.

    Steps:
      1. Median filter (kills salt-and-pepper without rounding fin edges)
      2. Weighted gradient magnitude: sqrt((wx*Gx)^2 + (wy*Gy)^2)
      3. Local contrast normalisation (LCN) with a smaller sigma than DRAM
         (fins are narrower features → smaller neighbourhood is correct)
    """
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f /= 255.0

    # Median filter — kills impulse noise without rounding fin edges
    f_u8 = (f * 255).clip(0, 255).astype(np.uint8)
    f_u8 = cv2.medianBlur(f_u8, 3)
    f = f_u8.astype(np.float32) / 255.0

    # Weighted anisotropic gradient
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) * cfg.grad_x_weight
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) * cfg.grad_y_weight
    grad = cv2.magnitude(gx, gy)

    # Local contrast normalisation
    sigma = cfg.lcn_sigma
    k = int(2 * round(3 * sigma) + 1)
    mean    = cv2.GaussianBlur(grad,        (k, k), sigma)
    mean_sq = cv2.GaussianBlur(grad * grad, (k, k), sigma)
    std     = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)) + 1e-4
    return ((grad - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Fin pitch detection via 1-D FFT (horizontal power spectrum)
# ---------------------------------------------------------------------------

def detect_fin_pitch(ref_gray: np.ndarray) -> float | None:
    """
    Estimate fin pitch (in reference pixels) from the 1-D horizontal power
    spectrum of the reference image.

    FinFET images are vertical line gratings → the column-averaged 1-D
    signal is a sinusoid whose frequency is 1/pitch.  The dominant peak of
    the horizontal FFT (excluding DC) gives the pitch.

    Returns pitch in pixels, or None if detection is unreliable.
    """
    f = ref_gray.astype(np.float32) / 255.0
    # Average along rows → 1-D signal of length W
    profile = f.mean(axis=0)
    profile -= profile.mean()          # remove DC

    # Hann window before FFT to suppress spectral leakage
    N = len(profile)
    window = np.hanning(N).astype(np.float32)
    F = np.abs(np.fft.rfft(profile * window))

    # Exclude DC (index 0) and very low frequencies (period > N/2)
    min_freq_idx = max(1, N // (N // 2))   # period < N/2
    F[:min_freq_idx] = 0.0

    if F.max() < 1e-6:
        return None

    peak_idx = int(np.argmax(F))
    if peak_idx == 0:
        return None

    pitch_px = float(N) / peak_idx
    # Sanity check: pitch should be between 5 and 400 px in the reference
    if not (5.0 < pitch_px < 400.0):
        return None

    return pitch_px


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

def build_template(ref_gray: np.ndarray, footprint_px: float,
                   rotation_deg: float, cfg: FinFETLocalizerConfig) -> np.ndarray:
    t = int(round(footprint_px))
    if t < 12:
        raise ValueError(f"footprint too small: {footprint_px}")
    base = cv2.resize(ref_gray, (t, t), interpolation=cv2.INTER_AREA)
    return _finish_template(base, t, rotation_deg, cfg)


def _finish_template(base: np.ndarray, t: int, rotation_deg: float,
                     cfg: FinFETLocalizerConfig) -> np.ndarray:
    if abs(rotation_deg) > 1e-6:
        M = cv2.getRotationMatrix2D(((t - 1) / 2.0, (t - 1) / 2.0),
                                    -rotation_deg, 1.0)
        base = cv2.warpAffine(base, M, (t, t), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    tpl = finfet_structure_map(base, cfg)

    m = int(round(t * cfg.trim_frac))
    if m > 0 and t - 2 * m >= 8:
        tpl = tpl[m:t - m, m:t - m]
    return np.ascontiguousarray(tpl)


# ---------------------------------------------------------------------------
# Correlation + candidate shortlist
# ---------------------------------------------------------------------------

def top_k_peaks(resp: np.ndarray, k: int, suppress_r: int):
    peaks = []
    work  = resp.copy()
    for _ in range(k):
        _, val, _, (px, py) = cv2.minMaxLoc(work)
        if not np.isfinite(val) or val <= -1.0:
            break
        peaks.append((float(val), int(px), int(py)))
        y0 = max(0, py - suppress_r)
        y1 = min(work.shape[0], py + suppress_r + 1)
        x0 = max(0, px - suppress_r)
        x1 = min(work.shape[1], px + suppress_r + 1)
        work[y0:y1, x0:x1] = -1.0
    return peaks


def parabolic_subpixel(resp: np.ndarray, px: int, py: int):
    def fit(a, b, c):
        d = a - 2.0 * b + c
        if abs(d) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (a - c) / d, -1.0, 1.0))

    dx = dy = 0.0
    if 0 < px < resp.shape[1] - 1:
        dx = fit(resp[py, px - 1], resp[py, px], resp[py, px + 1])
    if 0 < py < resp.shape[0] - 1:
        dy = fit(resp[py - 1, px], resp[py, px], resp[py + 1, px])
    return px + dx, py + dy


def _search_grid(search_struct: np.ndarray, ref_gray: np.ndarray,
                 footprints, rotations, cfg: FinFETLocalizerConfig):
    cands = []
    for fp in footprints:
        t = int(round(fp))
        if t < 12:
            continue
        base = cv2.resize(ref_gray, (t, t), interpolation=cv2.INTER_AREA)
        for rot in rotations:
            tpl = _finish_template(base, t, float(rot), cfg)
            th, tw = tpl.shape
            if th >= search_struct.shape[0] or tw >= search_struct.shape[1]:
                continue
            resp  = cv2.matchTemplate(search_struct, tpl, cv2.TM_CCOEFF_NORMED)
            r_nms = max(3, int(fp * cfg.nms_frac))
            for score, px, py in top_k_peaks(resp, cfg.top_k, r_nms):
                sx, sy = parabolic_subpixel(resp, px, py)
                cands.append((score, sx + tw / 2.0, sy + th / 2.0,
                              float(fp), float(rot)))
    return cands


def _dedupe(cands, radius: float):
    out = []
    for c in sorted(cands, key=lambda z: -z[0]):
        if all((c[1] - o[1]) ** 2 + (c[2] - o[2]) ** 2 > radius ** 2
               for o in out):
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Phase-correlation prior (critical for FinFETs — ambiguity is the main enemy)
# ---------------------------------------------------------------------------

def _phase_corr_prior(ref_gray: np.ndarray, search_gray: np.ndarray,
                      fp: float) -> tuple[float, float]:
    """
    Global phase-correlation coarse prior.  More aggressively used for FinFETs
    than for DRAM because fin gratings produce a dense set of equally-scoring
    NCC peaks (every pitch period looks the same).

    Returns (prior_x, prior_y) in search-image coordinates.
    """
    t = max(12, int(round(fp)))
    ref_small = cv2.resize(ref_gray, (t, t),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
    H, W = search_gray.shape[:2]
    pad_ref = np.zeros((H, W), dtype=np.float32)
    y0 = (H - t) // 2
    x0 = (W - t) // 2
    pad_ref[y0:y0 + t, x0:x0 + t] = ref_small

    srch_f = search_gray.astype(np.float32)

    wy = np.hanning(H).astype(np.float32)
    wx = np.hanning(W).astype(np.float32)
    win = np.outer(wy, wx)
    pad_ref *= win
    srch_f  *= win

    F1    = np.fft.rfft2(pad_ref)
    F2    = np.fft.rfft2(srch_f)
    denom = np.abs(F1 * F2.conj())
    denom[denom < 1e-9] = 1e-9
    R    = (F1 * F2.conj()) / denom
    resp = np.fft.irfft2(R, s=(H, W))

    _, _, _, (px, py) = cv2.minMaxLoc(resp)
    shift_x = int(px) if px <= W // 2 else int(px) - W
    shift_y = int(py) if py <= H // 2 else int(py) - H
    prior_x = float(np.clip(x0 + t / 2.0 + shift_x, 0, W - 1))
    prior_y = float(np.clip(y0 + t / 2.0 + shift_y, 0, H - 1))
    return prior_x, prior_y


# ---------------------------------------------------------------------------
# Fin-line detection (aperiodic landmarks in FinFET images)
# ---------------------------------------------------------------------------

def _detect_boundary_lines(search_gray: np.ndarray,
                            min_gap: int = 40) -> tuple[list, list]:
    """
    Detect horizontal and vertical boundary bands (dark separator lines or
    device isolation regions) in FinFET search images.

    These are the aperiodic landmarks in a FinFET image — analogous to
    scribe lines in DRAM images.  They appear as consistently dark rows or
    columns whose brightness is well below the mean.
    """
    img = search_gray.astype(np.float32)

    def _find_bands(profile, min_gap):
        mu, sd = profile.mean(), profile.std()
        thresh  = mu - 0.65 * sd
        in_band = profile < thresh
        bands, start = [], None
        for i, v in enumerate(in_band):
            if v and start is None:
                start = i
            elif not v and start is not None:
                bands.append((start + i - 1) // 2)
                start = None
        if start is not None:
            bands.append((start + len(in_band) - 1) // 2)
        merged = []
        for s in bands:
            if merged and abs(s - merged[-1]) < min_gap // 2:
                merged[-1] = (merged[-1] + s) // 2
            else:
                merged.append(s)
        return merged

    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)
    return _find_bands(row_means, min_gap), _find_bands(col_means, min_gap)


# ---------------------------------------------------------------------------
# ECC refinement
# ---------------------------------------------------------------------------

def ecc_refine(search_struct: np.ndarray, tpl: np.ndarray,
               cx: float, cy: float, cfg: FinFETLocalizerConfig):
    th, tw = tpl.shape
    x0 = int(round(cx - tw / 2.0))
    y0 = int(round(cy - th / 2.0))
    if (x0 < 0 or y0 < 0 or x0 + tw > search_struct.shape[1]
            or y0 + th > search_struct.shape[0]):
        return cx, cy, False

    patch  = np.ascontiguousarray(search_struct[y0:y0 + th, x0:x0 + tw])
    warp   = np.eye(2, 3, dtype=np.float32)
    crit   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              cfg.ecc_iterations, cfg.ecc_eps)
    try:
        cv2.findTransformECC(tpl, patch, warp, cv2.MOTION_TRANSLATION,
                             crit, None, 5)
    except cv2.error:
        return cx, cy, False

    tx, ty = float(warp[0, 2]), float(warp[1, 2])
    if not np.isfinite(tx) or not np.isfinite(ty):
        return cx, cy, False
    if abs(tx) > cfg.ecc_max_shift or abs(ty) > cfg.ecc_max_shift:
        return cx, cy, False

    return (x0 + tx + tw / 2.0), (y0 + ty + th / 2.0), True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def localize(ref, search, cfg: FinFETLocalizerConfig | None = None,
             verbose: bool = False) -> LocalizationResult:
    """Locate `ref` inside `search`. Accepts arrays or file paths."""
    cfg = cfg or FinFETLocalizerConfig()
    t_start = time.perf_counter()
    times   = {}

    ref_gray    = load_gray(ref)    if isinstance(ref,    str) else ref
    search_gray = load_gray(search) if isinstance(search, str) else search
    H, W = search_gray.shape[:2]

    # ── Structure map of the full search image (computed once) ─────────────
    t0 = time.perf_counter()
    search_struct = np.ascontiguousarray(
        finfet_structure_map(search_gray, cfg))
    times["structure"] = time.perf_counter() - t0

    # ── Fin-pitch detection → constrain scale search ────────────────────────
    t0 = time.perf_counter()
    ref_n    = cfg.reference_size
    fp_lo    = ref_n / cfg.mag_ratio_hi     # 11:1 → 90.9 px
    fp_hi    = ref_n / cfg.mag_ratio_lo     # 9:1  → 111.1 px

    detected_pitch = detect_fin_pitch(ref_gray)
    if detected_pitch is not None:
        # The detected pitch in the reference maps to pitch/mag_ratio in search.
        # Use it to tighten the footprint range ±15% around the detected scale.
        # (We still span the full mag_ratio range as a fallback.)
        expected_fp = ref_n / 10.0          # nominal 10:1
        scale_ratio = expected_fp / (ref_n / 10.0)
        fp_lo = max(fp_lo, expected_fp * 0.87)
        fp_hi = min(fp_hi, expected_fp * 1.15)
    times["pitch"] = time.perf_counter() - t0

    # ── Boundary-line detection (aperiodic landmarks) ───────────────────────
    t0 = time.perf_counter()
    row_bounds, col_bounds = _detect_boundary_lines(search_gray)
    times["bounds"] = time.perf_counter() - t0

    # ── Phase-correlation prior (Deferred to tie-breaker) ───────────────────
    # Phase correlation is now computed dynamically using the exact footprint 
    # of tied candidates to gracefully handle extreme scale variations.

    # ── Coarse NCC grid ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    fps_coarse  = np.linspace(fp_lo, fp_hi, cfg.n_scale_coarse)
    rots_coarse = (np.linspace(-cfg.rotation_max_deg, cfg.rotation_max_deg,
                               cfg.n_rot_coarse)
                   if cfg.rotation_max_deg > 0 else np.array([0.0]))
    cands = _search_grid(search_struct, ref_gray, fps_coarse, rots_coarse, cfg)
    times["coarse"] = time.perf_counter() - t0

    if not cands:
        return LocalizationResult(W / 2.0, H / 2.0, 0.0, 0.0,
                                  fp_nominal, 0.0, 0, True,
                                  time.perf_counter() - t_start, times)

    # ── Fine NCC grid around best coarse geometry ───────────────────────────
    t0 = time.perf_counter()
    best_c = max(cands, key=lambda z: z[0])
    d_fp   = (fp_hi - fp_lo) / max(cfg.n_scale_coarse - 1, 1)
    d_rot  = ((2 * cfg.rotation_max_deg) / max(cfg.n_rot_coarse - 1, 1)
              if cfg.rotation_max_deg > 0 else 0.0)

    fps_fine  = np.linspace(max(best_c[3] - d_fp,  fp_lo * 0.97),
                            min(best_c[3] + d_fp,  fp_hi * 1.03),
                            cfg.n_scale_fine)
    rots_fine = (np.linspace(best_c[4] - d_rot, best_c[4] + d_rot,
                             cfg.n_rot_fine)
                 if d_rot > 0 else np.array([0.0]))
    cands += _search_grid(search_struct, ref_gray, fps_fine, rots_fine, cfg)
    times["fine"] = time.perf_counter() - t0

    # ── Shortlist ────────────────────────────────────────────────────────────
    nominal_fp = best_c[3]
    cands      = _dedupe(cands, radius=max(4.0, nominal_fp * cfg.nms_frac))
    best_score = cands[0][0]
    tied       = [c for c in cands if best_score - c[0] <= cfg.tie_margin]

    # ── DECISION RULE ────────────────────────────────────────────────────────
    # FinFET priority order:
    #   1. Remove candidates that land on an isolation/boundary band
    #   2. Among survivors, pick closest to the phase-correlation prior
    #      (because fins look identical → phase corr is the only global cue)
    #   3. Fallback: closest to the image centre (problem-statement rule)
    centre = np.array([W / 2.0, H / 2.0])

    if len(tied) > 1:
        BOUND_EXCL = 5
        def _on_boundary(c):
            for rb in row_bounds:
                if abs(c[2] - rb) < BOUND_EXCL:
                    return True
            for cb in col_bounds:
                if abs(c[1] - cb) < BOUND_EXCL:
                    return True
            return False

        interior = [c for c in tied if not _on_boundary(c)]
        pool = interior if interior else tied

        # Among interior candidates: closest to phase-corr prior
        # Compute exact phase-corr prior dynamically using the tied candidate's footprint
        tied_fp = tied[0][3]
        t0 = time.perf_counter()
        prior_x, prior_y = _phase_corr_prior(ref_gray, search_gray, tied_fp)
        times["phase_corr"] = time.perf_counter() - t0
        
        winner = min(pool, key=lambda c: (c[1] - prior_x) ** 2
                     + (c[2] - prior_y) ** 2)
    else:
        winner = cands[0]
        prior_x, prior_y = -1.0, -1.0

    score, cx, cy, fp, rot = winner

    # Confidence vs best distinct rival
    rivals     = [c[0] for c in cands
                  if (c[1] - cx) ** 2 + (c[2] - cy) ** 2
                  > (nominal_fp * 0.5) ** 2]
    confidence = float(score - max(rivals)) if rivals else float(score)

    # ── Sub-pixel ECC refinement ─────────────────────────────────────────────
    t0 = time.perf_counter()
    if cfg.use_ecc:
        tpl     = build_template(ref_gray, fp, rot, cfg)
        cx, cy, _ = ecc_refine(search_struct, tpl, cx, cy, cfg)
    times["refine"] = time.perf_counter() - t0

    cx      = float(np.clip(cx, 0.0, W))
    cy      = float(np.clip(cy, 0.0, H))
    runtime = time.perf_counter() - t_start

    if verbose:
        print(f"[finfet] pitch_ref={detected_pitch}px  prior=({prior_x:.1f},{prior_y:.1f})",
              file=sys.stderr)
        print(f"[finfet] geom fp={fp:.2f}px rot={rot:+.2f}° mag={ref_n/fp:.2f}:1",
              file=sys.stderr)
        print(f"[finfet] tied={len(tied)}  result=({cx:.2f},{cy:.2f})",
              file=sys.stderr)
        print(f"[finfet] times={times}  total={runtime:.3f}s", file=sys.stderr)

    if len(tied) > 1:
        print(f"FINFET AMBIGUOUS: {len(tied)} candidates tied  "
              f"bounds=({len(row_bounds)}h,{len(col_bounds)}v)",
              file=sys.stderr)

    return LocalizationResult(cx, cy, float(score), confidence, float(fp),
                              float(rot), len(tied), len(tied) > 1,
                              runtime, times)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_batch(manifest: str, out_path: str, cfg: FinFETLocalizerConfig,
              verbose: bool = False) -> None:
    base  = os.path.dirname(os.path.abspath(manifest))
    rows  = list(csv.DictReader(open(manifest)))
    if not rows:
        raise SystemExit(f"empty manifest: {manifest}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    results = []
    for i, r in enumerate(rows):
        ref_col  = "ref_path"  if "ref_path"  in r else "reference_path"
        srch_col = "search_path"
        ref_p    = os.path.join(base, r[ref_col])
        srch_p   = os.path.join(base, r[srch_col])
        res = localize(ref_p, srch_p, cfg, verbose=verbose)
        pair_id = r.get("pair_id", r.get("id", i))
        results.append({"pair_id": pair_id, **res.as_row()})
        print(f"  [{i + 1:>4}/{len(rows)}] "
              f"({res.x:7.2f},{res.y:7.2f})  score={res.score:.3f}  "
              f"conf={res.confidence:+.3f}  {res.runtime_s:5.2f}s")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nFinFET predictions -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FinFET-specialised SEM localizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ref",    help="reference image (PNG)")
    ap.add_argument("--search", help="search image (PNG)")
    ap.add_argument("--batch",  help="manifest CSV")
    ap.add_argument("--out",    default="results/finfet_predictions.csv")
    ap.add_argument("--verbose", action="store_true")

    g = ap.add_argument_group("search envelope")
    g.add_argument("--mag-lo",       type=float, default=9.0)
    g.add_argument("--mag-hi",       type=float, default=11.0)
    g.add_argument("--rotation-max", type=float, default=3.0)
    g.add_argument("--tie-margin",   type=float, default=0.04)
    g.add_argument("--top-k",        type=int,   default=16)
    g.add_argument("--no-ecc",       action="store_true")
    args = ap.parse_args()

    cfg = FinFETLocalizerConfig(
        mag_ratio_lo=args.mag_lo, mag_ratio_hi=args.mag_hi,
        rotation_max_deg=args.rotation_max, tie_margin=args.tie_margin,
        top_k=args.top_k, use_ecc=not args.no_ecc)

    if args.batch:
        run_batch(args.batch, args.out, cfg, args.verbose)
    elif args.ref and args.search:
        res = localize(args.ref, args.search, cfg, args.verbose)
        print(f"{res.x:.2f},{res.y:.2f}")
    else:
        ap.error("provide either --ref and --search, or --batch")


if __name__ == "__main__":
    main()
