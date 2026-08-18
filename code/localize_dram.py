"""
localize_dram.py — Architecture-Specific Localizer for DRAM SEM Images
=======================================================================
Tuned for DRAM (dynamic random-access memory) SEM image characteristics:

DRAM Physical Characteristics:
  - Oval/pill-shaped active-area cells in a staggered 6F2 grid
  - 2D periodic pattern (both X and Y pitches are significant)
  - Horizontal bitline traces connecting cells within each row
  - Die boundary scribe lines (wide dark bands) — the primary aperiodic cue
  - Nodes: dram_legacy, dram_compact, dram_dense, dram_loose, dram_wide, dram_1x

Algorithm Differences from Generic Localizer:
  1. ISOTROPIC STRUCTURE MAP: Both Gx and Gy weighted equally (2D periodic).
     We add a mild second-derivative (Laplacian) channel to capture the
     oval interior-to-edge brightness transition.
  2. SCRIBE-LINE DETECTION: Dark horizontal/vertical bands are the best
     aperiodic landmark in DRAM; we detect them carefully and use them as
     the primary tiebreak (better than phase correlation for 2D periodic).
  3. CELL-AWARE TIEBREAK: After scribe detection, candidates are grouped
     by die-cell index; the phase-correlation prior identifies which cell
     is correct.
  4. WIDER ROTATION ENVELOPE: DRAM cells can be tilted ±5° by stage drift.
  5. LARGER NMS RADIUS: DRAM pitch is larger than FinFET pitch in the
     downsampled search image, so peaks from adjacent cells are further apart.

Usage (batch):
    python localize_dram.py \\
        --batch drift_sense_dataset/final/train/dram_manifest.csv \\
        --out   results/dram_predictions.csv

Usage (single pair):
    python localize_dram.py --ref REF.png --search SEARCH.png
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

SEARCH_SIZE_DEFAULT    = 1000
REFERENCE_SIZE_DEFAULT = 1000


# ---------------------------------------------------------------------------
# DRAM-specific configuration
# ---------------------------------------------------------------------------

@dataclass
class DRAMLocalizerConfig:
    # Magnification envelope
    mag_ratio_lo: float = 9.0
    mag_ratio_hi: float = 11.0
    n_scale_coarse: int  = 7
    n_scale_fine:   int  = 5

    # DRAM cells can tilt ±5° (stage drift during high-dose DRAM imaging)
    rotation_max_deg: float = 5.0
    n_rot_coarse:     int   = 7
    n_rot_fine:       int   = 5

    # Candidate shortlist
    top_k:      int   = 12
    nms_frac:   float = 0.55    # larger: DRAM cell pitch > FinFET pitch
    tie_margin: float = 0.05

    # Structure map — DRAM specific
    lcn_sigma:      float = 10.0  # larger sigma: ovals are bigger features
    lap_weight:     float = 0.35  # Laplacian contribution (oval boundary)
    trim_frac:      float = 0.10

    # Scribe-line detection
    scribe_valley_sigma: float = 0.70   # valley threshold: mean - sigma*std
    scribe_min_gap:      int   = 50     # min px between separate scribes

    # Refinement
    use_ecc:        bool  = True
    ecc_iterations: int   = 60
    ecc_eps:        float = 1e-5
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
# DRAM structure map — isotropic + Laplacian oval-edge channel
# ---------------------------------------------------------------------------

def dram_structure_map(img: np.ndarray, cfg: DRAMLocalizerConfig) -> np.ndarray:
    """
    Illumination-invariant structure representation for DRAM cells.

    DRAM active areas are oval blobs → both horizontal and vertical gradients
    carry signal (unlike FinFETs).  Adding a normalised Laplacian channel
    captures the oval's interior brightness transition, which provides
    additional contrast between the oval interior and the surrounding
    dielectric substrate.

    Steps:
      1. Median filter (removes impulse noise)
      2. Isotropic gradient magnitude
      3. Laplacian (captures oval interior transitions)
      4. Combine: grad + lap_weight * |Lap|
      5. LCN with a slightly larger sigma (ovals are bigger than fins)
    """
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f /= 255.0

    f_u8 = (f * 255).clip(0, 255).astype(np.uint8)
    f_u8 = cv2.medianBlur(f_u8, 3)
    f    = f_u8.astype(np.float32) / 255.0

    # Gradient magnitude (isotropic)
    gx   = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # Laplacian — captures the oval blob interior transitions
    lap = np.abs(cv2.Laplacian(f, cv2.CV_32F, ksize=3))

    # Combine
    combined = grad + cfg.lap_weight * lap

    # LCN
    sigma    = cfg.lcn_sigma
    k        = int(2 * round(3 * sigma) + 1)
    mean     = cv2.GaussianBlur(combined,             (k, k), sigma)
    mean_sq  = cv2.GaussianBlur(combined * combined,  (k, k), sigma)
    std      = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)) + 1e-4
    return ((combined - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

def build_template(ref_gray: np.ndarray, footprint_px: float,
                   rotation_deg: float, cfg: DRAMLocalizerConfig) -> np.ndarray:
    t = int(round(footprint_px))
    if t < 12:
        raise ValueError(f"footprint too small: {footprint_px}")
    base = cv2.resize(ref_gray, (t, t), interpolation=cv2.INTER_AREA)
    return _finish_template(base, t, rotation_deg, cfg)


def _finish_template(base: np.ndarray, t: int, rotation_deg: float,
                     cfg: DRAMLocalizerConfig) -> np.ndarray:
    if abs(rotation_deg) > 1e-6:
        M = cv2.getRotationMatrix2D(((t - 1) / 2.0, (t - 1) / 2.0),
                                    -rotation_deg, 1.0)
        base = cv2.warpAffine(base, M, (t, t), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    tpl = dram_structure_map(base, cfg)

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
                 footprints, rotations, cfg: DRAMLocalizerConfig):
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
# Scribe-line detection (primary DRAM aperiodic landmark)
# ---------------------------------------------------------------------------

def _detect_scribe_lines(search_gray: np.ndarray,
                         cfg: DRAMLocalizerConfig) -> tuple[list, list]:
    """
    Detect horizontal and vertical DRAM scribe lanes (wide dark separator
    bands between dies).

    Scribe lines are consistently dark rows/columns spanning the full image.
    They are the single most reliable aperiodic landmark in a DRAM image
    and the primary tool for disambiguating identical-looking cell repeats.

    Detection is the same as the generic localizer but with a configurable
    valley threshold (cfg.scribe_valley_sigma).
    """
    img = search_gray.astype(np.float32)

    def _find(profile, min_gap, sigma_mult):
        mu, sd   = profile.mean(), profile.std()
        thresh   = mu - sigma_mult * sd
        in_band  = profile < thresh
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

    row_means  = img.mean(axis=1)
    col_means  = img.mean(axis=0)
    row_scribes = _find(row_means, cfg.scribe_min_gap, cfg.scribe_valley_sigma)
    col_scribes = _find(col_means, cfg.scribe_min_gap, cfg.scribe_valley_sigma)
    return row_scribes, col_scribes


def _cell_index(x: float, y: float,
                row_scribes: list, col_scribes: list,
                H: int, W: int) -> tuple[int, int]:
    """Return (row_cell, col_cell) die-cell index for point (x, y)."""
    col_cell = sum(1 for b in col_scribes if x >= b)
    row_cell = sum(1 for b in row_scribes if y >= b)
    return row_cell, col_cell


# ---------------------------------------------------------------------------
# Phase-correlation prior
# ---------------------------------------------------------------------------

def _phase_corr_prior(ref_gray: np.ndarray, search_gray: np.ndarray,
                      fp: float) -> tuple[float, float]:
    """
    Phase-correlation coarse prior.

    For DRAM this is used as a secondary cue (scribe lines are primary).
    It still helps when the die is large and scribe lines are at the edges.
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

    wy  = np.hanning(H).astype(np.float32)
    wx  = np.hanning(W).astype(np.float32)
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
# ECC sub-pixel refinement
# ---------------------------------------------------------------------------

def ecc_refine(search_struct: np.ndarray, tpl: np.ndarray,
               cx: float, cy: float, cfg: DRAMLocalizerConfig):
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

def localize(ref, search, cfg: DRAMLocalizerConfig | None = None,
             verbose: bool = False) -> LocalizationResult:
    """Locate `ref` inside `search`. Accepts arrays or file paths."""
    cfg = cfg or DRAMLocalizerConfig()
    t_start = time.perf_counter()
    times   = {}

    ref_gray    = load_gray(ref)    if isinstance(ref,    str) else ref
    search_gray = load_gray(search) if isinstance(search, str) else search
    H, W = search_gray.shape[:2]

    # ── Structure map (computed once) ───────────────────────────────────────
    t0 = time.perf_counter()
    search_struct = np.ascontiguousarray(
        dram_structure_map(search_gray, cfg))
    times["structure"] = time.perf_counter() - t0

    # ── Scribe-line detection (primary aperiodic cue for DRAM) ──────────────
    t0 = time.perf_counter()
    row_scribes, col_scribes = _detect_scribe_lines(search_gray, cfg)
    times["scribes"] = time.perf_counter() - t0

    # ── Phase-correlation prior (Deferred to tie-breaker) ────────────────────
    ref_n      = cfg.reference_size
    fp_lo      = ref_n / cfg.mag_ratio_hi
    fp_hi      = ref_n / cfg.mag_ratio_lo
    fp_nominal = (fp_lo + fp_hi) / 2.0
    prior_x, prior_y = -1.0, -1.0

    # ── Coarse NCC grid ──────────────────────────────────────────────────────
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

    # ── Fine NCC grid ────────────────────────────────────────────────────────
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

    # ── DECISION RULE (DRAM-specific priority order) ─────────────────────────
    # Priority:
    #   1. Remove candidates landing on a scribe lane (boundary match artefact)
    #   2. Among survivors: prefer candidates in the SAME die cell as the
    #      phase-correlation prior (this is the primary disambiguation for DRAM)
    #   3. Final fallback: closest to image centre (problem-statement rule)
    centre = np.array([W / 2.0, H / 2.0])
    SCRIBE_EXCL = 8   # slightly wider than generic (DRAM scribes are wider)

    if len(tied) > 1:
        def _on_scribe(c):
            for rs in row_scribes:
                if abs(c[2] - rs) < SCRIBE_EXCL:
                    return True
            for cs in col_scribes:
                if abs(c[1] - cs) < SCRIBE_EXCL:
                    return True
            return False

        interior = [c for c in tied if not _on_scribe(c)]
        pool = interior if interior else tied

        if len(pool) > 1 and (row_scribes or col_scribes):
            # Step 2: prefer candidates in the same die cell as prior
            tied_fp = pool[0][3]
            t0 = time.perf_counter()
            prior_x, prior_y = _phase_corr_prior(ref_gray, search_gray, tied_fp)
            times["phase_corr"] = time.perf_counter() - t0
            
            prior_cell = _cell_index(prior_x, prior_y,
                                     row_scribes, col_scribes, H, W)
            same_cell  = [c for c in pool
                          if _cell_index(c[1], c[2], row_scribes, col_scribes,
                                         H, W) == prior_cell]
            pool = same_cell if same_cell else pool

        # Final tiebreak: closest to image centre
        winner = min(pool, key=lambda c: (c[1] - centre[0]) ** 2
                     + (c[2] - centre[1]) ** 2)
    else:
        winner = cands[0]

    score, cx, cy, fp, rot = winner

    # Confidence vs best distinct rival
    rivals     = [c[0] for c in cands
                  if (c[1] - cx) ** 2 + (c[2] - cy) ** 2
                  > (nominal_fp * 0.5) ** 2]
    confidence = float(score - max(rivals)) if rivals else float(score)

    # ── Sub-pixel ECC refinement ─────────────────────────────────────────────
    t0 = time.perf_counter()
    if cfg.use_ecc:
        tpl      = build_template(ref_gray, fp, rot, cfg)
        cx, cy, _ = ecc_refine(search_struct, tpl, cx, cy, cfg)
    times["refine"] = time.perf_counter() - t0

    cx      = float(np.clip(cx, 0.0, W))
    cy      = float(np.clip(cy, 0.0, H))
    runtime = time.perf_counter() - t_start

    if verbose:
        print(f"[dram] scribes=({len(row_scribes)}h,{len(col_scribes)}v)  "
              f"prior=({prior_x:.1f},{prior_y:.1f})", file=sys.stderr)
        print(f"[dram] geom fp={fp:.2f}px rot={rot:+.2f}° mag={ref_n/fp:.2f}:1",
              file=sys.stderr)
        print(f"[dram] tied={len(tied)}  result=({cx:.2f},{cy:.2f})",
              file=sys.stderr)
        print(f"[dram] times={times}  total={runtime:.3f}s", file=sys.stderr)

    if len(tied) > 1:
        print(f"DRAM AMBIGUOUS: {len(tied)} candidates tied  "
              f"scribes=({len(row_scribes)}h,{len(col_scribes)}v)",
              file=sys.stderr)

    return LocalizationResult(cx, cy, float(score), confidence, float(fp),
                              float(rot), len(tied), len(tied) > 1,
                              runtime, times)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_batch(manifest: str, out_path: str, cfg: DRAMLocalizerConfig,
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
    print(f"\nDRAM predictions -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DRAM-specialised SEM localizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ref",    help="reference image (PNG)")
    ap.add_argument("--search", help="search image (PNG)")
    ap.add_argument("--batch",  help="manifest CSV")
    ap.add_argument("--out",    default="results/dram_predictions.csv")
    ap.add_argument("--verbose", action="store_true")

    g = ap.add_argument_group("search envelope")
    g.add_argument("--mag-lo",       type=float, default=9.0)
    g.add_argument("--mag-hi",       type=float, default=11.0)
    g.add_argument("--rotation-max", type=float, default=5.0)
    g.add_argument("--tie-margin",   type=float, default=0.05)
    g.add_argument("--top-k",        type=int,   default=12)
    g.add_argument("--no-ecc",       action="store_true")
    args = ap.parse_args()

    cfg = DRAMLocalizerConfig(
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
