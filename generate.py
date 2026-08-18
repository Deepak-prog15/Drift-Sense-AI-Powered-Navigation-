"""
generator.py -- PS2 Drift-Sense synthetic wafer-inspection dataset generator.

Produces reference/search image pairs for the navigation-error localization
problem, with physics-grounded SEM degradations. Every degradation term is
justified by a published source (see PS2_DATASET_CARD.md).

    reference.png : 1000x1000 grayscale, high magnification, small FOV
    search.png    : 1000x1000 grayscale, 10x the physical FOV
                    -> the reference region occupies ~100x100 px inside it
    ground_truth.csv : pair_id, ref_path, search_path, true_x, true_y, + params

Pipeline per pair
-----------------
 1. Render ONE unit cell of the chosen layout style, np.tile it to a
    10000x10000 uint8 "master" = the full 10x field of view.
 2. Optionally add aperiodic content (scribe lines, dummy blocks, array edge).
 3. REFERENCE = full-resolution 1000x1000 crop at a random (tx, ty).
 4. SEARCH    = the whole master INTER_AREA-downsampled to 1000x1000.
    (Anti-aliased decimation is what an optical/scan system physically does.)
 5. Apply relative stage rotation + scale jitter to the REFERENCE
    (only the *relative* geometry between the two captures matters, and
     warping 1000x1000 is ~100x cheaper than warping 10000x10000).
 6. Degrade each image with an INDEPENDENT sem_capture() call.

Ground truth: (true_x, true_y) = (tx / 10, ty / 10) in search-image pixels.

Usage
-----
    python generator.py --style dram --num 500 --out ../data/train --seed 42
    python generator.py --num 40 --out ../data/noise_2x --seed 901 --noise-scale 2.0
    python generator.py --num 80 --out ../data/hard_periodic --seed 1234 --pure-periodic
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

ZOOM = 10          # search image covers 10x the linear FOV of the reference
OUT = 1000         # both images are 1000 x 1000
MASTER = OUT * ZOOM  # 10000 x 10000


# ============================================================================
# 1. LAYOUT STYLES  -- one unit cell each, then tiled
# ============================================================================
def cell_dram(pitch, line_w, via_r, rng):
    """DRAM array: horizontal word lines x vertical bit lines, contact/via
    dot at every intersection. Classic 'checkerboard of dots' top-down SEM
    appearance of a memory array."""
    c = np.full((pitch, pitch), 56, np.uint8)          # dark dielectric
    c[:line_w, :] = 132                                # word line (horizontal)
    c[:, :line_w] = 148                                # bit line  (vertical)
    cv2.circle(c, (line_w // 2, line_w // 2), via_r, 224, -1)
    return c


def cell_secondary(pitch, line_w, via_r, rng):
    """Secondary structure: dense vertical fins at half pitch, crossed by wider
    horizontal gate bars. Strongly 1-D periodic -> a harder ambiguity case
    than DRAM because there is far less 2-D structure to lock onto."""
    c = np.full((pitch, pitch), 52, np.uint8)
    fin_w = max(3, line_w // 2)
    c[:, :fin_w] = 150                                 # fin
    c[:, pitch // 2: pitch // 2 + fin_w] = 150         # second fin (half pitch)
    gate_h = max(5, int(line_w * 1.4))
    c[:gate_h, :] = 190                                # gate bar
    return c


def cell_logic(pitch, line_w, via_r, rng):
    """Standard-cell / logic-like: a routing track grid with vias present at
    only ~35% of intersections and short random jog segments. Semi-periodic,
    which is the realistic middle ground between DRAM and true random."""
    c = np.full((pitch, pitch), 60, np.uint8)
    c[:line_w, :] = 140
    c[:, :line_w] = 128
    if rng.random() < 0.35:
        cv2.circle(c, (line_w // 2, line_w // 2), via_r, 220, -1)
    if rng.random() < 0.5:                             # short jog / stub
        y0 = int(rng.integers(line_w, max(line_w + 1, pitch - line_w)))
        c[y0:y0 + max(2, line_w // 2), : pitch // 2] = 118
    return c


CELL_FN = {"dram": cell_dram, "secondary": cell_secondary, "logic": cell_logic}


def make_master(style, pitch, line_w, via_r, rng, pure_periodic=False):
    """Tile the unit cell to MASTER x MASTER.

    For 'logic', a single cell would make the random vias periodic too, so we
    build a 6x6 super-block of independently randomized cells and tile that --
    the layout is then periodic at the super-block scale, like real standard
    cell rows, rather than at the single-cell scale.
    """
    fn = CELL_FN[style]
    if style == "logic":
        blk = 6
        rows = [np.hstack([fn(pitch, line_w, via_r, rng) for _ in range(blk)])
                for _ in range(blk)]
        unit = np.vstack(rows)
    else:
        unit = fn(pitch, line_w, via_r, rng)

    reps = int(np.ceil(MASTER / unit.shape[0])) + 1
    m = np.tile(unit, (reps, reps))
    # random array phase so the grid origin is not pinned to (0, 0)
    m = np.roll(m, (int(rng.integers(0, unit.shape[0])),
                    int(rng.integers(0, unit.shape[1]))), axis=(0, 1))
    m = np.ascontiguousarray(m[:MASTER, :MASTER])
    if not pure_periodic:
        m = add_aperiodic(m, rng)
    return m


def add_aperiodic(m, rng):
    """Real dies are not infinite grids. Add scribe/route strips, dummy
    peripheral-circuitry blocks, and occasionally a hard array edge. These are
    the ONLY globally unambiguous cues a localizer can lock onto -- how many
    are present directly controls how hard the sample is."""
    s = MASTER
    for _ in range(int(rng.integers(1, 4))):           # vertical route strips
        w = int(rng.integers(s // 60, s // 22))
        x0 = int(rng.integers(0, s - w))
        m[:, x0:x0 + w] = int(rng.integers(70, 100))
    for _ in range(int(rng.integers(1, 4))):           # horizontal route strips
        h = int(rng.integers(s // 60, s // 22))
        y0 = int(rng.integers(0, s - h))
        m[y0:y0 + h, :] = int(rng.integers(70, 100))
    for _ in range(int(rng.integers(1, 5))):           # dummy blocks
        b = int(rng.integers(s // 22, s // 9))
        by, bx = int(rng.integers(0, s - b)), int(rng.integers(0, s - b))
        m[by:by + b, bx:bx + b] = int(rng.integers(150, 200))
    if rng.random() < 0.35:                            # hard array edge
        if rng.random() < 0.5:
            x0 = int(rng.integers(s // 8, s // 3))
            m[:, :x0] = 78
        else:
            y0 = int(rng.integers(s // 8, s // 3))
            m[:y0, :] = 78
    return m


# ============================================================================
# 2. SEM CAPTURE SIMULATION -- call ONCE PER IMAGE, never share noise arrays
# ============================================================================
def line_edge_roughness(img, rng, amp_px=0.9):
    """Line-edge roughness: real lithographic edges wander. LER is defined as
    the 3-sigma deviation of an edge from a straight line and is ~2 nm at the
    65 nm DRAM half-pitch node -- i.e. clearly visible at high magnification
    and completely invisible after 10x decimation, which is exactly how we
    apply it (reference only)."""
    h, w = img.shape
    small = (16, 16)
    dx = cv2.resize(rng.normal(0, amp_px, small).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.normal(0, amp_px, small).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def scan_distortion(img, rng, amp_px=1.2):
    """Raster scan drift: SEM builds an image line by line, so thermal drift,
    sample charging and scan-coil hysteresis shear successive scanlines
    horizontally. Modelled as a smooth low-frequency per-row x-offset."""
    h, w = img.shape
    ctrl = rng.normal(0, amp_px, 10).astype(np.float32)
    off = cv2.resize(ctrl.reshape(-1, 1), (1, h),
                     interpolation=cv2.INTER_CUBIC).ravel()
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, xx + off[:, None], yy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def sem_capture(clean_u8, rng, ns=1.0, dose=(90, 300), blur=(0.7, 1.6),
                edge_gain=(0.20, 0.45), do_ler=False, do_charge=True):
    """Full independent SEM capture of a clean layout.

    Order follows the physical chain:
      geometry distortion -> edge brightening -> beam PSF -> shot noise
      -> speckle -> readout noise -> gain/offset drift
    """
    img = clean_u8.astype(np.float32) / 255.0

    if do_ler:
        img = line_edge_roughness(img, rng, amp_px=rng.uniform(0.4, 1.3))
    img = scan_distortion(img, rng, amp_px=rng.uniform(0.3, 1.6) * ns)

    # -- secondary-electron edge brightening (bright halo on sidewalls) --
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    e = cv2.magnitude(gx, gy)
    img += rng.uniform(*edge_gain) * (e / (e.max() + 1e-8))

    # -- electron-beam point spread function --
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=rng.uniform(*blur))

    # -- sample charging: slow bright/dark field gradient --
    if do_charge and rng.random() < 0.5:
        h, w = img.shape
        g = cv2.resize(rng.normal(0, 0.05, (4, 4)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
        img += g

    # -- Poisson shot noise: THE dominant SEM noise term. Lower dose = grainier
    d = rng.uniform(*dose) / max(ns, 1e-6)
    img = rng.poisson(np.clip(img, 0, None) * d).astype(np.float32) / d

    # -- multiplicative speckle (detector gain non-uniformity / SE yield var) --
    img *= (1.0 + rng.normal(0, 0.06 * ns, img.shape).astype(np.float32))

    # -- additive Gaussian readout noise (amplifier chain) --
    img += rng.normal(0, 0.022 * ns, img.shape).astype(np.float32)

    # -- per-capture brightness / contrast drift (motivates NCC over SSD) --
    img = img * rng.uniform(0.88, 1.12) + rng.uniform(-0.06, 0.06)
    return img


def to_png(img):
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    return np.clip((img - lo) / (hi - lo + 1e-8) * 255.0, 0, 255).astype(np.uint8)


# ============================================================================
# 3. ONE PAIR
# ============================================================================
def generate_pair(pid, out_dir, rng, style="dram", ns=1.0, pure_periodic=False):
    # Pitch range chosen so that after 10x decimation the array pitch lands in
    # 4.5-12 search-image pixels: the low end is near-unresolvable gray texture
    # (localizer must use aperiodic cues), the high end stays clearly resolved.
    # Covering both is what makes the set a real difficulty sweep.
    pitch = int(rng.integers(45, 121))
    line_w = max(6, pitch // int(rng.integers(4, 8)))
    via_r = max(3, line_w // 2)

    master = make_master(style, pitch, line_w, via_r, rng, pure_periodic)

    # --- true location of the reference inside the wide FOV ---
    m = OUT // 2 + 120
    tx = int(rng.integers(m, MASTER - m))
    ty = int(rng.integers(m, MASTER - m))

    ref_clean = master[ty - OUT // 2: ty + OUT // 2,
                       tx - OUT // 2: tx + OUT // 2].copy()
    search_clean = cv2.resize(master, (OUT, OUT), interpolation=cv2.INTER_AREA)
    del master

    # --- relative stage rotation + zoom-calibration error on the reference ---
    ang = float(rng.uniform(-5, 5))
    sc = float(rng.uniform(0.96, 1.04))
    M = cv2.getRotationMatrix2D((OUT / 2, OUT / 2), ang, sc)
    ref_clean = cv2.warpAffine(ref_clean, M, (OUT, OUT),
                               borderMode=cv2.BORDER_REFLECT)

    gt_x, gt_y = tx / ZOOM, ty / ZOOM

    # --- two INDEPENDENT captures; the wide scan is faster => lower dose ---
    ref_img = sem_capture(ref_clean, rng, ns=ns, dose=(150, 320),
                          blur=(0.7, 1.4), do_ler=True)
    search_img = sem_capture(search_clean, rng, ns=ns * 1.4, dose=(70, 160),
                             blur=(0.8, 1.7), do_ler=False)

    rp = os.path.join(out_dir, "images", f"{pid:05d}_ref.png")
    sp = os.path.join(out_dir, "images", f"{pid:05d}_search.png")
    cv2.imwrite(rp, to_png(ref_img))
    cv2.imwrite(sp, to_png(search_img))
    return dict(pair_id=pid, ref_path=os.path.relpath(rp, out_dir),
                search_path=os.path.relpath(sp, out_dir),
                true_x=round(gt_x, 3), true_y=round(gt_y, 3),
                style=style, pitch=pitch, line_w=line_w, via_r=via_r,
                rot_deg=round(ang, 3), scale=round(sc, 4),
                noise_scale=ns, pure_periodic=int(pure_periodic))


# ============================================================================
# 4. GT OVERLAY CHECK
# ============================================================================
def gt_overlay(out_dir, rows, n=12):
    """Draw the GT box + a magnified inset on the first n search images.
    Looking at these is the ONLY way to catch a silent ground-truth bug."""
    d = os.path.join(out_dir, "gt_check")
    os.makedirs(d, exist_ok=True)
    half = OUT // (2 * ZOOM)
    for r in rows[:n]:
        img = cv2.imread(os.path.join(out_dir, r["search_path"]), 0)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x, y = int(r["true_x"]), int(r["true_y"])
        patch = vis[max(0, y - half):y + half, max(0, x - half):x + half].copy()
        cv2.rectangle(vis, (x - half, y - half), (x + half, y + half), (0, 0, 255), 2)
        if patch.size:
            vis[10:310, 10:310] = cv2.resize(patch, (300, 300),
                                             interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(vis, (10, 10), (310, 310), (0, 255, 255), 2)
        cv2.imwrite(os.path.join(d, f"{r['pair_id']:05d}_gt.png"), vis)


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="mixed",
                    choices=["dram", "secondary", "logic", "mixed"])
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--pure-periodic", action="store_true",
                    help="no aperiodic cues at all -- the genuinely hard case")
    ap.add_argument("--gt-check", type=int, default=12)
    ap.add_argument("--start-id", type=int, default=0,
                    help="offset pair ids and APPEND to ground_truth.csv, so a "
                         "large split can be generated in several chunks")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    styles = ["dram", "secondary", "logic"]
    rows, t0 = [], time.time()

    # CSV is written INCREMENTALLY (one row per pair, flushed) so that a run
    # interrupted part-way still leaves a valid, self-consistent dataset.
    csv_path = os.path.join(a.out, "ground_truth.csv")
    append = a.start_id > 0 and os.path.exists(csv_path)
    fields = ["pair_id", "ref_path", "search_path", "true_x", "true_y", "style",
              "pitch", "line_w", "via_r", "rot_deg", "scale", "noise_scale",
              "pure_periodic"]
    with open(csv_path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not append:
            w.writeheader()
        for i in range(a.num):
            st = (styles[int(rng.integers(0, 3))] if a.style == "mixed" else a.style)
            r = generate_pair(a.start_id + i, a.out, rng, st,
                              a.noise_scale, a.pure_periodic)
            rows.append(r)
            w.writerow(r)
            f.flush()
            if (i + 1) % 25 == 0 or i + 1 == a.num:
                el = time.time() - t0
                print(f"  {i+1}/{a.num}  {el:.0f}s  ({el/(i+1):.2f}s/pair)",
                      flush=True)

    if a.gt_check:
        gt_overlay(a.out, rows, a.gt_check)
    print(f"DONE {a.out}: {len(rows)} pairs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
