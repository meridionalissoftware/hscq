#!/usr/bin/env python3
"""The hsc compression-ratio model: an exact rate predictor and an empirical distortion predictor.

    python3 scripts/compression_ratio_model.py            # fit, validate, and print the report
    python3 scripts/compression_ratio_model.py --validate # exit nonzero if the RATE mismatches
    python3 scripts/compression_ratio_model.py --predict json 30
"""

import math
import os
import subprocess
import sys
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
BENCH = os.path.join(ROOT, "bin", "compression_ratio_bench")

WORK = 1 << 19  # must match k_work in benches/compression_ratio_bench.cpp

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# rate: the exact mirror of sphere/{s3,tree}.hpp and codec/pack.hpp

DQ1 = 1 << 24
DQ_MIN = 1678
DQ_MAX = 2 << 24

LEVEL_DQ = {
    1: 15099494, 2: 11744051, 3: 8388608, 4: 6710886, 5: 5033165, 6: 3355443,
    7: 1677722, 8: 838861, 9: 335544, 10: 167772, 11: 83886, 12: 33554,
    13: 16777, 14: 8389, 15: 3355, 16: 1678,
}


def dq_of(d):
    if d <= 0.0:
        return 0
    if d >= 2.0:
        return DQ_MAX
    return int(d * DQ1 + 0.5)


def d_of(dq):
    return dq / DQ1


# tree.hpp: 2048 members, d *= 16303/16384 from d = 2.  The collapse this induces on the reachable
# child-distance set is what makes the memo finite; changing it is a format break.
_GRID = []
_d = 2.0
for _ in range(2048):
    _GRID.append(dq_of(_d))
    _d = _d * 16303.0 / 16384.0


def grid_snap(dp):
    """smallest grid member >= dp, as d_q (tree.hpp:58)"""
    if not (dp > 0.0) or dp >= 2.0:
        return _GRID[0]
    dqp = int(math.ceil(dp * DQ1))
    if dqp >= _GRID[0]:
        return _GRID[0]
    if dqp <= _GRID[-1]:
        return _GRID[-1]
    lo, hi = 0, len(_GRID) - 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if _GRID[mid] >= dqp:
            lo = mid
        else:
            hi = mid - 1
    return _GRID[lo]


def _s3_d(dq):
    d = d_of(dq)
    return 2.0 if d > 2.0 else d


def _s3_half(d):
    return int(math.floor(math.pi / (4.0 * math.asin(d * 0.5)))) // 2


def _s3_m(d, ce):
    """points on one internal circle (Proposition 2a, s3.hpp:88)"""
    if d > 2.0 * ce:
        return 1
    return int(math.floor(math.pi / math.asin(d / (2.0 * ce))))


def _s3_n(d, ce, se, m):
    """internal circles on the torus (Proposition 2b, s3.hpp:98) -- note the even round-down"""
    if d > 2.0 * se:
        return 1
    n2 = math.floor(2.0 * math.pi / math.asin(d / (2.0 * se)))
    sh = math.sin(math.pi / (2.0 * m))
    arg = (d * d * 0.25) / (se * se) - (ce * ce) / (se * se) * sh * sh
    n1 = n2
    if arg > 0.0:
        r = math.sqrt(arg)
        n1 = 1.0 if r > 1.0 else math.floor(math.pi / math.asin(r))
    ntil = 2 * (int(min(n1, n2)) // 2)
    return ntil if ntil > 1 else 1


@lru_cache(maxsize=None)
def M_s3(dq):
    """M(4, d): the 4D base case, exact"""
    d = _s3_d(dq)
    half = _s3_half(d)
    deta = 2.0 * math.asin(d * 0.5)
    tot = 0
    for x in range(2 * half + 1):
        eta = math.pi / 4.0 + (x - half) * deta
        se, ce = math.sin(eta), math.cos(eta)
        m = _s3_m(d, ce)
        tot += m * _s3_n(d, ce, se, m)
    return tot


@lru_cache(maxsize=None)
def M_tree(dim_log2, dq):
    """M(2^dim_log2, d): the recursion, in exact Python bigints (the C++ carries it in arbint)"""
    if dim_log2 == 2:
        return M_s3(dq)
    d = _s3_d(dq)
    half = _s3_half(d)
    deta = 2.0 * math.asin(d * 0.5)
    tot = 0
    for x in range(2 * half + 1):
        eta = math.pi / 4.0 + (x - half) * deta
        se, ce = math.sin(eta), math.cos(eta)
        tot += M_tree(dim_log2 - 1, grid_snap(d / ce if ce > 0.0 else 2.0)) * M_tree(
            dim_log2 - 1, grid_snap(d / se if se > 0.0 else 2.0)
        )
    return tot


@lru_cache(maxsize=None)
def M_s2(dq):
    """M_S2(d): the quotient lane's band code (s2.hpp:59)"""
    d = _s3_d(dq)
    dth = 2.0 * math.asin(d * 0.5)
    count = int(math.floor(math.pi / dth)) + 1
    th0 = (math.pi - (count - 1) * dth) / 2.0
    tot = 0
    for b in range(count):
        th = th0 + b * dth
        st = math.sin(th)
        tot += int(math.floor(math.pi / math.asin(d / (2.0 * st)))) if d <= 2.0 * st else 1
    return tot


# susp.hpp: the quat/oct lanes' CAP-ANCHORED, EQUATOR-ANCHORED suspension over S^3/S^7 child
# codes.  Bands 0 and count-1 are EXACT pole codewords (theta = 0, pi); interior bands live in
# [dth, pi - dth] with the residual centered AND the count forced ODD so the equator is always
# a band CENTER -- both-unit pairs (pose/spinor data) have h = 0 exactly, and an even count
# would park them on a knife-edge cell boundary where representative noise flips the band.
# Each interior band carries the tree code at grid_snap(d / sin theta_b).  This layout
# deliberately diverges from s2.hpp's centered bands (poles are the [x:0]/[0:y] classes and
# must be exact); s2/quotient stay frozen as they are.
def susp_thetas(dq):
    """interior band colatitudes, poles excluded; [] when pi - 2*dth < 0 (poles only)"""
    d = _s3_d(dq)
    dth = 2.0 * math.asin(d * 0.5)
    w = math.pi - 2.0 * dth
    if w < 0.0:
        return []
    tp = int(math.floor(w / dth)) + 1
    if tp % 2 == 0:
        tp -= 1
    th_first = dth + (w - (tp - 1) * dth) / 2.0
    return [th_first + b * dth for b in range(tp)]


@lru_cache(maxsize=None)
def M_susp(child_dim_log2, dq):
    """M for the S^4 (child 2, mode quat) / S^8 (child 3, mode oct) suspension: 2 poles + bands"""
    d = _s3_d(dq)
    tot = 2
    for th in susp_thetas(dq):
        st = math.sin(th)
        tot += M_tree(child_dim_log2, grid_snap(d / st if st > 0.0 else 2.0))
    return tot


def shape_bits(dim_log2, dq):
    return (M_tree(dim_log2, dq) - 1).bit_length()


def rate(mode="bin", dim_log2=3, level=6, gain_bits=8, dq=None):
    """the closed form from src/hsc/rate.hpp, as a dict"""
    dq = dq if dq is not None else LEVEL_DQ[level]
    if mode == "quotient":
        sb, be, gb = (M_s2(dq) - 1).bit_length(), 4, 0
    elif mode in ("quat", "oct"):
        sb = (M_susp(2 if mode == "quat" else 3, dq) - 1).bit_length()
        be, gb = (8 if mode == "quat" else 16), 0
    else:
        sb = shape_bits(dim_log2, dq)
        be = 1 << dim_log2
        gb = gain_bits if mode in ("bin", "vec") else 0
    rec = gb + sb
    bpe = rec / be
    bpib = bpe if mode == "bin" else bpe / 4.0
    return {
        "mode": mode, "dim": be if mode != "quotient" else 4, "dq": dq, "d": d_of(dq),
        "shape_bits": sb, "gain_bits": gb, "record_bits": rec, "block_elems": be,
        "bits_per_elem": bpe, "bits_per_input_byte": bpib,
        "ratio": 8.0 / bpib if bpib else float("inf"),
        "degenerate": sb == 0,
    }


def encoded_size(n_elems, mode="bin", dim_log2=3, level=6, gain_bits=8, dq=None):
    r = rate(mode, dim_log2, level, gain_bits, dq)
    nb = (n_elems + r["block_elems"] - 1) // r["block_elems"]
    return 40 + (nb * r["record_bits"] + 7) // 8 + 8


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# goldens: the mirror has to reproduce the in-tree values before anything else it says counts

GOLDENS = [
    ("s3_m(L1)", lambda: M_s3(LEVEL_DQ[1]), 16),
    ("s3_m(L3)", lambda: M_s3(LEVEL_DQ[3]), 138),
    ("s3_m(L6)", lambda: M_s3(LEVEL_DQ[6]), 2588),
    ("s3_m(L9)", lambda: M_s3(LEVEL_DQ[9]), 2828294),
    ("s3_m(L10)", lambda: M_s3(LEVEL_DQ[10]), 22704306),
    ("s3_m(L13)", lambda: M_s3(LEVEL_DQ[13]), 22785126711),
    ("s3_m(L16)", lambda: M_s3(LEVEL_DQ[16]), 22780659936258),
    ("s2_m(L3)", lambda: M_s2(LEVEL_DQ[3]), 46),
    ("s2_m(L7)", lambda: M_s2(LEVEL_DQ[7]), 1236),
    ("s4_m(L3)", lambda: M_susp(2, LEVEL_DQ[3]), 332),
    ("s4_m(L6)", lambda: M_susp(2, LEVEL_DQ[6]), 16720),
    ("s4_m(L10)", lambda: M_susp(2, LEVEL_DQ[10]), 3002716172),
    ("s4_m(L16)", lambda: M_susp(2, LEVEL_DQ[16]), 301510320474072434),
    ("s4_bits(L6)", lambda: (M_susp(2, LEVEL_DQ[6]) - 1).bit_length(), 15),
    ("s8_m(L3)", lambda: M_susp(3, LEVEL_DQ[3]), 4552),
    ("s8_m(L6)", lambda: M_susp(3, LEVEL_DQ[6]), 10923842),
    ("s8_m_mod64(L10)", lambda: M_susp(3, LEVEL_DQ[10]) % (1 << 64), 378280722294130556),
    ("s8_m_mod64(L12)", lambda: M_susp(3, LEVEL_DQ[12]) % (1 << 64), 10542505927947969120),
    ("s8_bits(L6)", lambda: (M_susp(3, LEVEL_DQ[6]) - 1).bit_length(), 24),
    ("s8_bits(L12)", lambda: (M_susp(3, LEVEL_DQ[12]) - 1).bit_length(), 77),
    ("tree_m(dim8,L3)", lambda: M_tree(3, LEVEL_DQ[3]), 2310),
    ("tree_m(dim16,L3)", lambda: M_tree(4, LEVEL_DQ[3]), 60316),
    ("tree_m_mod64(dim16,L6)", lambda: M_tree(4, LEVEL_DQ[6]) % (1 << 64), 73150212400),
    ("shape_bits(dim8,L3)", lambda: shape_bits(3, LEVEL_DQ[3]), 12),
    ("shape_bits(dim8,L5)", lambda: shape_bits(3, LEVEL_DQ[5]), 17),
    ("shape_bits(dim8,L6)", lambda: shape_bits(3, LEVEL_DQ[6]), 22),
    ("shape_bits(dim16,L3)", lambda: shape_bits(4, LEVEL_DQ[3]), 16),
    ("shape_bits(dim16,L6)", lambda: shape_bits(4, LEVEL_DQ[6]), 37),
    ("shape_bits(dim32,L5)", lambda: shape_bits(5, LEVEL_DQ[5]), 39),
    ("shape_bits(dim32,L7)", lambda: shape_bits(5, LEVEL_DQ[7]), 90),
    ("shape_bits(dim64,L5)", lambda: shape_bits(6, LEVEL_DQ[5]), 58),
    ("shape_bits(dim64,L7)", lambda: shape_bits(6, LEVEL_DQ[7]), 148),
]


def check_goldens(verbose=True):
    bad = 0
    for name, fn, want in GOLDENS:
        got = fn()
        ok = got == want
        bad += not ok
        if verbose:
            print(f"  {name:<26} {got:>16} expect {want:>16}  {'OK' if ok else '** MISMATCH **'}")
    return bad


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# the measured grid, and the distortion fit


def bench_rows(path=None):
    """run (or read) bin/compression_ratio_bench and parse its fixed-rate grid"""
    if path and os.path.exists(path):
        text = open(path).read()
    else:
        if not os.path.exists(BENCH):
            print(f"missing {BENCH} -- run: duck batch build.duck", file=sys.stderr)
            return [], {}
        text = subprocess.run([BENCH], capture_output=True, text=True, cwd=ROOT).stdout
    rows, sizes, cur = [], {}, None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("[") and "n=" in s:
            cur = s[1: s.index("]")]
            try:
                sizes[cur] = int(s.split("n=")[1].split()[0])
            except (ValueError, IndexError):
                pass
            continue
        f = s.split()
        if len(f) == 10 and cur and f[0] == cur and f[1].startswith("d") and f[2].startswith("L"):
            try:
                rows.append({
                    "corpus": cur, "dim": int(f[1][1:]), "level": int(f[2][1:]),
                    "shape_bits": int(f[3]), "out": int(f[4]), "ratio": float(f[5]),
                    "bits_byte": float(f[6]), "rmse": float(f[7]), "psnr": float(f[8]),
                    "degen": f[9] == "1",
                })
            except ValueError:
                pass
    return rows, sizes


def corpus_sigma(name_to_file, n=WORK):
    """sigma = RMS of the centered bytes; the only input statistic the distortion model needs"""
    import numpy as np

    out = {}
    for label, fn in name_to_file.items():
        p = os.path.join(CORPUS, fn)
        if not os.path.exists(p):
            continue
        b = np.fromfile(p, dtype=np.uint8, count=n).astype(np.float64) - 127.5
        if b.size:
            out[label] = float(np.sqrt(np.mean(b * b)))
    return out


CORPUS_FILES = {
    "urandom": "urandom.bin", "random": "random.bin", "image-raw": "photo.ppm",
    "image-png": "photo.png", "json": "data.json", "text": "text.txt",
    "source": "micron_src.txt", "records": "records.bin", "mixed": "mixed.bin",
    "runs": "runs.bin", "words": "words.bin",
}

SAT = math.sqrt(2.0)  # saturation: an uncorrelated reconstruction of matched energy


def predict_rmse(kappa, dim, d, sigma):
    return min(kappa * d * sigma, SAT * sigma)


def fit_kappa(rows, sigma):
    """least squares on the UNSATURATED cells only -- a saturated cell carries no rate information"""
    import numpy as np

    kap = {}
    for dim in sorted({r["dim"] for r in rows}):
        num = den = 0.0
        for r in rows:
            if r["dim"] != dim or r["degen"] or r["corpus"] not in sigma:
                continue
            s = sigma[r["corpus"]]
            if r["rmse"] >= 0.9 * SAT * s:  # saturated
                continue
            x = d_of(LEVEL_DQ[r["level"]]) * s
            num += x * r["rmse"]
            den += x * x
        kap[dim] = num / den if den else float("nan")
    return kap


def report_fit(rows, sigma, kap):
    """Residuals, split by VALIDITY DOMAIN.

    The two-branch law is accurate where a configuration is actually shippable, and it
    under-predicts as a cell approaches saturation -- coarse d at high dim, where the shape code
    carries under ~1 bit per dimension and the reconstruction is decorrelating.  That region is
    reported separately rather than fitted away: it is the region the degeneracy analysis tells you
    not to use, and smoothing the corner would only hide where the model stops being trustworthy.
    """
    import numpy as np

    # relative error is only meaningful above the bench's print resolution (2 decimals); below it
    # the ratio divides by rounding noise.  Those cells are scored on absolute error instead.
    RES_FLOOR, USABLE_PSNR = 1.0, 20.0

    print()
    print("  kappa(dim) -- the packing constant, fitted on unsaturated cells")
    print(f"    {'dim':>5}  {'kappa':>7}   {'0.29*sqrt(dim)':>15}    (the structural law: kappa ~ sqrt(dim))")
    for dim, k in sorted(kap.items()):
        print(f"    {dim:>5}  {k:>7.3f}   {0.29 * math.sqrt(dim):>15.3f}")

    for domain, keep, blurb in (
        ("USABLE (measured psnr >= 20 dB -- the shippable region)", lambda r: r["psnr"] >= USABLE_PSNR, ""),
        ("COLLAPSE (psnr < 20 dB -- code too sparse for the dimension; model under-predicts)",
         lambda r: r["psnr"] < USABLE_PSNR, ""),
    ):
        print()
        print(f"  {domain}{blurb}")
        print(f"    {'corpus':<12} {'sigma':>7} {'nrel':>5} {'R^2':>8} {'med|rel|':>9} {'max|rel|':>9}  worst cell")
        allp, alla = [], []
        for c in sorted(sigma):
            pr, ac, worst, wr = [], [], 0.0, "-"
            for r in rows:
                if r["corpus"] != c or r["degen"] or not keep(r):
                    continue
                p = predict_rmse(kap[r["dim"]], r["dim"], d_of(LEVEL_DQ[r["level"]]), sigma[c])
                pr.append(p)
                ac.append(r["rmse"])
                if r["rmse"] >= RES_FLOOR:
                    rel = abs(p - r["rmse"]) / r["rmse"]
                    if rel > worst:
                        worst, wr = rel, f"d{r['dim']} L{r['level']} pred {p:.2f} vs {r['rmse']:.2f}"
            if not pr:
                continue
            pr, ac = np.array(pr), np.array(ac)
            m = ac >= RES_FLOOR
            ss_res = float(np.sum((pr - ac) ** 2))
            ss_tot = float(np.sum((ac - ac.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rel = np.abs(pr[m] - ac[m]) / ac[m] if m.any() else np.array([0.0])
            mae = float(np.max(np.abs(pr[~m] - ac[~m]))) if (~m).any() else 0.0
            print(f"    {c:<12} {sigma[c]:>7.2f} {int(m.sum()):>5} {r2:>8.4f} "
                  f"{float(np.median(rel)):>9.3f} {worst:>9.3f}  {wr}"
                  f"{f'   (near-lossless: max abs err {mae:.2f})' if mae else ''}")
            allp += list(pr)
            alla += list(ac)
        if allp:
            pr, ac = np.array(allp), np.array(alla)
            m = ac >= RES_FLOOR
            ss_res = float(np.sum((pr - ac) ** 2))
            ss_tot = float(np.sum((ac - ac.mean()) ** 2))
            rel = np.abs(pr[m] - ac[m]) / ac[m] if m.any() else np.array([0.0])
            print(f"    {'ALL':<12} {'':>7} {int(m.sum()):>5} {1 - ss_res / ss_tot:>8.4f} "
                  f"{float(np.median(rel)):>9.3f} {float(np.max(rel)):>9.3f}")

    print()
    print("  Known structural deviations, reported rather than fitted away:")
    print("    * runs.bin over-predicts BADLY -- see its USABLE row above (median relative error")
    print("      ~0.87, max ~1.25 on the shipped corpus): a constant run centers to a vector along")
    print("      (1,1,...,1)/sqrt(n), which is eta = pi/4 exactly -- the DENSEST leaf at every")
    print("      recursion level. Flat data gets a better code than a generic block does.")
    print("    * the collapse rows under-predict: min() has a hard corner, reality rounds it, and")
    print("      the error reaches sqrt(2)*sigma well before kappa*d does. Do not ship those cells.")


def predict_best(sigma_v, target_psnr, kap, dims=(4, 8, 16, 32, 64), levels=range(1, 17)):
    """the inverse, and the point of the whole exercise: cheapest config meeting a PSNR target,
    WITHOUT running the codec -- one sigma from the input is the entire data dependence"""
    best = None
    for dim in dims:
        dl = int(math.log2(dim))
        for lvl in levels:
            r = rate("bin", dl, lvl)
            if r["degenerate"]:
                continue
            rm = predict_rmse(kap.get(dim, 0.29 * math.sqrt(dim)), dim, r["d"], sigma_v)
            psnr = 99.99 if rm < 1e-9 else 20.0 * math.log10(255.0 / rm)
            if psnr < target_psnr:
                continue
            if best is None or r["bits_per_input_byte"] < best[0]["bits_per_input_byte"]:
                best = (r, dim, lvl, psnr)
    return best


def main():
    args = sys.argv[1:]
    print("=== hsc compression-ratio model ===")
    print()
    print("RATE -- exact.  Python mirror of the cardinality recursion vs the in-tree goldens:")
    bad = check_goldens()
    print(f"  {len(GOLDENS) - bad}/{len(GOLDENS)} exact")
    if bad:
        print("  MIRROR IS WRONG -- fix it before reading anything below", file=sys.stderr)
        return 2

    rows, sizes = bench_rows(args[1] if len(args) > 1 and os.path.exists(args[1]) else None)
    if rows:
        print()
        print("  predicted vs measured STREAM SIZE over every measured cell (must be exact):")
        mism = 0
        for r in rows:
            n = sizes.get(r["corpus"], 0)
            if not n:
                continue
            want = encoded_size(n, "bin", int(math.log2(r["dim"])), r["level"])
            if want != r["out"]:
                mism += 1
                print(f"    ** d{r['dim']} L{r['level']} {r['corpus']}: model {want} vs stream {r['out']}")
        print(f"    {len(rows) - mism}/{len(rows)} cells exact"
              f"{'' if not mism else '   ** THE RATE MODEL IS WRONG **'}")
        if mism and "--validate" in args:
            return 3

    if "--validate" in args:
        print()
        print("validate: rate model exact.")
        return 0

    if not rows:
        print("\n(no bench output -- run ./bin/compression_ratio_bench for the distortion half)")
        return 0

    sigma = corpus_sigma(CORPUS_FILES)
    if not sigma:
        print("\n(no corpus/ -- run scripts/corpus.py fetch for the distortion half)")
        return 0

    print()
    print("DISTORTION -- empirical.  rmse ~= min(kappa(dim) * d * sigma, sqrt(2) * sigma)")
    print("  sigma = RMS of the centered bytes; the second branch is saturation, where the code is")
    print("  too sparse to carry the direction and the reconstruction decorrelates entirely.")
    kap = fit_kappa(rows, sigma)
    report_fit(rows, sigma, kap)

    if len(args) >= 2 and args[0] == "--predict":
        c, t = args[1], float(args[2]) if len(args) > 2 else 30.0
        b = predict_best(sigma[c], t, kap)
        print()
        print(f"  predict {c} @ {t} dB -> ", end="")
        print(f"d{b[1]} L{b[2]}  ratio {b[0]['ratio']:.2f}x  "
              f"{b[0]['bits_per_input_byte']:.2f} bits/B  psnr {b[3]:.1f}" if b else "nothing in range")
        return 0

    print()
    print("  INVERSE: cheapest config for a PSNR target, predicted from sigma alone (no codec run),")
    print("  against the search the bench ran over real streams:")
    print(f"    {'corpus':<12} {'target':>7}  {'predicted':<12} {'ratio':>7}   {'measured best':<12} {'ratio':>7}")
    for c in sorted(sigma):
        for t in (20.0, 30.0, 40.0):
            b = predict_best(sigma[c], t, kap)
            cand = [r for r in rows if r["corpus"] == c and not r["degen"] and r["psnr"] >= t]
            meas = min(cand, key=lambda r: r["bits_byte"]) if cand else None
            ps = f"d{b[1]} L{b[2]}" if b else "none"
            pr = f"{b[0]['ratio']:.2f}x" if b else "-"
            ms = f"d{meas['dim']} L{meas['level']}" if meas else "none"
            mr = f"{meas['ratio']:.2f}x" if meas else "-"
            print(f"    {c:<12} {int(t):>4} dB  {ps:<12} {pr:>7}   {ms:<12} {mr:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
