#!/usr/bin/env python3
"""Score hsc on real open-weight vectors: per-weight rate, cosine, and nearest-neighbour recall.

benches/vector_bench.cpp does the codec work (and, in --in/--out filter mode, hands back the
decoded f32); everything statistical lives here, in numpy, where it belongs.

What this measures, and why each choice:

  * bits/weight from the actual stream size, against the int-N a quantizer would charge for the
    same storage.  hsc's rate is FIXED -- it depends on (dim, level) alone -- so a row is only
    fairly compared against the int-N with the SAME bits/weight, never against int8 by default.

  * cosine similarity per row and recall@k over exact top-k neighbour lists.  For an embedding
    table, recall is the metric that matters: nobody cares about the RMSE of a retrieval index,
    they care whether the same tokens come back.

  * LANE CHOICE IS SUBSTANTIVE.  unit mode drops the gain field, so a block of `dim` elements is
    stored as a pure direction.  A 384- or 576-wide embedding row spans MANY blocks, and unit mode
    keeps no per-block magnitude, so the row's internal shape is destroyed however fine d gets.
    vec mode keeps a per-block gain and is the correct lane for rows wider than dim.  Both are run
    below so the numbers show it rather than the docs asserting it.

  * THE QUOTIENT FAMILY (quotient, quat, oct) IS EXCLUDED.  Those modes quotient out a fiber
    symmetry on complex/quaternion/octonion pairs; embeddings have no such symmetry, so they
    would discard 1/3/7 real dimensions per pair -- signal, not redundancy.

    python3 scripts/vector_eval.py            # the full report
    python3 scripts/vector_eval.py --rows 8192
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "corpus", "models")
BENCH = os.path.join(ROOT, "bin", "vector_bench")

# (dim, level) spread chosen to bracket int2..int4 rates in both lanes
CFGS = [(8, 5), (8, 6), (8, 7), (16, 6), (16, 7), (32, 6), (32, 7), (64, 7)]
KS = (1, 10, 100)


def run_filter(src, mode, dim, level, normalize=False):
    """round-trip src through hsc; returns (decoded, reference, stream_bytes) or None"""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.f32")
        cmd = [BENCH, "--in", src, "--out", out, "--mode", mode,
               "--dim", str(int(np.log2(dim))), "--level", str(level)]
        if normalize:
            cmd.append("--normalize")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        stream = 0
        for tok in r.stdout.split():
            if tok.startswith("stream="):
                stream = int(tok.split("=")[1])
        dec = np.fromfile(out, dtype=np.float32)
        ref = np.fromfile(out + ".ref", dtype=np.float32) if os.path.exists(out + ".ref") else None
        return dec, ref, stream


def intn(x, bits):
    """symmetric per-row int-N quantization: the baseline any weight-quantizer would ship"""
    if bits >= 32:
        return x.copy()
    lim = (1 << (bits - 1)) - 1
    s = np.abs(x).max(axis=1, keepdims=True)
    s[s == 0] = 1.0
    q = np.rint(x / s * lim).clip(-lim, lim)
    return (q / lim * s).astype(np.float32)


def rowwise_cos(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    c = np.zeros(len(a))
    c[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
    return c


def topk(mat, k):
    """exact top-k cosine neighbours of every row (self excluded), ordered nearest-first.

    argpartition alone guarantees only MEMBERSHIP in the top k, in arbitrary order; slicing
    its first j entries is not top-j. The argsort over the partitioned slice restores the
    order so recall(base[:, :j], t[:, :j], j) is a true recall@j."""
    u = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-30)
    s = u @ u.T
    np.fill_diagonal(s, -2.0)
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    row = np.arange(s.shape[0])[:, None]
    return idx[row, np.argsort(-s[row, idx], axis=1)]


def recall(base, test, k):
    hit = 0
    for i in range(base.shape[0]):
        hit += len(set(base[i].tolist()) & set(test[i].tolist()))
    return hit / (base.shape[0] * k)


def block_normalize(a, dim):
    """what unit mode does to the data before it codes anything: L2-normalize every dim-element
    chunk of the flat stream.  Isolating it is the only way to see the lane mismatch, because a
    unit-mode row scored against its own normalized reference cannot show it."""
    flat = a.reshape(-1).astype(np.float64).copy()
    n = (flat.size // dim) * dim
    v = flat[:n].reshape(-1, dim)
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    v /= nrm
    return flat[:n].reshape(a.shape[0], -1)[:, : a.shape[1]].astype(np.float32) if n == flat.size else \
        flat[:n].astype(np.float32).reshape(-1, a.shape[1])


def rel_rmse(a, b):
    d = float(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    n = float(np.sum(a.astype(np.float64) ** 2))
    return (d / n) ** 0.5 if n > 0 else 0.0


def main():
    rows_cap = 4096
    if "--rows" in sys.argv:
        rows_cap = int(sys.argv[sys.argv.index("--rows") + 1])
    mpath = os.path.join(MODELS, "manifest.json")
    if not os.path.exists(mpath):
        print("no corpus/models/manifest.json -- run: python3 scripts/corpus.py models", file=sys.stderr)
        return 1
    if not os.path.exists(BENCH):
        print(f"no {BENCH} -- run: duck batch build.duck", file=sys.stderr)
        return 1
    man = json.load(open(mpath))

    print("=== hsc on open-weight model vectors ===")
    print()
    print("Rate is FIXED: bits/weight depends on (dim, level) alone, not on the weights. Compare a")
    print("row against the int-N of the SAME bits/w. fp32 = 32 b/w, fp16 = 16, int8 = 8, int4 = 4.")
    print("int-N rows charge exactly N bits/weight (the usual convention) and exclude the per-row")
    print("f16 scale, 16/cols (~0.04 b/w at 384 cols).")
    print(f"Embedding tables are sampled to {rows_cap} rows (uniform random, seed 0xC0FFEE) so")
    print("top-k neighbour lists stay exact; the preset chart uses the FIRST 4096 rows of the same")
    print("tensor, so its int-N cells differ slightly from the ones here.")

    for rec in man:
        arr = np.fromfile(os.path.join(MODELS, rec["file"]), dtype=np.float32,
                          count=rec["rows"] * rec["cols"]).reshape(rec["rows"], rec["cols"])
        is_embed = rec["kind"] == "embed"
        if is_embed and arr.shape[0] > rows_cap:
            rs = np.random.default_rng(0xC0FFEE).choice(arr.shape[0], rows_cap, replace=False)
            arr = np.ascontiguousarray(arr[np.sort(rs)])
        # only the embedding tables get the recall treatment: a weight matrix's rows are not a
        # retrieval index, so recall@k there would be a number without a meaning
        base = topk(arr, max(KS)) if is_embed else None

        print()
        print(f"[{rec['model']}] {rec['tensor']}  {arr.shape}  ({rec['kind']}, rms {rec['rms']:.4f})")
        hdr = f"  {'method':<16} {'bits/w':>7} {'vs fp32':>8} {'rel rmse':>9} {'cos mean':>9} {'cos p01':>8}"
        if is_embed:
            hdr += "".join(f" {'r@'+str(k)+'e2e':>7}" for k in KS)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for bits in (16, 8, 4, 3, 2):
            q = intn(arr, bits) if bits < 16 else arr.astype(np.float16).astype(np.float32)
            c = rowwise_cos(arr, q)
            line = (f"  {('int' + str(bits)) if bits < 16 else 'fp16':<16} {float(bits):>7.2f} "
                    f"{32.0 / bits:>8.2f} {rel_rmse(arr, q):>9.4f} {c.mean():>9.4f} "
                    f"{np.quantile(c, 0.01):>8.4f}")
            if is_embed:
                t = topk(q, max(KS))
                line += "".join(f" {recall(base[:, :k], t[:, :k], k):>7.3f}" for k in KS)
            print(line)

        # what unit mode's block normalization costs BEFORE any coding: a wide row spans many
        # blocks and unit keeps no per-block magnitude, so this is pure, unrecoverable lane damage
        if is_embed:
            for dim in sorted({d for d, _ in CFGS}):
                nz = block_normalize(arr, dim)
                if nz.shape != arr.shape:
                    continue
                c = rowwise_cos(arr, nz)
                t = topk(nz, max(KS))
                line = (f"  {'(unit d' + str(dim) + ' prep)':<16} {0.0:>7.2f} {'-':>8} "
                        f"{rel_rmse(arr, nz):>9.4f} {c.mean():>9.4f} {np.quantile(c, 0.01):>8.4f}")
                line += "".join(f" {recall(base[:, :k], t[:, :k], k):>7.3f}" for k in KS)
                print(line)

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.f32")
            arr.tofile(src)
            for lane in ("vec", "unit"):
                for dim, lvl in CFGS:
                    got = run_filter(src, lane, dim, lvl)
                    if got is None:
                        continue
                    dec, ref, stream = got
                    if ref is None or dec.size != ref.size:
                        continue
                    cols = arr.shape[1]
                    nrow = dec.size // cols
                    if nrow == 0:
                        continue
                    d2 = dec[: nrow * cols].reshape(nrow, cols)
                    r2 = ref[: nrow * cols].reshape(nrow, cols)
                    bpw = (stream - 48) * 8.0 / dec.size
                    c = rowwise_cos(r2, d2)
                    line = (f"  {lane + ' d' + str(dim) + ' L' + str(lvl):<16} {bpw:>7.2f} "
                            f"{32.0 / bpw:>8.2f} {rel_rmse(r2, d2):>9.4f} {c.mean():>9.4f} "
                            f"{np.quantile(c, 0.01):>8.4f}")
                    if is_embed:
                        # recall is END-TO-END, against the ORIGINAL table's neighbours -- not
                        # against the encoder's input. That is the only version a user cares about,
                        # and it is what correctly charges unit mode for discarding block
                        # magnitudes: its rows can never beat their own `(unit dN prep)` ceiling.
                        t = topk(d2, max(KS))
                        line += "".join(f" {recall(base[:, :k], t[:, :k], k):>7.3f}" for k in KS)
                    print(line)

    print()
    print("Reading this table:")
    print("  * unit mode is only correct when the vector's dimension IS the block dimension. For a")
    print("    384/576-wide row it stores each dim-sized chunk as a pure direction and keeps no")
    print("    per-chunk magnitude, so the row's shape cannot be recovered at any d. Its rows are")
    print("    scored against the normalized input the encoder saw, which is why they can look fine")
    print("    on cos -- the `(unit dN prep)` rows show what that normalization costs on its own,")
    print("    at zero bits, and that cost is the floor no level can buy back. Their rel rmse can")
    print("    sit far above 1 on tables whose rows are not near unit norm (and below 1 on tables")
    print("    that are) because normalization rescales wholesale -- that is the lane's transform,")
    print("    not an error the codec made.")
    print("  * rel rmse and cos are measured against the input the ENCODER saw; r@k is measured")
    print("    end-to-end against the ORIGINAL table. Mixing those two baselines in one row is")
    print("    deliberate: the first isolates the codec, the second is the outcome a user gets.")
    print("  * vec mode keeps a per-block gain and is the lane to use for wide rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
