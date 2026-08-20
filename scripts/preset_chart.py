#!/usr/bin/env python3
"""The one chart: hsc's stock presets against the coders they would actually replace.

`src/hsc/quant.hpp` declares a seven-entry preset ladder -- q_min .. q_finest -- and those are the
library's own default answers to "how should I store this?".  Nothing in the report measured them.
This script does, and it puts them in the same table as the alternatives, in two blocks:

  block 1  f32 model tensors.  The presets' own lane: they are declared in bits/WEIGHT, and the
           things a weight tensor actually gets stored with are fp16/int8/int4/int3/int2 and, if
           someone tries, a general-purpose lossless coder.
  block 2  the 512 KiB byte corpus.  The same (dim, level) cells driven through mode::bin, against
           the four lossless coders on the same prefixes.

THE COMPARISON IS NOT A CONTEST, and the two blocks are dishonest if read as one. zstd/xz/bzip2/gzip
are LOSSLESS: they reproduce the input exactly and their ratio is whatever the data allows -- 1.00x
on /dev/urandom, 104x on long runs, and unknowable before you run them. int-N and hsc are LOSSY.
What hsc adds over int-N is not "better compression", it is a fixed rate with a lower error at the
same rate; what it adds over an entropy coder is a ratio that is a constant of the configuration,
guaranteed on any input at all.  Every table below carries a `lossy` column for exactly that reason.

RATE CONVENTION.  hsc rows print the PAYLOAD rate -- block records only -- which is the number the
preset constants in quant.hpp state and which tests/quant.cpp pins against hsc::rate().  The frame
is a flat 48 bytes per stream (40 header + 8 trailer) regardless of length, i.e. below the displayed
precision at these sizes; `qplan::bits_per_weight` is the framed number if you need it.  Coder rows
count every byte the coder emits, framing included.

    python3 scripts/preset_chart.py
    python3 scripts/preset_chart.py --bench-out saved.txt     # reuse a captured bench run
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compression_ratio_model as crm      # rate mirror + the bench-output parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
MODELS = os.path.join(CORPUS, "models")
VBENCH = os.path.join(ROOT, "bin", "vector_bench")

# the four coders ratio_report.py already runs on the byte corpus, same flags
CODERS = [
    ("zstd -19", ["zstd", "-19", "-q", "-c"]),
    ("xz -9", ["xz", "-9", "-c"]),
    ("bzip2 -9", ["bzip2", "-9", "-c"]),
    ("gzip -9", ["gzip", "-9", "-c"]),
]

# src/hsc/quant.hpp:139-146.  bits_per_weight here is API -- tests/quant.cpp:71-87 pins every one of
# these literals bit-for-bit against hsc::rate(), so a mismatch below is a bug, not a rounding.
PRESETS = [
    # name        dim_log2 level gain  bits/weight
    ("q_min", 6, 5, 8, 1.03125),
    ("q_tiny", 6, 6, 8, 1.4375),
    ("q_small", 5, 6, 8, 2.03125),
    ("q_compact", 4, 6, 8, 2.8125),
    ("q_balanced", 5, 7, 8, 3.0625),
    ("q_fine", 3, 6, 8, 3.75),
    ("q_finest", 2, 6, 8, 5.0),
]

# block 1's tensor, and the row cap that keeps a regeneration quick.  Named in the caption: an
# aggregate over tensors of different scale would average RMSEs that are not commensurable.
TENSOR = "all-MiniLM-L6-v2.embeddings_word_embeddings_weight.f32"
TENSOR_LABEL = "all-MiniLM-L6-v2 embeddings.word_embeddings.weight"
TENSOR_COLS = 384
TENSOR_ROWS = 4096

# block 2's corpus for the distortion columns.  Named, never averaged across corpora.
BYTE_CORPUS = "mixed"

BAR_W = 22


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# shared


def bar(ratio, lo=1.0, hi=64.0):
    """log-scale ratio bar; this is the part that makes the table a chart"""
    if not ratio or ratio <= 0:
        return ""
    f = (math.log(max(ratio, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    k = max(1, min(BAR_W, int(round(f * BAR_W))))
    return "#" * k


def coder_bytes(data, cmd):
    """compressed length, or None when the binary is absent or the run fails"""
    if shutil.which(cmd[0]) is None:
        return None
    r = subprocess.run(cmd, input=data, capture_output=True)
    return len(r.stdout) if r.returncode == 0 and r.stdout else None


def psnr_peak(ref, rmse):
    """20 log10(peak / rmse) -- the definition hsc::measure uses (quant.hpp:1064).  NOT the 8-bit
    full-scale psnr the byte-corpus tables use; the two are not comparable, hence two blocks."""
    import numpy as np

    peak = float(np.abs(ref).max())
    if rmse <= 0 or peak <= 0:
        return float("inf")
    return 20.0 * math.log10(peak / rmse)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# block 1 -- f32 model tensors


def block_tensors(out):
    # numpy and vector_eval are pulled in here, not at module scope, so a report run without them
    # degrades to a skip line instead of failing the whole document
    try:
        import numpy as np

        import vector_eval as ve
    except ImportError as e:
        out.append(f"  skip: {e}")
        return

    src = os.path.join(MODELS, TENSOR)
    if not os.path.exists(src):
        out.append(f"  skip: no {src} -- run: python3 scripts/corpus.py models")
        return
    if not os.path.exists(VBENCH):
        out.append(f"  skip: no {VBENCH} -- run: duck batch build.duck")
        return

    n = TENSOR_ROWS * TENSOR_COLS
    x = np.fromfile(src, dtype=np.float32, count=n).reshape(TENSOR_ROWS, TENSOR_COLS)
    raw = x.tobytes()
    elems = float(x.size)

    rows = []

    for label, cmd in CODERS:
        z = coder_bytes(raw, cmd)
        if z is None:
            rows.append((label, "no", None, None, 0.0, float("inf")))
            continue
        bpw = 8.0 * z / elems
        rows.append((label, "no", bpw, 32.0 / bpw, 0.0, float("inf")))

    for bits in (16, 8, 4, 3, 2):
        # fp16 is a real half round-trip, not int16 -- same convention as vector_eval.py:161
        q = ve.intn(x, bits) if bits < 16 else x.astype(np.float16).astype(np.float32)
        rr = ve.rel_rmse(x, q)
        rmse = float(np.sqrt(np.mean((x.astype(np.float64) - q.astype(np.float64)) ** 2)))
        name = "fp16" if bits == 16 else f"int{bits}"
        rows.append((name, "yes", float(bits), 32.0 / bits, rr, psnr_peak(x, rmse)))

    bad = []
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "t.f32")
        x.tofile(tmp)
        for name, dl, lvl, _gb, want in PRESETS:
            r = ve.run_filter(tmp, "vec", 1 << dl, lvl)
            if r is None:
                rows.append((name, "yes", want, 32.0 / want, None, None))
                continue
            dec, ref, stream = r
            base = ref if ref is not None else x.reshape(-1)[: dec.size]
            bpw = (stream - 48) * 8.0 / dec.size
            rr = ve.rel_rmse(base, dec)
            rmse = float(np.sqrt(np.mean((base.astype(np.float64) - dec.astype(np.float64)) ** 2)))
            if abs(bpw - want) > 5e-4:
                bad.append(f"  !! PRESET RATE MISMATCH {name}: measured {bpw:.5f} vs quant.hpp {want}")
            rows.append((name, "yes", bpw, 32.0 / bpw, rr, psnr_peak(base, rmse)))

    out.append(f"[block 1] f32 model tensor -- {TENSOR_LABEL}")
    out.append(f"          first {TENSOR_ROWS} rows x {TENSOR_COLS} cols = {int(elems):,} weights, "
               f"{len(raw):,} B as fp32")
    out.append("          psnr is 20*log10(peak/rmse) against the tensor's own peak magnitude")
    out.append("          int-N rows charge exactly N bits/weight (usual convention; the per-row f16")
    out.append("          scale, 16/cols, is not billed). vector_eval samples 4096 ROWS AT RANDOM from")
    out.append("          this tensor while this chart takes the FIRST 4096, so int-N cells differ.")
    out.append("")
    out.append("  method       lossy   bits/w   vs fp32  rel rmse   psnr dB   ratio (log scale)")
    out.append("  " + "-" * 76)
    for name, lossy, bpw, ratio, rr, ps in rows:
        b = "     -  " if bpw is None else f"{bpw:8.3f}"
        v = "      - " if ratio is None else f"{ratio:7.2f}x"
        e = "       - " if rr is None else ("  lossless" if rr == 0.0 else f"{rr:9.4f}")
        p = "        -" if ps is None else ("      inf" if math.isinf(ps) else f"{ps:9.2f}")
        out.append(f"  {name:<12} {lossy:<5} {b} {v} {e} {p}   {bar(ratio or 0)}")
    out.extend(bad)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# block 2 -- the 512 KiB byte corpus


def block_bytes(out, bench_out=None):
    rows_all, sizes = crm.bench_rows(bench_out)
    if not rows_all:
        out.append("  skip: no bin/compression_ratio_bench output -- run: duck batch build.duck")
        return

    # the lossless coders over every corpus prefix: one named row for the distortion columns, and
    # the min..max swing over all 11, which is the whole point of the block
    swing, named = {}, {}
    for label, fn in crm.CORPUS_FILES.items():
        p = os.path.join(CORPUS, fn)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            data = f.read(crm.WORK)
        for cname, cmd in CODERS:
            z = coder_bytes(data, cmd)
            if z is None:
                continue
            swing.setdefault(cname, []).append(len(data) / z)
            if label == BYTE_CORPUS:
                named[cname] = (8.0 * z / len(data), len(data) / z)

    n = sizes.get(BYTE_CORPUS)
    if n is None:
        out.append(f"  skip: corpus '{BYTE_CORPUS}' not in the bench output")
        return

    out.append(f"[block 2] 512 KiB byte corpus -- distortion columns from `{BYTE_CORPUS}` "
               f"(unstructured heterogeneous, n={n:,} B)")
    out.append("          hsc rows are the SAME (dim, level) cells as the presets above, driven")
    out.append("          through mode::bin; the presets' declared lane is block 1.")
    out.append("          psnr is 20*log10(255/rmse) over 8-bit full scale.")
    out.append("")
    out.append("  method               lossy  bits/B    ratio      rmse   psnr dB   ratio over 11 corpora   ratio (log scale)")
    out.append("  " + "-" * 112)

    for cname, _ in CODERS:
        if cname not in named:
            out.append(f"  {cname:<20} {'no':<5}       -        -         -         -   {'(coder missing)':>21}")
            continue
        bpb, ratio = named[cname]
        s = swing[cname]
        rng = f"{min(s):.2f}x .. {max(s):.2f}x"
        out.append(f"  {cname:<20} {'no':<5} {bpb:7.3f} {ratio:7.2f}x  lossless       inf   {rng:>21}   {bar(ratio)}")

    for name, dl, lvl, _gb, _want in PRESETS:
        dim = 1 << dl
        hit = [r for r in rows_all if r["corpus"] == BYTE_CORPUS and r["dim"] == dim and r["level"] == lvl]
        label = f"{name} (d{dim} L{lvl})"
        if not hit:
            out.append(f"  {label:<20} {'yes':<5}       -        -         -         -   {'(cell not in grid)':>21}")
            continue
        r = hit[0]
        tag = "   DEGENERATE -- not compression" if r["degen"] else ""
        # the bench prints bits/B with 2 decimals; printing that parse with 3 would manufacture a
        # false digit. The payload rate is data-independent, so state it exactly from the model.
        pay = crm.rate(dim_log2=dl, level=lvl)["bits_per_input_byte"]
        out.append(f"  {label:<20} {'yes':<5} {pay:7.3f} {r['ratio']:7.2f}x {r['rmse']:9.2f} "
                   f"{r['psnr']:9.2f}   {'(fixed, any input)':>21}   {bar(r['ratio'])}{tag}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def report(bench_out=None):
    out = []
    out.append("=== the stock presets against the coders they would replace ===")
    out.append("")
    out.append("zstd/xz/bzip2/gzip are LOSSLESS and reproduce the input exactly; their ratio is")
    out.append("whatever the data allows and is unknown until they have run.  int-N and hsc are")
    out.append("LOSSY.  hsc's ratio is a constant of the configuration, guaranteed on ANY input.")
    out.append("Read the `lossy` column before reading the ratio column.")
    out.append("")
    out.append("hsc rows print the PAYLOAD rate, which is what the quant.hpp constants state; the")
    out.append("frame is a flat 48 B per stream, below the precision shown here.")
    out.append("")
    block_tensors(out)
    out.append("")
    block_bytes(out, bench_out)
    return "\n".join(out)


def main():
    bench_out = None
    if "--bench-out" in sys.argv:
        bench_out = sys.argv[sys.argv.index("--bench-out") + 1]
    print(report(bench_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
