#!/usr/bin/env python3
"""Build the hsc benchmark corpus.

hsc has never been measured on real data -- every workload in benches/_corpus.hpp is a pinned
xorshift generator.  This script assembles a corpus of real files under corpus/ (gitignored) for
benches/compression_ratio_bench.cpp, and exports open-weight model tensors as raw f32 for
benches/vector_bench.cpp.

Nothing here is checked in and nothing is required: every bench prints `skip` for a file it
cannot open, so a bare checkout still builds and runs.

    python3 scripts/corpus.py fetch     # the byte corpora
    python3 scripts/corpus.py models    # open-weight tensors -> corpus/models/*.f32
    python3 scripts/corpus.py list      # what is present, with sizes

Dependencies: numpy and pillow (present); requests optional (urllib fallback).  Deliberately NOT
huggingface_hub or torch -- safetensors is an 8-byte length, a JSON header and a raw buffer, which
is 30 lines of stdlib, and the embedding matrix is just a matrix of vectors.
"""

import json
import os
import random
import shutil
import struct
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
MODELS = os.path.join(CORPUS, "models")

CAP = 4 << 20  # 4 MiB per corpus: enough for stable statistics, small enough for static buffers

UA = {"User-Agent": "hsc-bench/1.0 (compression benchmark)"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def write(name, data, note=""):
    os.makedirs(CORPUS, exist_ok=True)
    p = os.path.join(CORPUS, name)
    with open(p, "wb") as f:
        f.write(data)
    log(f"  {name:<18} {len(data):>10,} B  {note}")
    return p


def fetch_url(url, cap=None):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read(cap) if cap else r.read()


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# byte corpora


def c_random():
    # /dev/random and /dev/urandom are the SAME CSPRNG on Linux >= 5.6 -- /dev/random stopped
    # blocking on an entropy estimate.  Both rows are here because both were asked for; expect
    # them to land on identical numbers, and that agreement is itself the check.
    with open("/dev/urandom", "rb") as f:
        write("urandom.bin", f.read(CAP), "incompressible worst case")
    with open("/dev/random", "rb") as f:
        write("random.bin", f.read(CAP), "same CSPRNG as urandom since Linux 5.6")


def c_image():
    # Kodak kodim23 -- the standard image-compression test corpus.
    png = fetch_url("http://r0k.us/graphics/kodak/kodak/kodim23.png")
    write("photo.png", png, "already entropy-coded: expect noise-like behaviour")
    try:
        import io

        from PIL import Image

        im = Image.open(io.BytesIO(png)).convert("RGB")
        raw = im.tobytes()
        write("photo.ppm", raw[:CAP], f"raw RGB {im.size[0]}x{im.size[1]}: where lossy makes sense")
    except Exception as e:  # pragma: no cover
        log(f"  photo.ppm          SKIPPED ({e})")


def c_json():
    src = first_existing("/code/C++/cjson/sample/large-file.json")
    if src:
        with open(src, "rb") as f:
            write("data.json", f.read(CAP), f"prefix of {src}")
    else:
        log("  data.json          SKIPPED (no local JSON sample)")


def c_text():
    parts = []
    tdir = "/code/Git/brotli/tests/testdata"
    for n in ("alice29.txt", "asyoulik.txt", "lcet10.txt", "plrabn12.txt"):
        p = os.path.join(tdir, n)
        if os.path.exists(p):
            with open(p, "rb") as f:
                parts.append(f.read())
    have = sum(len(p) for p in parts)
    if have < CAP:
        try:
            parts.append(fetch_url("https://www.gutenberg.org/files/2600/2600-0.txt", CAP - have))
        except Exception as e:
            log(f"  (gutenberg top-up skipped: {e})")
    data = b"\n".join(parts)[:CAP]
    if data:
        write("text.txt", data, "Canterbury + Gutenberg, natural-language prose")


def c_source():
    # the ask was explicit: pull the micron corelib from GitHub and use it as the source corpus
    dst = os.path.join(CORPUS, "_micron.cpp")
    if not os.path.isdir(dst):
        log("  cloning rfgplk/micron.cpp (depth 1) ...")
        r = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", "https://github.com/rfgplk/micron.cpp", dst],
            capture_output=True,
        )
        if r.returncode != 0:
            log(f"  micron_src.txt     SKIPPED (clone failed: {r.stderr.decode()[:200]})")
            return
    blob = bytearray()
    for dirpath, dirnames, filenames in os.walk(os.path.join(dst, "src")):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.endswith((".hpp", ".cpp", ".h", ".c")):
                with open(os.path.join(dirpath, fn), "rb") as f:
                    blob += f.read()
                if len(blob) >= CAP:
                    break
        if len(blob) >= CAP:
            break
    write("micron_src.txt", bytes(blob[:CAP]), "C++ source: structured text, high local redundancy")


def c_structured():
    # fixed-schema binary records: the "structured data" case -- monotone timestamps, small ints,
    # correlated floats.  This is the shape hsc's block layer should like most.
    rnd = random.Random(0x9E3779B9)
    out = bytearray()
    ts = 1_700_000_000_000
    val = 20.0
    i = 0
    while len(out) < CAP:
        ts += rnd.randint(1, 40)
        val += rnd.gauss(0, 0.35)
        out += struct.pack("<IIqff", i, rnd.randint(0, 255), ts, val, val * 0.5 + 1.0)
        i += 1
    write("records.bin", bytes(out[:CAP]), "fixed-schema records: id/enum/timestamp/f32/f32")


def c_unstructured():
    # INTERLEAVED, not concatenated: a concatenation's first megabyte is just its first member, so
    # any bench working on a prefix would measure that member and call it "unstructured".
    chunk = 64 << 10
    srcs = []
    for n in ("micron_src.txt", "photo.png", "data.json", "records.bin", "text.txt"):
        p = os.path.join(CORPUS, n)
        if os.path.exists(p):
            srcs.append(open(p, "rb"))
    if not srcs:
        return
    blob = bytearray()
    live = True
    while live and len(blob) < CAP:
        live = False
        for f in srcs:
            b = f.read(chunk)
            if b:
                blob += b
                live = True
            if len(blob) >= CAP:
                break
    for f in srcs:
        f.close()
    write("mixed.bin", bytes(blob[:CAP]), "heterogeneous, 64 KiB interleave: source/image/json/records/prose")


def c_synth():
    rnd = random.Random(0xC0FFEE)
    out = bytearray()
    while len(out) < CAP:
        out += bytes([rnd.randint(0, 255)]) * min(rnd.randint(1, 512), CAP - len(out))
    write("runs.bin", bytes(out[:CAP]), "long constant runs: geometric lengths")

    words = first_existing("/usr/share/dict/linux.words", "/usr/share/dict/words")
    if not words:
        log("  words.bin          SKIPPED (no system dictionary)")
        return
    with open(words, "rb") as f:
        vocab = [w for w in f.read().split(b"\n") if 2 <= len(w) <= 14][:60000]
    # Zipf-weighted draws: real text is not uniform over its vocabulary
    n = len(vocab)
    weights = [1.0 / (i + 1) for i in range(n)]
    out = bytearray()
    while len(out) < CAP:
        out += b" ".join(rnd.choices(vocab, weights=weights, k=2000)) + b" "
    write("words.bin", bytes(out[:CAP]), f"Zipf-weighted words from {os.path.basename(words)}")


def cmd_fetch():
    os.makedirs(CORPUS, exist_ok=True)
    log(f"corpus -> {CORPUS}  (cap {CAP:,} B per file)")
    for fn in (c_random, c_image, c_json, c_text, c_source, c_structured, c_synth, c_unstructured):
        try:
            fn()
        except Exception as e:
            log(f"  !! {fn.__name__}: {e}")
    log("done.")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# open-weight model tensors

# small, permissively licensed, and downloadable without an auth token.
MODEL_REPOS = [
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),   # 90 MB f32, embedding-focused
    ("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M"),                   # 269 MB, a real decoder LM
]


def safetensors_read(path):
    """Parse safetensors with the stdlib: u64 LE header length, JSON header, then the buffer."""
    import numpy as np

    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        buf = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt, shape, (a, b) = meta["dtype"], meta["shape"], meta["data_offsets"]
        raw = buf[base + a : base + b]
        if dt == "F32":
            arr = raw.view(np.float32)
        elif dt == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif dt == "BF16":
            # numpy has no bfloat16: bf16 IS the top 16 bits of an f32, so widen and reinterpret
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            continue
        out[name] = arr.reshape(shape) if shape else arr
    return out


def download(url, path):
    if os.path.exists(path):
        log(f"  have {os.path.basename(path)} ({os.path.getsize(path):,} B)")
        return path
    log(f"  downloading {url} ...")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(path + ".part", path)
    log(f"  got {os.path.basename(path)} ({os.path.getsize(path):,} B)")
    return path


def pick_tensors(name, tensors):
    """One embedding matrix plus a spread of weight matrices -- not the whole model.

    Rank-1/rank-0 tensors (biases, layernorm scales) are excluded: they are a negligible share of
    the parameters and they are not vectors in any useful sense.
    """
    import numpy as np

    keep = []
    emb = [k for k in tensors if tensors[k].ndim == 2 and ("embed" in k.lower() or "word_embeddings" in k.lower())]
    if emb:
        k = max(emb, key=lambda k: tensors[k].size)
        keep.append(("embed", k))
    mats = sorted(
        (k for k in tensors if tensors[k].ndim == 2 and k not in dict(keep).values()),
        key=lambda k: -tensors[k].size,
    )
    for k in mats[:4]:
        keep.append(("weight", k))
    return keep


def cmd_models():
    import numpy as np

    os.makedirs(MODELS, exist_ok=True)
    manifest = []
    for short, repo in MODEL_REPOS:
        url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
        st = os.path.join(MODELS, f"{short}.safetensors")
        try:
            download(url, st)
        except Exception as e:
            log(f"  !! {short}: {e}")
            continue
        tensors = safetensors_read(st)
        log(f"  {short}: {len(tensors)} tensors")
        for kind, key in pick_tensors(short, tensors):
            arr = np.ascontiguousarray(tensors[key], dtype=np.float32)
            base = f"{short}.{key.replace('/', '_').replace('.', '_')}"
            fp = os.path.join(MODELS, base + ".f32")
            arr.tofile(fp)
            rec = {
                "model": short,
                "tensor": key,
                "kind": kind,
                "file": os.path.basename(fp),
                "rows": int(arr.shape[0]),
                "cols": int(arr.shape[1]),
                "elems": int(arr.size),
                "bytes": int(arr.nbytes),
                "rms": float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))),
            }
            manifest.append(rec)
            log(f"    {kind:<7} {key:<44} {arr.shape} -> {os.path.basename(fp)} ({arr.nbytes:,} B)")
    with open(os.path.join(MODELS, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"wrote {os.path.join(MODELS, 'manifest.json')} ({len(manifest)} tensors)")


def cmd_list():
    if not os.path.isdir(CORPUS):
        log("no corpus/ -- run: python3 scripts/corpus.py fetch")
        return
    for d in (CORPUS, MODELS):
        if not os.path.isdir(d):
            continue
        log(f"{d}:")
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                log(f"  {fn:<44} {os.path.getsize(p):>13,} B")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    {"fetch": cmd_fetch, "models": cmd_models, "list": cmd_list}.get(cmd, cmd_fetch)()
