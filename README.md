# hsc

#### Hopf Spherical Compression (& Quantization)

... is a header-only C++23 fixed-rate lossy compression library built on the [micron](https://github.com/rfgplk/micron.cpp) corelib, inspired in part via *Constructive Spherical Codes by Hopf Foliations* (SCHF; Miyamoto, Costa, Sá Earp, IEEE Trans. Inf. Theory 2021).

The sphere S^(2n−1) foliates into leaves (S^(n−1) × S^(n−1))_η, recursively down to S³, which
foliates into flat tori carrying points on internal circles. A data block becomes Hopf
coordinates, then quantized and finally reduced to a single integer index.

Encoding and decoding are **(O(n log n))** with *no stored codebook*: both sides deterministically derive the same integer skeleton tables from the single fixed-point distance parameter stored in the stream header.

Records have constant width, making the payload randomly accessible by block number. The compression ratio is also guaranteed for every input, including **incompressible noise**, where entropy-based compressors must actually expand the data.

```cpp
#include <hsc/hsc.hpp>

micron::vector<u8> data = ...;
hsc::fhsc z = hsc::hopf(data, hsc::hopf_opts{ .level = 6, .dim_log2 = 3 });   // any bytes; cannot fail
auto back  = hsc::unhopf(z);                                                  // result<fhsc>

// typed float lanes
auto zv = hsc::hopf(my_f32_vector, hsc::hopf_opts{ .m = hsc::mode::vec });    // shape-gain f32 blocks
auto zu = hsc::hopf(embeddings,   hsc::hopf_opts{ .m = hsc::mode::unit });    // unit-norm: no gain field
auto zq = hsc::hopf(states,       hsc::hopf_opts{ .m = hsc::mode::quotient });// phase-invariant complex pairs
auto z4 = hsc::hopf(poses,        hsc::hopf_opts{ .m = hsc::mode::quat });    // gauge-invariant quaternion pairs
auto z8 = hsc::hopf(pairs16,      hsc::hopf_opts{ .m = hsc::mode::oct });     // octonion pairs (the last fibration)

// compile time, same wire bytes
static constexpr hsc::ct::str body{ "hopfed at compile time" };
constexpr auto stream = hsc::ct::hopf<body>();
constexpr auto plain  = hsc::ct::unhopf<stream>();
```

> [!WARNING]
> hsc has been tested exclusively on amd64 (AVX2+) for now.

> [!WARNING]
> hsc is currently Linux only, by nature of micron being Linux only. No other kernels or operating systems are supported. BSDs *may* work, but this is not guaranteed. 

## Installation

hsc depends only on the micron corelib, specifically micron must be reachable in source as `#include <micron/...>`. To build the in-tree tests, benches or examples, first run `cd include/ && sh fetch_micron.sh`, which fetches a local copy of micron to the `include/` directory — **this step is required**: every shipped build file (`*.duck` and `*.sh` alike) resolves micron at `include/micron` and compiles its `start/` runtime from there. For your own projects you can instead do `git clone --depth=1 --branch 1.9.11-lts https://github.com/rfgplk/micron.cpp.git` and install it to your /usr/include via its install scripts.

hsc by default uses `duck` as the build tool, which is optional and not at all required. The `*.sh` scripts are portable variants of the `*.duck` scripts with one caveat: they build without duck's default `-fstack-protector-all` and `-flto=8` (they pass `-fno-stack-protector`), which is the configuration the published perf numbers do *not* use — results from the `*.sh` builds are correct but not comparable to the perf tables.


## Generating BENCHMARKS.md

BENCHMARKS.md is already pregenerated and commited along with the rest of the source code. If you would like to regenerate it, first set up the working environment of the library, as described in Installation, and follow the following steps:

Build the reference corpus first:

`python3 scripts/corpus.py fetch && python3 scripts/corpus.py models`

Then generate the report:

`python3 scripts/ratio_report.py > BENCHMARKS.md`

## Mathematics

hsc is a **vector quantizer with no codebook**. Everything stream-defining is derived, on both
sides, from one integer: `d_q = round(d · 2^24)`, the fixed-point minimum codeword distance.

### Shape and gain

A block of `n = 2^dim_log2` values (n ∈ {4, 8, 16, 32, 64}) is split into a magnitude and a
direction:

```
g = ‖v‖₂                         x̂ = v / g  ∈ S^(n−1)
step = scale / (2^b − 1)         q  = clamp(round(g / step), 0, 2^b − 1)
v̂ = q · step · x̂                 q = 0 reconstructs exact zeros — and is reachable: any
                                 block with g < step/2 rounds to it (the gain cliff below)
```

`b` is `gain_bits` (default 8). For `bin`, bytes are centered on 127.5 first, so
`scale = 127.5·√n` is the exact maximum norm of such a block. `unit` and the quotient family
(`quotient`, `quat`, `oct`) carry no gain field at all. The direction `x̂` is what the sphere
layer codes, and it is where all the work is.

### Generalized Hopf foliations

Every `y ∈ S^(n−1)` decomposes along a half-dimension split as

```
y = (cos η · u,  sin η · v)      u, v ∈ S^(n/2−1),  η ∈ [0, π/2]
```

so the sphere foliates into leaves `(S^(n/2−1) × S^(n/2−1))_η` indexed by the energy-split angle
η. hsc discretizes η into a fan centered on the equal-energy split π/4:

```
Δη = 2·asin(d/2)        t = ⌊π / (4·asin(d/2))⌋        half = ⌊t/2⌋
η_i = π/4 + i·Δη        i ∈ [−half, +half]             leaves = 2·half + 1
```

Two codewords in the same leaf, differing only in their `u` factor, are separated by
`cos η · (their separation in S^(n/2−1))`. So to keep the whole code ≥ d apart, the child code
in the `u` factor must be built at distance `d / cos η`, and the one in `v` at `d / sin η`.
That is the recursion: **the same construction, one dimension down, at a leaf-dependent
distance.**

### Distance grid and why the tables stay finite

Left alone, the set of distinct `(dim, d′)` skeletons multiplies by ~t per split, on the order
of 10⁸ tables at dim 64.  hsc snaps every child distance **up** onto
a fixed geometric grid:

```
grid[g] = d_q( 2 · (16303/16384)^g )       g = 0 .. 2047,  spanning [7.8e-5, 2]
```

Snapping up is what preserves the guarantee: a child built at `d″ ≥ d/cos η` and then scaled by
`cos η` is still ≥ d apart. And it collapses the reachable set to at most one node per grid
member per level, which is what makes the memo finite and the skeleton buildable at all.
`16303/16384 = 16303/2^14` is dyadic, hence exactly representable, so the grid comes out
bit-identical at consteval and at runtime. **Changing these constants is a format break.**

### Cardinality

```
M(n, d) = Σ_i  M(n/2, snap(d/cos η_i)) · M(n/2, snap(d/sin η_i))
```

with the 4D base case below. Note that M is genuinely large (log₂ M ≈ 147.7 at dim 64 level 7, i.e. 148 shape bits) therefore it's carried in `micron::arbuint<1024>`; only `M mod 2^64` is kept in the skeleton nodes.

### 4D base case

S³ foliates into **flat tori** `T_η = S¹_{cos η} × S¹_{sin η}`, each carrying `n` internal
circles of `m` equidistant points, consecutive circles shifted by half a step. (This `n` is the
code's circle count, following the paper and `s3.hpp`; it is not the block dimension.)

```
m  = ⌊ π / asin( d / (2 cos η) ) ⌋      — m = 1 when d > 2 cos η
n₂ = ⌊ 2π / asin( d / (2 sin η) ) ⌋     — n = 1 outright when d > 2 sin η
arg = d²/(4 sin²η) − (cos²η / sin²η) · sin²(π / 2m)
n₁ = ⌊ π / asin(√arg) ⌋   when 0 < arg ≤ 1;  n₁ = 1 when √arg > 1;  else n₁ = n₂
n  = max( 2·⌊ min(n₁, n₂) / 2 ⌋ , 1 )   — forced even (so the half-step shift closes)
                                          except the single-circle floor n = 1
M(4, d) = Σ_x m_x · n_x
```

Note that at d = 2 the whole torus collapses to m = n = 1 and
M(4, 2) = 1; exactly what `tests/s3.cpp` pins at `dq_max`.

A codeword is

```
Δξ₁ = 2π/m      ξ₁ = j·Δξ₁ + k·Δξ₁/2
Δξ₂ = 2π/n      ξ₂ = k·Δξ₂
x = ( cos η cos ξ₁,  cos η sin ξ₁,  sin η cos ξ₂,  sin η sin ξ₂ )
```

### Quantizing

Encoding is a descent with no codebook:

```
η = atan2( √Σ_{c ≥ h} y_c² ,  √Σ_{c < h} y_c² )        h = n/2
i = clamp( round((η − π/4) / Δη), −half, +half )
                                        … then recurse into both halves
```

and decoding is the same walk in reverse, scaling the two halves by `cos η_x` and `sin η_x`.

### Index

Per-node fields collapse into one integer by mixed radix, using arbint prefix sums over the
leaf fan:

```
row_off[x] = Σ_{x′ < x} M(c₁(x′)) · M(c₂(x′))
a          = row_off[x] + a₂ · M(c₁) + a₁
```

Unranking inverts it with a binary search over the monotone prefix sums plus one `divmod`. Both
sides rebuild `row_off` and `M` from `d_q` alone, so the "codebook" is a pure function of the
header.

### Rate identity

```
block_elems = n        (the quotient family pins n: quotient 4, quat 8, oct 16)
shape_bits  = ⌈log₂ M⌉ = bit_length(M − 1)
record_bits = (bin|vec ? gain_bits : 0) + shape_bits
stream      = 40 + ⌈ nblocks · record_bits / 8 ⌉ + 8
bits/elem   = record_bits / block_elems
```

Closed form in `(mode, dim, d_q, gain_bits, n_elems)` and in **nothing about the data** — the
same for `/dev/urandom` and for JSON. `hsc::rate()` returns it without running the codec.

`M == 1` gives `shape_bits == 0`: the shape field has zero width, the record carries the gain
alone, and the block reconstructs to one fixed direction times a scalar. That is data loss
wearing a compression ratio, and `hsc::degenerate()` is the query for it. At d ≥ 1 every dim ≥ 8
collapses to it, and dim ≥ 16 is already there at level 1.

### Distortion via packing

SCHF guarantees that codewords are ≥ d apart. It says **nothing** about the distance from an
arbitrary input to the nearest codeword, and uneven energy splits clamp to the edge of the leaf
fan and stack down the recursion. So there is no derived distortion bound, only a **fitted** law:

```
rmse ≈ min( κ(n) · d · σ ,  √2 · σ )        σ = RMS of the centered input
κ(n) = 0.55, 0.81, 1.15, 1.70, 2.35  at n = 4, 8, 16, 32, 64   —  tracking 0.29·√n
```

`√2·σ` is the no-information level: error and signal uncorrelated. The fit holds to a pooled median relative error of 0.09 wherever a cell is shippable (PSNR ≥ 20 dB) — per-corpus medians run 0.05–0.14 on the ten non-flat corpora. On flat data it over-predicts badly (median ~0.87 on the `runs` corpus: a constant run centers onto `(1..1)/√n`, i.e. η = π/4 exactly — the densest leaf) and it under-predicts as a cell approaches saturation. Those domains are reported separately in BENCHMARKS.md rather than averaged into one flattering number.

### Compression by symmetry (quotient)

Read a 4-float block as `(z₀, z₁) = (y₀ + i y₁, y₂ + i y₃) ∈ S³`. If the data is insensitive to
a global phase `q ~ q·e^{iψ}`, the fiber coordinate is pure redundancy. The Hopf map

```
h(z₀, z₁) = ( 2 z₀ z̄₁ ,  |z₀|² − |z₁|² )        S³ → S² = S³/S¹
```

projects onto the quotient, and only the class is stored. `h(λq) = λ²·h(q)`, so dividing by
`‖q‖²` normalizes with no square root anywhere. S² gets a latitude-band code:

```
Δθ = 2·asin(d/2)     T = ⌊π/Δθ⌋ + 1     θ₀ = (π − (T−1)Δθ)/2
m_b = ⌊ π / asin( d / (2 sin θ_b) ) ⌋   (m_b = 1 when d > 2 sin θ_b)     M = Σ_b m_b
```

which is exactly `log₂(1/d)`-ish bits cheaper than the full S³ code at the same d — measured 2
bits/block at level 3 growing to 4 at level 7. Decoding returns the canonical section
`z₀ = √((1+p₂)/2)` real ≥ 0, `z₁ = (p₀ − i p₁)/(2 z₀)`.

### The full Hopf family (quat, oct)

The Hopf family of sphere-by-sphere fibrations has exactly four members; the three with
positive-dimensional fiber are the ones a codec can exploit, and hsc implements all of them:

`mode::quotient: S³ → S² = S³/S¹`
`mode::quat: S⁷ → S⁴ = S⁷/S³`
`mode::oct: S¹⁵ → S⁸ = S¹⁵/S⁷`

(The fourth is the real double cover S¹ → S¹ with fiber S⁰ — a zero-dimensional fiber saves no
bits.) The Hopf Invariant One Theorem (Adams' theorem: maps of Hopf invariant one exist only for
n ∈ {1, 2, 4, 8}) rules out any fibration of a sphere by spheres beyond these four.


All three use the same map shape:
```
h(x, y) = ( 2 x ȳ ,  |x|² − |y|² )         normalized by ‖(x,y)‖², scale-invariant
```

The base retains the projective point `[x : y]`; the fiber is the discarded symmetry. For quaternions, this is exact invariance under simultaneous right multiplication:

`(q₀, q₁) → (q₀g, q₁g)`

Left multiplication is not quotiented and rotates the base. Octonions have no corresponding group action because multiplication is non-associative; their fibers are the graph spheres
`{ (a, q·a) : |a| = 1 }`

where `q = y·x⁻¹`.

**Component convention:** Our codec uses vector-first, scalar-last components

`quaternion: (x, y, z, w)`,

`octonion: (e₁..e₇, e₀) with e₁,e₂,e₃ = i,j,k`, `e₄ = ℓ`,
`e₅,e₆,e₇ = iℓ,jℓ,kℓ` (Cayley–Dickson construction).

Pairs are adjacent: `f[0..n-1]`, `f[n..2n-1]`. 
This matches pose-data memory layouts; quotient's frozen `(re, im)` scalar-first reading does not
change.

The bases S⁴ ⊂ ℝ⁵ and S⁸ ⊂ ℝ⁹ get a **cap-anchored, equator-anchored suspension code**
(`sphere/susp.hpp`), different from the S² band layout above. The poles are
*exact codewords*  (`[x:0]` and `[0:y]` classes, spinor basis states) and the
interior band count is **odd** with the equator a band *center*.

Measured fiber savings vs `unit` at the same d (level 6): **quat 22 → 15 bits/block (7
saved), oct 37 → 24 (13 saved)**, growing to 10 and 20 at level 7. The quat index stays u64
end to end (M = 2^58.06 at the d = 1e-4 floor, measured); the oct index crosses u64 at level 11
(M ≈ 2^66.4 there; level 10, at 59 bits, is the last that fits) and rides the arbint pack lane.
Neither mode can go degenerate: even at d = 2 the two poles remain (M = 2).


### Pre-rotation

`.transform = true` (wire flags bit 2) applies

```
y = (1/√n) · H · D · x        H = Walsh–Hadamard,  D = pinned ±1 diagonal
```

`H = Hᵀ` and `H² = nI`, so `(1/√n)H` is orthogonal *and* involutive. Every basis vector maps to an exact 50/50
half-energy split at every recursion level: η = π/4, the densest leaf, making one-hot rows and axis spikes codable. It costs no rate at all, but smooth data prefers the raw
basis, so it is **off by default**, and it is excluded from the whole quotient family, whose
Hopf maps need the algebra structure a generic rotation breaks. The butterfly is adds and subs only (no fma); contractions on one side and not the other would break consteval/runtime bit-identity.

### Determinism

Both sides rebuild every skeleton from `d_q`, never from a raw f64. One ulp of drift in `asin`
would move a `⌊⌋` and change M, so every stream carries `skel_guard = M mod 2^64`: a drifted
decoder fails `bad_skeleton` instead of quietly emitting wrong floats. `tests/comptime.cpp`
proves consteval streams byte-identical to runtime streams, and `tests/exact.cpp` pins every
SIMD kernel against its scalar twin — that equality *is* the format contract.

## Rate and distortion, cell by cell

The rates are **exact and data-independent**: `hsc::rate()` returns it without
reading a byte or running the codec, and `scripts/compression_ratio_model.py` reproduces every
cell in independent Python. Distortions (`rmse`/`psnr`) below are measured on one
named corpus, `mixed` (unstructured heterogeneous, 512 KiB), in `bin` mode. Per-corpus grids for
all eleven corpora are in [BENCHMARKS.md](BENCHMARKS.md). PSNR is `20·log10(255/rmse)` over 8-bit
full scale.

`d` is the nominal minimum codeword distance of the level preset; the stream carries
`d_q = round(d·2^24)`. `bits/elem` is `record_bits / n` — bits per input byte in `bin` mode, bits
per weight in the f32 lanes — so the two ratio columns are the same number read against an 8-bit
and a 32-bit source. All rate columns here are **payload only** (record bits, no frame): a real
stream adds a flat 48 bytes (40-byte header + 8-byte trailer), which is why BENCHMARKS.md's
measured 512 KiB ratios read a hair lower (5.56x vs 5.57x at dim64 L6). Both ratio columns are
computed from the exact `record_bits / n`, not from the rounded `bits/w` shown.

| dim | L | d | shape bits | record bits | bits/w | vs bytes | vs fp32 | rmse | psnr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | .9 | 4 | 12 | 3.00 | 2.67x | 10.67x | 38.55 | 16.41 |
| 4 | 2 | .7 | 6 | 14 | 3.50 | 2.29x | 9.14x | 21.85 | 21.34 |
| 4 | 3 | .5 | 8 | 16 | 4.00 | 2.00x | 8.00x | 16.22 | 23.93 |
| 4 | 4 | .4 | 9 | 17 | 4.25 | 1.88x | 7.53x | 13.90 | 25.27 |
| 4 | 5 | .3 | 10 | 18 | 4.50 | 1.78x | 7.11x | 9.79 | 28.31 |
| 4 | 6 | .2 | 12 | 20 | 5.00 | 1.60x | 6.40x | 6.15 | 32.35 |
| 4 | 7 | .1 | 15 | 23 | 5.75 | 1.39x | 5.57x | 2.99 | 38.62 |
| 4 | 8 | .05 | 18 | 26 | 6.50 | 1.23x | 4.92x | 1.54 | 44.36 |
| 4 | 9 | .02 | 22 | 30 | 7.50 | 1.07x | 4.27x | 0.66 | 51.80 |
| 4 | 10 | .01 | 25 | 33 | 8.25 | 0.97x | 3.88x | 0.36 | 56.94 |
| 4 | 11 | .005 | 28 | 36 | 9.00 | 0.89x | 3.56x | 0.14 | 65.12 |
| 4 | 12 | .002 | 32 | 40 | 10.00 | 0.80x | 3.20x | 0.00 | 98.34 |
| 4 | 13 | .001 | 35 | 43 | 10.75 | 0.74x | 2.98x | 0.00 | 99.99 |
| 4 | 14 | 5e-4 | 38 | 46 | 11.50 | 0.70x | 2.78x | 0.00 | 99.99 |
| 8 | 1 | .9 | 6 | 14 | 1.75 | 4.57x | 18.29x | 48.57 | 14.40 |
| 8 | 2 | .7 | 9 | 17 | 2.13 | 3.76x | 15.06x | 38.28 | 16.47 |
| 8 | 3 | .5 | 12 | 20 | 2.50 | 3.20x | 12.80x | 28.36 | 19.08 |
| 8 | 4 | .4 | 14 | 22 | 2.75 | 2.91x | 11.64x | 20.89 | 21.73 |
| 8 | 5 | .3 | 17 | 25 | 3.13 | 2.56x | 10.24x | 16.95 | 23.55 |
| 8 | 6 | .2 | 22 | 30 | 3.75 | 2.13x | 8.53x | 9.88 | 28.24 |
| 8 | 7 | .1 | 29 | 37 | 4.63 | 1.73x | 6.92x | 4.75 | 34.59 |
| 8 | 8 | .05 | 36 | 44 | 5.50 | 1.45x | 5.82x | 2.36 | 40.67 |
| 8 | 9 | .02 | 45 | 53 | 6.63 | 1.21x | 4.83x | 0.99 | 48.19 |
| 8 | 10 | .01 | 52 | 60 | 7.50 | 1.07x | 4.27x | 0.55 | 53.32 |
| 8 | 11 | .005 | 59 | 67 | 8.38 | 0.96x | 3.82x | 0.26 | 59.67 |
| 8 | 12 | .002 | 69 | 77 | 9.63 | 0.83x | 3.32x | 0.08 | 69.64 |
| **16** | **1** | .9 | **0** | 8 | 0.50 | — † | — † | 102.92 | 7.88 |
| 16 | 2 | .7 | 13 | 21 | 1.31 | 6.10x | 24.38x | 49.12 | 14.31 |
| 16 | 3 | .5 | 16 | 24 | 1.50 | 5.33x | 21.33x | 42.27 | 15.61 |
| 16 | 4 | .4 | 21 | 29 | 1.81 | 4.41x | 17.66x | 35.47 | 17.13 |
| 16 | 5 | .3 | 28 | 36 | 2.25 | 3.56x | 14.22x | 22.46 | 21.10 |
| 16 | 6 | .2 | 37 | 45 | 2.81 | 2.84x | 11.38x | 15.69 | 24.22 |
| 16 | 7 | .1 | 52 | 60 | 3.75 | 2.13x | 8.53x | 7.13 | 31.06 |
| 16 | 8 | .05 | 68 | 76 | 4.75 | 1.68x | 6.74x | 3.47 | 37.32 |
| 16 | 9 | .02 | 88 | 96 | 6.00 | 1.33x | 5.33x | 1.42 | 45.11 |
| 16 | 10 | .01 | 103 | 111 | 6.94 | 1.15x | 4.61x | 0.76 | 50.52 |
| 16 | 11 | .005 | 118 | 126 | 7.88 | 1.02x | 4.06x | 0.42 | 55.71 |
| **32** | **1** | .9 | **0** | 8 | 0.25 | — † | — † | 102.99 | 7.87 |
| 32 | 2 | .7 | 14 | 22 | 0.69 | 11.64x | 46.55x | 102.51 | 7.92 |
| 32 | 3 | .5 | 19 | 27 | 0.84 | 9.48x | 37.93x | 98.49 | 8.26 |
| 32 | 4 | .4 | 29 | 37 | 1.16 | 6.92x | 27.68x | 45.55 | 14.96 |
| 32 | 5 | .3 | 39 | 47 | 1.47 | 5.45x | 21.79x | 36.26 | 16.94 |
| 32 | 6 | .2 | 57 | 65 | 2.03 | 3.94x | 15.75x | 23.42 | 20.74 |
| 32 | 7 | .1 | 90 | 98 | 3.06 | 2.61x | 10.45x | 10.81 | 27.45 |
| 32 | 8 | .05 | 122 | 130 | 4.06 | 1.97x | 7.88x | 5.08 | 34.01 |
| 32 | 9 | .02 | 164 | 172 | 5.38 | 1.49x | 5.95x | 2.02 | 42.01 |
| 32 | 10 | .01 | 195 | 203 | 6.34 | 1.26x | 5.04x | 1.05 | 47.73 |
| **64** | **1** | .9 | **0** | 8 | 0.13 | — † | — † | 102.98 | 7.88 |
| 64 | 2 | .7 | 15 | 23 | 0.36 | 22.26x | 89.04x | 102.98 | 7.88 |
| 64 | 3 | .5 | 21 | 29 | 0.45 | 17.66x | 70.62x | 102.84 | 7.89 |
| 64 | 4 | .4 | 32 | 40 | 0.63 | 12.80x | 51.20x | 100.65 | 8.07 |
| 64 | 5 | .3 | 58 | 66 | 1.03 | 7.76x | 31.03x | 47.29 | 14.64 |
| 64 | 6 | .2 | 84 | 92 | 1.44 | 5.57x | 22.26x | 34.68 | 17.33 |
| 64 | 7 | .1 | 148 | 156 | 2.44 | 3.28x | 13.13x | 16.21 | 23.94 |
| 64 | 8 | .05 | 214 | 222 | 3.47 | 2.31x | 9.23x | 7.35 | 30.80 |
| 64 | 9 | .02 | 299 | 307 | 4.80 | 1.67x | 6.67x | 2.87 | 38.96 |

† **Degenerate, not compression.** `shape_bits == 0` means the recursion collapsed to `M == 1`:
the shape field has zero width, the record is the gain alone, and the direction is gone. The
apparent 16x/32x/64x is pure data loss — the measured RMSE sits at the `√2·σ` no-information
level — so no ratio is quoted for those rows. `hsc::degenerate()` is the query, and the `quant.hpp`
porcelain refuses such a plan unless you type out `allow_degenerate`. The high-ratio region that
*is* real is the **dim** axis: dim32 L2 = 11.6x, dim64 L2 = 22.3x, dim64 L3 = 17.7x, at honest
(if large) distortion.

Two readings worth taking from the table. Down a column, **dimension buys rate**: the same d = 0.2
costs 5.00 bits/elem at dim 4 and 1.44 at dim 64. Across a row, **d buys accuracy** at roughly
`rmse ∝ d`. The two are the whole trade-off surface, and `hsc::plan_for()` walks it for you.

**Bit-exact bytes.** The `psnr 99.99` in the two finest dim-4 rows is not "very good" — the bench
clamps psnr to 99.99 below rmse 1e-9, and an integer rmse is either 0 or ≥ 1/√n, so those cells are
**exact**. `mode::bin` reconstructs integers, so once the worst per-element error falls under half a
count the bytes round back to themselves. `hsc::opts_exact_bytes` names that cell (dim 4, L13,
`gain_bits` 8) and `hsc::exact_bytes(o)` is the predicate; at compile time
`hsc::ct::exact<Src, Back>()` proves it for one payload, which is what `examples/07_comptime.cpp`
does. Read the price honestly: **10.75 bits/byte, ratio 0.744x — that is expansion**, and it must
never be quoted as compression. Note also which axis got you there: level up, dim *down*. Since
`kappa(dim) ≈ 0.29·√dim`, a wider block is less accurate at the same d, and dim 4 is the only lane
where the fine end of the ladder is reachable inside `hsc::ct`'s fixed comptime arenas (dim 8 caps
at L11, dim 16 at L10, dim 32/64 at L8). `hsc::bin_exact_level(dim_log2)` answers 0 for every other dim, meaning
*unmeasured*, not *no*: the corpus sweep stops at d8 L12, d16 L11, d32 L10, d64 L9.

## Preset ladder

`quant.hpp` provides for seven predefined quantization defaults: one cell per dim (plus
`q_balanced`), each Pareto-optimal within its own dim column. The ladder is *not* a Pareto
frontier over the whole dim × {L5, L6, L7} grid — dim8 L7 (4.625 b/w, est 0.081) strictly
dominates `q_finest`, and dim16 L7 matches `q_fine`'s rate at lower estimated error; reach those
cells through `hsc::plan_for()`.

| preset | dim | L | d | bits/weight | vs fp32 | est rel-rmse | measured rel-rmse | psnr dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q_min` | 64 | 5 | .3 | 1.03125 | 31.03x | 0.705 | 0.710 | 29.28 |
| `q_tiny` | 64 | 6 | .2 | 1.4375 | 22.26x | 0.470 | 0.561 | 31.33 |
| `q_small` | 32 | 6 | .2 | 2.03125 | 15.75x | 0.340 | 0.374 | 34.86 |
| `q_compact` | 16 | 6 | .2 | 2.8125 | 11.38x | 0.230 | 0.246 | 38.49 |
| `q_balanced` | 32 | 7 | .1 | 3.0625 | 10.45x | 0.170 | 0.171 | 41.64 |
| `q_fine` | 8 | 6 | .2 | 3.75 | 8.53x | 0.162 | 0.161 | 42.17 |
| `q_finest` | 4 | 6 | .2 | 5.0 | 6.40x | 0.110 | 0.100 | 46.28 |

`est rel-rmse` is the fitted law at that cell, available at compile time from the shape alone.
`measured` is a real round-trip of the first 4096 rows of `all-MiniLM-L6-v2`'s
`embeddings.word_embeddings.weight` (384 wide) through `mode::vec`; PSNR there is
`20·log10(peak/rmse)` against the tensor's own peak magnitude, which is *not* the 8-bit-full-scale
PSNR of the table above. Rates shown are the payload rate the constants state; the frame is a flat
48 bytes per stream.

For context, on that same tensor `int4` costs 4 bits/weight at rel-RMSE 0.140 and `int2` costs 2
at 0.837 (int-N billed at exactly N by the usual convention — the per-row f16 scale, 16/cols,
is not counted), while zstd-19 — lossless, so no error at all — manages 1.19x. The full chart, with the
lossless coders and int-N in one table, is the opening section of
[BENCHMARKS.md](BENCHMARKS.md).

## Model and vector quantization

A tensor is not a flat run of elements, so the codec verbs above are the wrong altitude for one.
`quant.hpp` is the porcelain that knows about rows, bits per weight and the lanes that are wrong
for a weight matrix:

```cpp
hsc::tensor t = hsc::tensor::of(weights, 384);            // [n, 384] row-major f32

auto q = hsc::quantize(t, { .bits_per_weight = 3.0 });    // result<qstream>: bytes + the plan
auto e = hsc::measure(t, hsc::floats{...});               // rel-rmse / cosine / psnr

auto blob = hsc::pack(t, { .rel_rmse = 0.2 });            // self-describing HSCQ container
hsc::qreader rd{ blob_view, sc };
rd.row(9182, out);                                        // one vector, O(cols), no table decode
```

Build/test with duck: `duck batch test.duck` · benches `duck batch build.duck` ·
examples `duck batch examples.duck` · comptime stress `scripts/ctbuild`. Never `-Ofast` (floating-point codec path). MIT license.
