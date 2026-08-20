//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "codec/scratch.hpp"
#include "config.hpp"
#include "error.hpp"
#include "format.hpp"
#include "hopf.hpp"
#include "level.hpp"
#include "rate.hpp"
#include "unhopf.hpp"

#include <micron/math/log.hpp>
#include <micron/types.hpp>
#include <micron/vector/vector.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  model and vector quantization
//
//  hsc::tensor t = hsc::tensor::of(weights, 384);              // [n, 384] row-major f32
//  auto z = hsc::quantize(t, { .bits_per_weight = 3.0 });      // result<qstream> (pack() is the result<fhsc> entry)
//  auto e = hsc::measure(t.v, back);                           // rel-rmse / cos / psnr; back = floats dequantize() filled
//
//  blocks are dim = 2^dim_log2 elements wide;
//  a row of col floats is ceil(cols/dim) blocks, zero-padded at the tail
//  hsc is fixed-rate, so bits/weight is a closed-form property of (mode, dim, d_q, gain_bits) and the shape
//  mode::vec takes gmax over the whole input in a first pass (hopf.hpp) and stores a one f32 full scale in the header

namespace hsc
{

struct tensor {
  floats v{};
  usize rows = 0;
  usize cols = 0;

  static constexpr tensor
  of(floats v, usize cols) noexcept
  {
    const usize r = cols ? v.size() / cols : 0;
    return tensor{ floats{ v.ptr, r * cols }, r, cols };
  }

  template<f32_source C>
  static tensor
  of(const C &c, usize cols) noexcept
  {
    return of(as_floats(c), cols);
  }

  static constexpr tensor
  vector_of(floats v) noexcept
  {
    return tensor{ v, 1, v.size() };
  }

  constexpr floats
  row(usize i) const noexcept
  {
    return floats{ v.ptr + i * cols, cols };
  }

  constexpr usize
  elems() const noexcept
  {
    return rows * cols;
  }

  constexpr bool
  valid() const noexcept
  {
    return cols != 0 && rows != 0 && rows * cols == v.size() && v.ptr != nullptr;
  }
};

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  fitted distortion law
//
//  rmse ~= min(kappa(dim) * d * sigma, sqrt(2) * sigma)

inline constexpr f64 k_no_info = 1.4142135623730951;      //  sqrt(2): err and signal uncorrelated

constexpr f64
kappa(u32 dim_log2) noexcept
{
  switch ( dim_log2 ) {
  case 2:
    return 0.55;
  case 3:
    return 0.81;
  case 4:
    return 1.15;
  case 5:
    return 1.70;
  default:
    return 2.35;
  }
}

constexpr f64
est_rel_rmse(u32 dim_log2, f64 d) noexcept
{
  const f64 e = kappa(dim_log2) * d;
  return e < k_no_info ? e : k_no_info;
}

constexpr f64
est_rmse(u32 dim_log2, f64 d, f64 sigma) noexcept
{
  return est_rel_rmse(dim_log2, d) * sigma;
}

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  stock presets
//
//  NOTE: the ladder picks one cell per dim (plus q_balanced) and is Pareto within each dim column,
//  NOT over the whole dim x {L5,L6,L7} grid: dim8 L7 (4.625 b/w, est 0.081) strictly dominates
//  q_finest and dim16 L7 (3.75 b/w, est 0.115) matches q_fine's rate at lower estimated error --
//  reach those cells through plan_for()/target, not the ladder

struct qpreset {
  const char *name = nullptr;
  u32 dim_log2 = 3;
  i32 level = 6;
  u32 gain_bits = 8;
  f64 bits_per_weight = 0;      //  exact rate() accounting, framing excluded
  f64 est_rel_rmse = 0;         //  the fitted law at this cell
};

constexpr f64
__preset_rmse(u32 dim_log2, i32 level) noexcept
{
  return est_rel_rmse(dim_log2, d_of(level_dq(level)));
}

//                                          name          dim  L  gb  bits/w      est rel-rmse
inline constexpr qpreset q_min{ "min", 6, 5, 8, 1.03125, __preset_rmse(6, 5) };
inline constexpr qpreset q_tiny{ "tiny", 6, 6, 8, 1.4375, __preset_rmse(6, 6) };
inline constexpr qpreset q_small{ "small", 5, 6, 8, 2.03125, __preset_rmse(5, 6) };
inline constexpr qpreset q_compact{ "compact", 4, 6, 8, 2.8125, __preset_rmse(4, 6) };
inline constexpr qpreset q_balanced{ "balanced", 5, 7, 8, 3.0625, __preset_rmse(5, 7) };
inline constexpr qpreset q_fine{ "fine", 3, 6, 8, 3.75, __preset_rmse(3, 6) };
inline constexpr qpreset q_finest{ "finest", 2, 6, 8, 5.0, __preset_rmse(2, 6) };

inline constexpr qpreset q_presets[] = { q_min, q_tiny, q_small, q_compact, q_balanced, q_fine, q_finest };
inline constexpr usize k_qpresets = sizeof(q_presets) / sizeof(q_presets[0]);

static_assert(q_presets[0].bits_per_weight < q_presets[1].bits_per_weight);
static_assert(q_presets[1].bits_per_weight < q_presets[2].bits_per_weight);
static_assert(q_presets[2].bits_per_weight < q_presets[3].bits_per_weight);
static_assert(q_presets[3].bits_per_weight < q_presets[4].bits_per_weight);
static_assert(q_presets[4].bits_per_weight < q_presets[5].bits_per_weight);
static_assert(q_presets[5].bits_per_weight < q_presets[6].bits_per_weight);
static_assert(q_presets[0].est_rel_rmse > q_presets[1].est_rel_rmse);
static_assert(q_presets[1].est_rel_rmse > q_presets[2].est_rel_rmse);
static_assert(q_presets[2].est_rel_rmse > q_presets[3].est_rel_rmse);
static_assert(q_presets[3].est_rel_rmse > q_presets[4].est_rel_rmse);
static_assert(q_presets[4].est_rel_rmse > q_presets[5].est_rel_rmse);
static_assert(q_presets[5].est_rel_rmse > q_presets[6].est_rel_rmse);

//  NOTE: compares the preset's FRAMING-EXCLUDED rate (qpreset::bits_per_weight) against the budget;
//  plan_for() enforces the framing-INCLUDED qplan::bits_per_weight, so a plan built from pick(b)'s
//  answer can exceed b by the frame share (see examples/03_budget.cpp)
constexpr const qpreset *
pick(f64 bits_per_weight) noexcept
{
  const qpreset *best = nullptr;
  for ( usize i = 0; i < k_qpresets; ++i )
    if ( q_presets[i].bits_per_weight <= bits_per_weight ) best = &q_presets[i];
  return best;
}

//  WARNING: C++ unlike C requires designated initializers in declaration order with no way to override it,
//  therefore we laid out in order of most to least important
struct target {
  mode m = mode::vec;      //  vec, or unit when a row is exactly one block
  i32 level = 0;           //  explicit 1..16 (0 = resolve)
  u32 dim_log2 = 0;        //  explicit 2..6  (0 = resolve)
  u32 gain_bits = 8;       //  per-block gain field, 1..24; vec only

  f64 bits_per_weight = 0;      //  hard budget: the artifact must not exceed it (0 = unset)
  f64 rel_rmse = 0;             //  quality goal against est_rel_rmse
  u32 min_dim_log2 = 2;
  u32 max_dim_log2 = 5;

  usize chunk_rows = 0;               //  container only; 0 = auto (framing <= 0.5% of payload)
  bool chunked = false;               //  false: one bare hsc stream. true: an HSCQ container.
  bool transform = false;             //  H*D pre-rotation: for one-hot / spiky rows (codec/rot.hpp)
  bool row_aligned = true;            //  no block straddles a row; required for row access
  bool allow_degenerate = false;      //  let shape_bits == 0 through (it is not compression)
};

constexpr target
as_target(const qpreset &q) noexcept
{
  target t{};
  t.level = q.level;
  t.dim_log2 = q.dim_log2;
  t.gain_bits = q.gain_bits;
  return t;
}

struct qplan {
  mode m = mode::vec;
  u32 dim_log2 = 3;
  i32 level = 6;
  u32 dq = 0;
  u32 gain_bits = 8;
  bool transform = false;
  bool row_aligned = true;
  bool chunked = false;
  bool degenerate = false;

  usize rows = 0, cols = 0;
  usize blocks_per_row = 0;      //  row_aligned: ceil(cols/dim). 0 when flat.
  usize padded_cols = 0;         //  blocks_per_row * dim; == cols when nothing is padded
  u64 blocks = 0;                //  over the whole tensor, padding included

  usize chunk_rows = 0;      //  == rows when not chunked
  usize chunks = 1;
  usize chunk_bytes = 0;           //  one full chunk's hsc stream
  usize last_chunk_bytes = 0;      //  the final chunk holds the remaining rows

  u32 shape_bits = 0;       //  ceil(log2 M)
  u32 record_bits = 0;      //  gain_bits + shape_bits: the constant per-block width

  usize bytes = 0;
  f64 bits_per_weight = 0;
  f64 payload_bits_per_weight = 0;
  f64 ratio = 0;             //  4 * rows * cols / bytes
  f64 est_rel_rmse = 0;      //  the fitted law, not a bound
};

constexpr hopf_opts
opts_of(const qplan &p) noexcept
{
  return hopf_opts{
    .m = p.m, .level = p.level, .d = 0.0, .dim_log2 = p.dim_log2, .gain_bits = p.gain_bits, .refine = 0, .transform = p.transform
  };
}

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  HSCQ container
//
//  offset   width  field
//   0       4   magic 0x51534889 ("\x89HSQ" on disk; the codec's is "\x89HSC")
//   4       1   version = 1
//   5       1   flags
//               bit0 row_aligned; bits 1..7 reserved zero
//   6       2   reserved zero
//   8       8   rows
//   16      8   cols
//   24      4   chunk_rows
//   28      3   reserved zero
//   31      1   hc = (xxh32(head[0..31)) >> 8) & 0xFF        (as the codec header's own hc byte)
//   32  ..      chunk[0..chunks), chunks = ceil(rows / chunk_rows)

inline constexpr u32 k_qmagic = 0x51534889u;
inline constexpr u8 k_qversion = 1;
inline constexpr usize k_qhead_size = 32;

struct qinfo {
  usize rows = 0, cols = 0;
  usize chunk_rows = 0, chunks = 0;
  bool row_aligned = true;
  qplan p{};
};

namespace __quant
{

constexpr u64
blocks_of(const qplan &p, usize nrows) noexcept
{
  const u64 dim = 1ull << p.dim_log2;
  if ( p.row_aligned ) return static_cast<u64>(nrows) * p.blocks_per_row;
  return (static_cast<u64>(nrows) * p.cols + dim - 1) / dim;
}

constexpr usize
stream_bytes(u64 nblocks, u32 record_bits) noexcept
{
  return k_header_size + static_cast<usize>((nblocks * static_cast<u64>(record_bits) + 7) / 8) + k_trailer_size;
}

inline result<qplan>
build(usize rows, usize cols, u32 dim_log2, i32 level, const target &t, hopf_scratch &sc)
{
  const usize dim = static_cast<usize>(1u) << dim_log2;
  qplan p{};
  p.m = t.m;
  p.dim_log2 = dim_log2;
  p.level = level;
  p.gain_bits = t.gain_bits;
  p.transform = t.transform;
  p.row_aligned = t.row_aligned;
  p.chunked = t.chunked;
  p.rows = rows;
  p.cols = cols;

  const hopf_opts o{
    .m = t.m, .level = level, .d = 0.0, .dim_log2 = dim_log2, .gain_bits = t.gain_bits, .refine = 0, .transform = t.transform
  };
  p.dq = opts_dq(o);
  const result<rate_info> r = rate(o, sc);
  if ( !r.is_first() ) [[unlikely]]
    return r.cast<error>();
  const rate_info &ri = r.cast<rate_info>();
  p.shape_bits = ri.shape_bits;
  p.record_bits = ri.record_bits;
  p.degenerate = ri.shape_bits == 0;

  if ( p.row_aligned ) {
    p.blocks_per_row = (cols + dim - 1) / dim;
    p.padded_cols = p.blocks_per_row * dim;
  } else {
    p.blocks_per_row = 0;
    p.padded_cols = cols;
  }
  p.blocks = blocks_of(p, rows);

  p.chunk_rows = rows;
  p.chunks = 1;
  if ( p.chunked ) {
    usize cr = t.chunk_rows;
    if ( cr == 0 ) {
      const u64 row_bits = static_cast<u64>(p.blocks_per_row) * p.record_bits;
      cr = row_bits ? static_cast<usize>((9600ull * 8 + row_bits - 1) / row_bits) : rows;
    }
    if ( cr < 1 ) cr = 1;
    if ( cr > rows ) cr = rows;
    p.chunk_rows = cr;
    p.chunks = (rows + cr - 1) / cr;
  }

  const usize last_rows = rows - (p.chunks - 1) * p.chunk_rows;
  p.chunk_bytes = stream_bytes(blocks_of(p, p.chunk_rows), p.record_bits);
  p.last_chunk_bytes = stream_bytes(blocks_of(p, last_rows), p.record_bits);
  p.bytes = (p.chunked ? k_qhead_size : 0) + (p.chunks - 1) * p.chunk_bytes + p.last_chunk_bytes;

  const f64 elems = static_cast<f64>(rows) * static_cast<f64>(cols);
  p.payload_bits_per_weight = static_cast<f64>(p.blocks) * static_cast<f64>(p.record_bits) / elems;
  p.bits_per_weight = static_cast<f64>(p.bytes) * 8.0 / elems;
  p.ratio = elems * 4.0 / static_cast<f64>(p.bytes);
  p.est_rel_rmse = est_rel_rmse(dim_log2, d_of(p.dq));
  return p;
}

inline const qpreset &
default_preset(usize cols) noexcept
{
  if ( cols % (static_cast<usize>(1) << q_balanced.dim_log2) == 0 ) return q_balanced;
  for ( usize i = k_qpresets; i-- > 0; )
    if ( cols % (static_cast<usize>(1) << q_presets[i].dim_log2) == 0 ) return q_presets[i];
  return q_finest;
}

};      //  namespace __quant

inline result<qplan>
plan_for(usize rows, usize cols, const target &t, hopf_scratch &sc)
{
  if ( rows == 0 || cols == 0 ) [[unlikely]]
    return error::bad_opts;
  if ( t.m != mode::vec && t.m != mode::unit ) [[unlikely]]
    return error::bad_opts;
  if ( t.gain_bits < 1 || t.gain_bits > 24 ) [[unlikely]]
    return error::bad_opts;
  if ( t.chunked && !t.row_aligned ) [[unlikely]]
    return error::bad_opts;

  u32 lo = t.min_dim_log2 < 2 ? 2 : t.min_dim_log2;
  u32 hi = t.max_dim_log2 > 6 ? 6 : t.max_dim_log2;
  if ( t.dim_log2 ) lo = hi = t.dim_log2;
  if ( lo < 2 || hi > 6 || lo > hi ) [[unlikely]]
    return error::bad_opts;

  if ( t.m == mode::unit ) {
    u32 dl = 0;
    for ( u32 k = 2; k <= 6; ++k )
      if ( (static_cast<usize>(1) << k) == cols ) dl = k;
    if ( dl == 0 || dl < lo || dl > hi ) [[unlikely]]
      return error::bad_opts;
    lo = hi = dl;
  }

  const bool have_budget = t.bits_per_weight > 0.0;
  const bool have_goal = t.rel_rmse > 0.0;

  if ( !have_budget && !have_goal ) {
    const qpreset &d = __quant::default_preset(cols);
    u32 dl = t.dim_log2 ? t.dim_log2 : d.dim_log2;
    if ( dl < lo ) dl = lo;
    if ( dl > hi ) dl = hi;
    const i32 lv = t.level ? t.level : d.level;
    result<qplan> r = __quant::build(rows, cols, dl, lv, t, sc);
    if ( !r.is_first() ) [[unlikely]]
      return r;
    if ( r.cast<qplan>().degenerate && !t.allow_degenerate ) [[unlikely]]
      return error::bad_opts;
    return r;
  }

  qplan best{};
  bool found = false;
  for ( u32 dl = lo; dl <= hi; ++dl ) {
    qplan cand{};
    bool have = false;

    if ( t.level ) {
      const result<qplan> r = __quant::build(rows, cols, dl, t.level, t, sc);
      if ( r.is_first() ) {
        const qplan &q = r.cast<qplan>();
        const bool ok = (!q.degenerate || t.allow_degenerate) && (!have_budget || q.bits_per_weight <= t.bits_per_weight)
                        && (!have_goal || q.est_rel_rmse <= t.rel_rmse);
        if ( ok ) {
          cand = q;
          have = true;
        }
      }
    } else if ( have_budget ) {
      i32 blo = 1, bhi = max_level;
      while ( blo <= bhi ) {
        const i32 mid = blo + (bhi - blo) / 2;
        const result<qplan> r = __quant::build(rows, cols, dl, mid, t, sc);
        if ( !r.is_first() ) break;
        const qplan &q = r.cast<qplan>();
        if ( q.bits_per_weight <= t.bits_per_weight ) {
          if ( !q.degenerate || t.allow_degenerate ) {
            cand = q;
            have = true;
          }
          blo = mid + 1;
        } else {
          bhi = mid - 1;
        }
      }
    } else {
      i32 lv = 0;
      for ( i32 l = 1; l <= max_level; ++l ) {
        if ( est_rel_rmse(dl, d_of(level_dq(l))) <= t.rel_rmse ) {
          lv = l;
          break;
        }
      }
      if ( lv ) {
        const result<qplan> r = __quant::build(rows, cols, dl, lv, t, sc);
        if ( r.is_first() ) {
          const qplan &q = r.cast<qplan>();
          if ( !q.degenerate || t.allow_degenerate ) {
            cand = q;
            have = true;
          }
        }
      }
    }

    if ( !have ) continue;
    if ( !found ) {
      best = cand;
      found = true;
      continue;
    }
    const bool better = have_budget ? (cand.est_rel_rmse < best.est_rel_rmse
                                       || (cand.est_rel_rmse == best.est_rel_rmse && cand.bits_per_weight < best.bits_per_weight))
                                    : (cand.bits_per_weight < best.bits_per_weight);
    if ( better ) best = cand;
  }

  if ( !found ) [[unlikely]]
    return error::bad_opts;
  return best;
}

inline result<qplan>
plan_for(const tensor &x, const target &t, hopf_scratch &sc)
{
  if ( !x.valid() ) [[unlikely]]
    return error::bad_opts;
  return plan_for(x.rows, x.cols, t, sc);
}

constexpr usize
qbound(const qplan &p) noexcept
{
  return p.bytes;
}

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  encoding

namespace __quant
{

//  rows [row0, row0+nrows) of x
inline max_t
encode_chunk(const tensor &x, const qplan &p, usize row0, usize nrows, u8 *buf, usize cap, hopf_scratch &sc, micron::vector<f32> &stage)
{
  const hopf_opts o = opts_of(p);
  if ( p.padded_cols == p.cols ) {
    const floats in{ x.v.ptr + row0 * p.cols, nrows * p.cols };
    return hopf_into(in, o, buf, cap, sc);
  }
  const usize n = nrows * p.padded_cols;
  if ( stage.size() < n ) stage.resize(n);
  f32 *dst = stage.data();
  for ( usize r = 0; r < nrows; ++r ) {
    const f32 *src = x.v.ptr + (row0 + r) * p.cols;
    f32 *o2 = dst + r * p.padded_cols;
    for ( usize c = 0; c < p.cols; ++c ) o2[c] = src[c];
    for ( usize c = p.cols; c < p.padded_cols; ++c ) o2[c] = 0.0f;
  }
  return hopf_into(floats{ dst, n }, o, buf, cap, sc);
}

constexpr void
write_qhead(u8 *h, const qplan &p) noexcept
{
  __store32(h + 0, k_qmagic);
  h[4] = k_qversion;
  h[5] = static_cast<u8>(p.row_aligned ? 0x01u : 0x00u);
  h[6] = h[7] = 0;
  __store64(h + 8, static_cast<u64>(p.rows));
  __store64(h + 16, static_cast<u64>(p.cols));
  __store32(h + 24, static_cast<u32>(p.chunk_rows));
  h[28] = h[29] = h[30] = 0;
  h[31] = static_cast<u8>(xxh32(bytes{ h, 31 }) >> 8);
}

inline result<usize>
write_all(const tensor &x, const qplan &p, wbytes out, hopf_scratch &sc)
{
  if ( !x.valid() || x.rows != p.rows || x.cols != p.cols ) [[unlikely]]
    return error::bad_opts;
  if ( p.degenerate ) [[unlikely]]
    return error::bad_opts;
  if ( out.size() < p.bytes ) [[unlikely]]
    return error::short_output;

  usize off = 0;
  if ( p.chunked ) {
    write_qhead(out.ptr, p);
    off = k_qhead_size;
  }
  micron::vector<f32> stage;
  for ( usize c = 0; c < p.chunks; ++c ) {
    const usize row0 = c * p.chunk_rows;
    const usize nrows = c + 1 == p.chunks ? p.rows - row0 : p.chunk_rows;
    const usize want = c + 1 == p.chunks ? p.last_chunk_bytes : p.chunk_bytes;
    const max_t w = encode_chunk(x, p, row0, nrows, out.ptr + off, out.size() - off, sc, stage);
    if ( w < 0 ) [[unlikely]]
      return as_error(w);
    if ( static_cast<usize>(w) != want ) [[unlikely]]
      return error::bad_length;
    off += static_cast<usize>(w);
  }
  return off;
}

};      //  namespace __quant

struct qstream {
  fhsc z{};
  qplan p{};

  usize
  size() const noexcept
  {
    return z.size();
  }

  bytes
  view() const noexcept
  {
    return bytes{ z.begin(), z.size() };
  }
};

inline result<usize>
quantize_into(const tensor &x, const qplan &p, wbytes out, hopf_scratch &sc)
{
  if ( p.chunked ) [[unlikely]]
    return error::bad_opts;
  return __quant::write_all(x, p, out, sc);
}

inline result<qstream>
__quantize_planned(const tensor &x, const qplan &p, hopf_scratch &sc)
{
  if ( p.chunked ) [[unlikely]]
    return error::bad_opts;
  qstream q{};
  q.p = p;
  q.z = fhsc(fhsc::__uninit_t{}, p.bytes + 1);
  const result<usize> r = __quant::write_all(x, p, wbytes{ q.z.first(), p.bytes }, sc);
  if ( !r.is_first() ) [[unlikely]]
    return r.cast<error>();
  q.z.mark(r.cast<usize>());
  return q;
}

template<typename P>
  requires micron::is_same_v<micron::remove_cvref_t<P>, qplan>
inline result<qstream>
quantize(const tensor &x, P &&p, hopf_scratch &sc)
{
  return __quantize_planned(x, p, sc);
}

inline result<qstream>
quantize(const tensor &x, const target &t, hopf_scratch &sc)
{
  target tt = t;
  tt.chunked = false;
  const result<qplan> p = plan_for(x, tt, sc);
  if ( !p.is_first() ) [[unlikely]]
    return p.cast<error>();
  return __quantize_planned(x, p.cast<qplan>(), sc);
}

inline result<qstream>
quantize(const tensor &x, const target &t = {})
{
  hopf_scratch sc;
  return quantize(x, t, sc);
}

inline result<usize>
pack_into(const tensor &x, const qplan &p, wbytes out, hopf_scratch &sc)
{
  if ( !p.chunked ) [[unlikely]]
    return error::bad_opts;
  return __quant::write_all(x, p, out, sc);
}

inline result<fhsc>
__pack_planned(const tensor &x, const qplan &p, hopf_scratch &sc)
{
  if ( !p.chunked ) [[unlikely]]
    return error::bad_opts;
  fhsc out(fhsc::__uninit_t{}, p.bytes + 1);
  const result<usize> r = __quant::write_all(x, p, wbytes{ out.first(), p.bytes }, sc);
  if ( !r.is_first() ) [[unlikely]]
    return r.cast<error>();
  out.mark(r.cast<usize>());
  return out;
}

template<typename P>
  requires micron::is_same_v<micron::remove_cvref_t<P>, qplan>
inline result<fhsc>
pack(const tensor &x, P &&p, hopf_scratch &sc)
{
  return __pack_planned(x, p, sc);
}

inline result<fhsc>
pack(const tensor &x, const target &t, hopf_scratch &sc)
{
  target tt = t;
  tt.chunked = true;
  tt.row_aligned = true;
  const result<qplan> p = plan_for(x, tt, sc);
  if ( !p.is_first() ) [[unlikely]]
    return p.cast<error>();
  return __pack_planned(x, p.cast<qplan>(), sc);
}

inline result<fhsc>
pack(const tensor &x, const target &t = {})
{
  hopf_scratch sc;
  return pack(x, t, sc);
}

namespace __quant
{

inline max_t
decode_chunk(bytes z, const qplan &p, usize nrows, f32 *out, hopf_scratch &sc, micron::vector<f32> &stage)
{
  if ( p.padded_cols == p.cols ) {
    const result<usize> r = unhopf(z, wfloats{ out, nrows * p.cols }, sc);
    if ( !r.is_first() ) [[unlikely]]
      return fail(r.cast<error>());
    return static_cast<max_t>(r.cast<usize>());
  }
  const usize n = nrows * p.padded_cols;
  if ( stage.size() < n ) stage.resize(n);
  const result<usize> r = unhopf(z, wfloats{ stage.data(), n }, sc);
  if ( !r.is_first() ) [[unlikely]]
    return fail(r.cast<error>());
  for ( usize i = 0; i < nrows; ++i ) {
    const f32 *src = stage.data() + i * p.padded_cols;
    f32 *dst = out + i * p.cols;
    for ( usize c = 0; c < p.cols; ++c ) dst[c] = src[c];
  }
  return static_cast<max_t>(nrows * p.cols);
}

//  rows [first, first+count)
inline max_t
decode_rows(bytes z, const qplan &p, usize first, usize count, f32 *out, hopf_scratch &sc, micron::vector<f32> &stage)
{
  if ( !p.row_aligned ) [[unlikely]]
    return fail(error::bad_opts);
  const u64 fb = static_cast<u64>(first) * p.blocks_per_row;
  const u64 nb = static_cast<u64>(count) * p.blocks_per_row;
  if ( p.padded_cols == p.cols ) {
    const result<usize> r = unhopf_range(z, fb, nb, wfloats{ out, count * p.cols }, sc);
    if ( !r.is_first() ) [[unlikely]]
      return fail(r.cast<error>());
    return static_cast<max_t>(r.cast<usize>());
  }
  const usize n = count * p.padded_cols;
  if ( stage.size() < n ) stage.resize(n);
  const result<usize> r = unhopf_range(z, fb, nb, wfloats{ stage.data(), n }, sc);
  if ( !r.is_first() ) [[unlikely]]
    return fail(r.cast<error>());
  for ( usize i = 0; i < count; ++i ) {
    const f32 *src = stage.data() + i * p.padded_cols;
    f32 *dst = out + i * p.cols;
    for ( usize c = 0; c < p.cols; ++c ) dst[c] = src[c];
  }
  return static_cast<max_t>(count * p.cols);
}

};      //  namespace __quant

inline result<usize>
dequantize(bytes z, const qplan &p, wfloats out, hopf_scratch &sc)
{
  if ( p.chunked ) [[unlikely]]
    return error::bad_opts;
  if ( out.size() < p.rows * p.cols ) [[unlikely]]
    return error::short_output;
  micron::vector<f32> stage;
  const max_t r = __quant::decode_chunk(z, p, p.rows, out.ptr, sc, stage);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  return p.rows * p.cols;
}

inline result<fhsc32>
dequantize(bytes z, const qplan &p)
{
  hopf_scratch sc;
  fhsc32 out(fhsc32::__uninit_t{}, p.rows * p.cols + 1);
  const result<usize> r = dequantize(z, p, wfloats{ out.first(), p.rows * p.cols }, sc);
  if ( !r.is_first() ) [[unlikely]]
    return r.cast<error>();
  out.mark(r.cast<usize>());
  return out;
}

//  rows [first, first+count) of a bare stream
inline result<usize>
dequantize_rows(bytes z, const qplan &p, usize first, usize count, wfloats out, hopf_scratch &sc)
{
  if ( p.chunked ) [[unlikely]]
    return error::bad_opts;
  if ( first > p.rows || count > p.rows - first ) [[unlikely]]
    return error::bad_length;
  if ( out.size() < count * p.cols ) [[unlikely]]
    return error::short_output;
  micron::vector<f32> stage;
  const max_t r = __quant::decode_rows(z, p, first, count, out.ptr, sc, stage);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  return count * p.cols;
}

inline result<usize>
dequantize_row(bytes z, const qplan &p, usize i, wfloats out, hopf_scratch &sc)
{
  return dequantize_rows(z, p, i, 1, out, sc);
}

inline result<usize>
dequantize(const qstream &q, wfloats out, hopf_scratch &sc)
{
  return dequantize(q.view(), q.p, out, sc);
}

inline result<fhsc32>
dequantize(const qstream &q)
{
  return dequantize(q.view(), q.p);
}

inline result<usize>
dequantize_rows(const qstream &q, usize first, usize count, wfloats out, hopf_scratch &sc)
{
  return dequantize_rows(q.view(), q.p, first, count, out, sc);
}

inline result<usize>
dequantize_row(const qstream &q, usize i, wfloats out, hopf_scratch &sc)
{
  return dequantize_rows(q.view(), q.p, i, 1, out, sc);
}

inline result<qinfo>
qprobe(bytes blob)
{
  if ( blob.size() < k_qhead_size + k_header_size + k_trailer_size ) [[unlikely]]
    return error::short_input;
  const u8 *h = blob.ptr;
  if ( __load32(h) != k_qmagic ) [[unlikely]]
    return error::bad_container;
  if ( h[4] != k_qversion ) [[unlikely]]
    return error::unsupported;
  if ( h[31] != static_cast<u8>(xxh32(bytes{ h, 31 }) >> 8) ) [[unlikely]]
    return error::bad_container;
  if ( (h[5] & 0xFEu) || h[6] || h[7] || h[28] || h[29] || h[30] ) [[unlikely]]
    return error::bad_container;

  qinfo qi{};
  qi.row_aligned = (h[5] & 0x01u) != 0;
  qi.rows = static_cast<usize>(__load64(h + 8));
  qi.cols = static_cast<usize>(__load64(h + 16));
  qi.chunk_rows = static_cast<usize>(__load32(h + 24));
  if ( qi.rows == 0 || qi.cols == 0 || qi.chunk_rows == 0 || qi.chunk_rows > qi.rows ) [[unlikely]]
    return error::bad_container;
  if ( !qi.row_aligned ) [[unlikely]]
    return error::bad_container;      //  a chunk boundary must fall on a block boundary
  qi.chunks = (qi.rows + qi.chunk_rows - 1) / qi.chunk_rows;

  const u8 *h0 = blob.ptr + k_qhead_size;
  const u8 m0 = h0[6];
  if ( m0 > 5 ) [[unlikely]]
    return error::bad_container;
  const mode m = static_cast<mode>(m0);
  const u32 dl = h0[7];
  if ( dl < 2 || dl > 6 ) [[unlikely]]
    return error::bad_container;
  const u32 gb = h0[12];
  const u32 sb = __load16(h0 + 14);
  const u32 rb = (__has_gain(m) ? gb : 0) + sb;

  qplan &p = qi.p;
  p.m = m;
  p.dim_log2 = dl;
  p.gain_bits = gb;
  p.shape_bits = sb;
  p.record_bits = rb;
  p.degenerate = sb == 0;
  p.rows = qi.rows;
  p.cols = qi.cols;
  p.row_aligned = qi.row_aligned;
  p.chunked = true;
  p.chunk_rows = qi.chunk_rows;
  p.chunks = qi.chunks;
  const usize dim = static_cast<usize>(1) << dl;
  p.blocks_per_row = (qi.cols + dim - 1) / dim;
  p.padded_cols = p.blocks_per_row * dim;
  p.blocks = __quant::blocks_of(p, qi.rows);
  const usize last_rows = qi.rows - (qi.chunks - 1) * qi.chunk_rows;
  p.chunk_bytes = __quant::stream_bytes(__quant::blocks_of(p, qi.chunk_rows), rb);
  p.last_chunk_bytes = __quant::stream_bytes(__quant::blocks_of(p, last_rows), rb);
  p.bytes = k_qhead_size + (qi.chunks - 1) * p.chunk_bytes + p.last_chunk_bytes;
  if ( blob.size() != p.bytes ) [[unlikely]]
    return error::bad_length;

  const result<hopf_info> hp = hopf_probe(bytes{ h0, p.chunk_bytes });
  if ( !hp.is_first() ) [[unlikely]]
    return hp.cast<error>();
  const hopf_info &fi = hp.cast<hopf_info>();
  if ( fi.m != m || fi.dim_log2 != dl || fi.bits_per_block != sb || fi.gain_bits != (__has_gain(m) ? gb : 0u) ) [[unlikely]]
    return error::bad_container;
  if ( fi.n_elems != static_cast<u64>(qi.chunk_rows) * p.padded_cols ) [[unlikely]]
    return error::bad_length;
  p.dq = fi.d_q;
  p.transform = fi.transform;
  p.level = 0;      //  a stream carries d_q, not the preset it came from
  const f64 elems = static_cast<f64>(qi.rows) * static_cast<f64>(qi.cols);
  p.payload_bits_per_weight = static_cast<f64>(p.blocks) * static_cast<f64>(rb) / elems;
  p.bits_per_weight = static_cast<f64>(p.bytes) * 8.0 / elems;
  p.ratio = elems * 4.0 / static_cast<f64>(p.bytes);
  p.est_rel_rmse = est_rel_rmse(dl, d_of(fi.d_q));
  return qi;
}

namespace __quant
{

constexpr bytes
chunk_at(bytes blob, const qplan &p, usize c) noexcept
{
  const usize off = k_qhead_size + c * p.chunk_bytes;
  const usize n = c + 1 == p.chunks ? p.last_chunk_bytes : p.chunk_bytes;
  return bytes{ blob.ptr + off, n };
}

};      //  namespace __quant

inline result<usize>
unpack(bytes blob, wfloats out, hopf_scratch &sc)
{
  const result<qinfo> qr = qprobe(blob);
  if ( !qr.is_first() ) [[unlikely]]
    return qr.cast<error>();
  const qinfo &qi = qr.cast<qinfo>();
  if ( out.size() < qi.rows * qi.cols ) [[unlikely]]
    return error::short_output;
  micron::vector<f32> stage;
  for ( usize c = 0; c < qi.chunks; ++c ) {
    const usize row0 = c * qi.chunk_rows;
    const usize nrows = c + 1 == qi.chunks ? qi.rows - row0 : qi.chunk_rows;
    const max_t r = __quant::decode_chunk(__quant::chunk_at(blob, qi.p, c), qi.p, nrows, out.ptr + row0 * qi.cols, sc, stage);
    if ( r < 0 ) [[unlikely]]
      return as_error(r);
  }
  return qi.rows * qi.cols;
}

inline result<fhsc32>
unpack(bytes blob)
{
  const result<qinfo> qr = qprobe(blob);
  if ( !qr.is_first() ) [[unlikely]]
    return qr.cast<error>();
  const qinfo &qi = qr.cast<qinfo>();
  hopf_scratch sc;
  fhsc32 out(fhsc32::__uninit_t{}, qi.rows * qi.cols + 1);
  const result<usize> r = unpack(blob, wfloats{ out.first(), qi.rows * qi.cols }, sc);
  if ( !r.is_first() ) [[unlikely]]
    return r.cast<error>();
  out.mark(r.cast<usize>());
  return out;
}

//  rows [first, first+count) of a container
inline result<usize>
unpack_rows(bytes blob, usize first, usize count, wfloats out, hopf_scratch &sc)
{
  const result<qinfo> qr = qprobe(blob);
  if ( !qr.is_first() ) [[unlikely]]
    return qr.cast<error>();
  const qinfo &qi = qr.cast<qinfo>();
  if ( first > qi.rows || count > qi.rows - first ) [[unlikely]]
    return error::bad_length;
  if ( out.size() < count * qi.cols ) [[unlikely]]
    return error::short_output;
  micron::vector<f32> stage;
  usize done = 0;
  while ( done < count ) {
    const usize row = first + done;
    const usize c = row / qi.chunk_rows;
    const usize in_chunk = row - c * qi.chunk_rows;
    const usize chunk_rows_here = c + 1 == qi.chunks ? qi.rows - c * qi.chunk_rows : qi.chunk_rows;
    usize take = chunk_rows_here - in_chunk;
    if ( take > count - done ) take = count - done;
    const max_t r = __quant::decode_rows(__quant::chunk_at(blob, qi.p, c), qi.p, in_chunk, take, out.ptr + done * qi.cols, sc, stage);
    if ( r < 0 ) [[unlikely]]
      return as_error(r);
    done += take;
  }
  return count * qi.cols;
}

inline result<usize>
unpack_row(bytes blob, usize i, wfloats out, hopf_scratch &sc)
{
  return unpack_rows(blob, i, 1, out, sc);
}

struct qreader {
  bytes __blob{};
  hopf_scratch *__sc = nullptr;
  qinfo __i{};
  error __st = error::ok;
  micron::vector<f32> __stage{};

  ~qreader() = default;

  qreader(bytes blob, hopf_scratch &sc) : __blob(blob), __sc(&sc)
  {
    const result<qinfo> r = qprobe(blob);
    if ( r.is_first() )
      __i = r.cast<qinfo>();
    else
      __st = r.cast<error>();
  }

  qreader(const qreader &) = delete;
  qreader &operator=(const qreader &) = delete;

  error
  status() const noexcept
  {
    return __st;
  }

  const qinfo &
  info() const noexcept
  {
    return __i;
  }

  result<usize>
  rows_into(usize first, usize count, wfloats out)
  {
    if ( __st != error::ok ) [[unlikely]]
      return __st;
    if ( first > __i.rows || count > __i.rows - first ) [[unlikely]]
      return error::bad_length;
    if ( out.size() < count * __i.cols ) [[unlikely]]
      return error::short_output;
    usize done = 0;
    while ( done < count ) {
      const usize row = first + done;
      const usize c = row / __i.chunk_rows;
      const usize in_chunk = row - c * __i.chunk_rows;
      const usize chunk_rows_here = c + 1 == __i.chunks ? __i.rows - c * __i.chunk_rows : __i.chunk_rows;
      usize take = chunk_rows_here - in_chunk;
      if ( take > count - done ) take = count - done;
      const max_t r
         = __quant::decode_rows(__quant::chunk_at(__blob, __i.p, c), __i.p, in_chunk, take, out.ptr + done * __i.cols, *__sc, __stage);
      if ( r < 0 ) [[unlikely]]
        return as_error(r);
      done += take;
    }
    return count * __i.cols;
  }

  result<usize>
  row(usize i, wfloats out)
  {
    return rows_into(i, 1, out);
  }
};

//  rel_rmse = ||err|| / ||ref||     cos = <ref,got> / (||ref|| ||got||)
struct qerror {
  f64 rmse = 0;
  f64 rel_rmse = 0;
  f64 cos = 0;
  f64 psnr_db = 0;      //  against the peak magnitude of ref
  usize elems = 0;
};

inline qerror
measure(floats ref, floats got) noexcept
{
  qerror e{};
  const usize n = ref.size() < got.size() ? ref.size() : got.size();
  e.elems = n;
  if ( n == 0 ) return e;
  f64 se = 0, sg = 0, so = 0, dot = 0, peak = 0;
  for ( usize i = 0; i < n; ++i ) {
    const f64 a = static_cast<f64>(ref.ptr[i]), b = static_cast<f64>(got.ptr[i]);
    const f64 d = b - a;
    se += d * d;
    sg += a * a;
    so += b * b;
    dot += a * b;
    const f64 m = a < 0 ? -a : a;
    if ( m > peak ) peak = m;
  }
  e.rmse = micron::math::sd_sqrt(se / static_cast<f64>(n));
  e.rel_rmse = sg > 0 ? micron::math::sd_sqrt(se / sg) : 0.0;
  e.cos = (sg > 0 && so > 0) ? dot / micron::math::sd_sqrt(sg * so) : 0.0;
  e.psnr_db = (peak > 0 && e.rmse > 0) ? 20.0 * static_cast<f64>(micron::math::log10(static_cast<double>(peak / e.rmse))) : 0.0;
  return e;
}

inline qerror
measure(const tensor &ref, floats got) noexcept
{
  return measure(ref.v, got);
}

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  dynamic range
//  ..ratio <<  2^gain_bits     one stream is fine
//  ..ratio ~=  2^gain_bits     quiet blocks are losing gain resolution
//  ..ratio >   2^(gain_bits+1) quiet blocks are being nulled (gq_encode rounds g/step < 0.5 to
//                              code 0, i.e. norm < max/(2*(2^b-1))): pack(), or raise gain_bits

struct grange {
  f64 max_norm = 0;
  f64 min_norm = 0;      //  smallest nonull block norm
  f64 ratio = 0;         //  max_norm / min_norm; 0 when there is nothing to compare
  u64 blocks = 0;
  u64 empty = 0;      //  blocks that were all zeros to begin with
  u64 zeroed = 0;
};

inline grange
gain_range(const tensor &x, u32 dim_log2, u32 gain_bits = 8) noexcept
{
  grange r{};
  if ( !x.valid() ) return r;
  const usize dim = static_cast<usize>(1) << dim_log2;
  const usize bpr = (x.cols + dim - 1) / dim;
  for ( usize row = 0; row < x.rows; ++row ) {
    const f32 *p = x.v.ptr + row * x.cols;
    for ( usize b = 0; b < bpr; ++b ) {
      f64 s = 0;
      for ( usize c = b * dim; c < (b + 1) * dim && c < x.cols; ++c ) s += static_cast<f64>(p[c]) * static_cast<f64>(p[c]);
      const f64 n = micron::math::sd_sqrt(s);
      ++r.blocks;
      if ( !(n > 0) ) {
        ++r.empty;
        continue;
      }
      if ( n > r.max_norm ) r.max_norm = n;
      if ( r.min_norm == 0 || n < r.min_norm ) r.min_norm = n;
    }
  }
  if ( r.min_norm > 0 ) r.ratio = r.max_norm / r.min_norm;
  if ( r.max_norm > 0 ) {
    const f64 step = static_cast<f64>(static_cast<f32>(r.max_norm)) / static_cast<f64>((1u << gain_bits) - 1);
    for ( usize row = 0; row < x.rows; ++row ) {
      const f32 *p = x.v.ptr + row * x.cols;
      for ( usize b = 0; b < bpr; ++b ) {
        f64 s = 0;
        for ( usize c = b * dim; c < (b + 1) * dim && c < x.cols; ++c ) s += static_cast<f64>(p[c]) * static_cast<f64>(p[c]);
        const f64 n = micron::math::sd_sqrt(s);
        if ( n > 0 && __round(n / step) == 0.0 ) ++r.zeroed;
      }
    }
  }
  return r;
}

inline grange
gain_range(const tensor &x, const qplan &p) noexcept
{
  return gain_range(x, p.dim_log2, p.gain_bits);
}

};      //  namespace hsc
