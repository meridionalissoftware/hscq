//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"
#include "../error.hpp"
#include "tree.hpp"

#include <micron/math.hpp>
#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  latitude-band suspension
//  (S^4 / S^8 base quantizers for quat and oct)

namespace hsc
{

struct susp_band {
  f64 st = 0;           //  sin(theta_b): band sphere radius (0 at the poles)
  f64 ct = 0;           //  cos(theta_b): band height (+1 north pole, -1 south pole)
  u32 child = 0;        //  tree node id of the child code (interior bands only)
  u64 m_mod = 0;        //  child cardinality % 2^64 (1 at the poles; exact for dim-4 children)
  u64 off_mod = 0;      //  cumulative codeword offset % 2^64 (exact when total M < 2^64: the quat lane)
};

struct susp_skeleton {
  const susp_band *bd = nullptr;
  u32 count = 0;               //  T' + 2, always >= 2 (== 2 means poles only)
  u32 child_dim_log2 = 2;      //  2: S^3 children (mode quat) | 3: S^7 children (mode oct)
  f64 d = 0;
  f64 dth = 0;           //  2 asin(d/2)
  f64 th_first = 0;      //  colatitude of interior band 1 (0 when count == 2)
  f64 th_last = 0;       //  colatitude of interior band T' (0 when count == 2)
  u64 m_mod = 0;         //  total cardinality % 2^64 (exact for child_dim_log2 == 2)
};

constexpr u32
susp_band_count(u32 dq) noexcept
{
  const f64 d = __s3_d(dq);
  const f64 dth = 2.0 * micron::asin(d * 0.5);
  const f64 w = k_pi - 2.0 * dth;
  if ( w < 0.0 ) return 2;
  u32 tp = static_cast<u32>(micron::math::mkbits::round_ns::floor<f64>(w / dth)) + 1;
  if ( (tp & 1u) == 0 ) --tp;      //  anchored at the equator
  return tp + 2;
}

//  worst case over valid streams (dq_min = 1678, d = d_of(dq_min)) is 31411 entries (~1.2 MiB)
inline constexpr u32 susp_max_bands = 31413;

constexpr max_t
susp_build(u32 dq, u32 child_dim_log2, tree_arena &ar, susp_band *buf, susp_skeleton &out) noexcept
{
  const f64 d = __s3_d(dq);
  const f64 dth = 2.0 * micron::asin(d * 0.5);
  const f64 w = k_pi - 2.0 * dth;
  u32 tp = w < 0.0 ? 0 : static_cast<u32>(micron::math::mkbits::round_ns::floor<f64>(w / dth)) + 1;
  if ( tp != 0 && (tp & 1u) == 0 ) --tp;
  const u32 count = tp + 2;
  const f64 th_first = tp ? dth + (w - static_cast<f64>(tp - 1) * dth) * 0.5 : 0.0;
  const f64 th_last = tp ? th_first + static_cast<f64>(tp - 1) * dth : 0.0;
  buf[0] = susp_band{ 0.0, 1.0, 0, 1, 0 };
  u64 m_mod = 1;
  for ( u32 b = 0; b < tp; ++b ) {
    const f64 th = th_first + static_cast<f64>(b) * dth;
    f64 st = 0, ct = 0;
    micron::sincos(th, st, ct);
    const max_t c = tree_build(child_dim_log2, grid_snap(st > 0.0 ? d / st : 2.0), ar);
    if ( c < 0 ) [[unlikely]]
      return c;
    const u64 cm = ar.nodes[static_cast<u32>(c)].m_mod;
    buf[1 + b] = susp_band{ st, ct, static_cast<u32>(c), cm, m_mod };
    m_mod += cm;      //  u64 wrap by design (matches the wire's M % 2^64 guard)
  }
  buf[count - 1] = susp_band{ 0.0, -1.0, 0, 1, m_mod };
  m_mod += 1;
  out = susp_skeleton{ buf, count, child_dim_log2, d, dth, th_first, th_last, m_mod };
  return 0;
}

constexpr u64
susp_size(const susp_skeleton &sk) noexcept
{
  return sk.m_mod;
}

constexpr tree_skeleton
susp_child_view(const susp_skeleton &sk, const tree_skeleton &tv, u32 band) noexcept
{
  const u32 c = sk.bd[band].child;
  return tree_skeleton{ tv.nodes, tv.rows, tv.leaves, c, sk.child_dim_log2, tv.nodes[c].dq };
}

constexpr u32
__susp_band_of(const susp_skeleton &sk, u64 a) noexcept
{
  u32 lo = 0, hi = sk.count - 1;
  while ( lo < hi ) {
    const u32 mid = (lo + hi + 1) >> 1;
    if ( sk.bd[mid].off_mod <= a )
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

constexpr void
susp_decode(const susp_skeleton &sk, const tree_skeleton &tv, u32 band, const tree_fields &f, f64 *p) noexcept
{
  const u32 n = 1u << sk.child_dim_log2;
  const susp_band &bd = sk.bd[band];
  if ( band == 0 || band == sk.count - 1 ) {
    for ( u32 c = 0; c < n; ++c ) p[c] = 0.0;
    p[n] = bd.ct;      //  exact +-1
    return;
  }
  tree_decode(susp_child_view(sk, tv, band), f, p);
  for ( u32 c = 0; c < n; ++c ) p[c] *= bd.st;
  p[n] = bd.ct;
}

constexpr f64
__susp_dot(const susp_skeleton &sk, const tree_skeleton &tv, u32 band, const f64 *p, tree_fields &f, u32 refine) noexcept
{
  namespace fu = micron::math::mk::fused;
  const u32 n = 1u << sk.child_dim_log2;
  const susp_band &bd = sk.bd[band];
  if ( band == 0 || band == sk.count - 1 ) return bd.ct * p[n];
  const tree_skeleton cv = susp_child_view(sk, tv, band);
  tree_quantize(cv, p, f, refine);
  f64 v[64];
  tree_decode(cv, f, v);
  f64 s = v[0] * p[0];
  for ( u32 c = 1; c < n; ++c ) s = fu::fma(v[c], p[c], s);
  return fu::fma(bd.st, s, bd.ct * p[n]);
}

//  n+1 f64s, height last (p[n] = h), n = 1 << child_dim_log2
constexpr u32
susp_quantize(const susp_skeleton &sk, const tree_skeleton &tv, const f64 *p, tree_fields &f, u32 refine = 0) noexcept
{
  const u32 n = 1u << sk.child_dim_log2;
  const f64 hz = p[n] < -1.0 ? -1.0 : (p[n] > 1.0 ? 1.0 : p[n]);
  const f64 th = micron::acos(hz);
  u32 band = 0;
  if ( sk.count == 2 ) {
    band = th > k_pi * 0.5 ? 1u : 0u;      //  tie at the equator -> north
  } else {
    const i64 tp = static_cast<i64>(sk.count) - 2;
    i64 j = static_cast<i64>(__round((th - sk.th_first) / sk.dth));
    j = j < 0 ? 0 : j;      //  two selects, not a nested branch pair (see s3_quantize)
    j = j >= tp ? tp - 1 : j;
    band = 1u + static_cast<u32>(j);
    band = th < 0.5 * sk.th_first ? 0u : band;                         //  north strictly nearer than band 1
    band = th > 0.5 * (sk.th_last + k_pi) ? sk.count - 1u : band;      //  south strictly nearer than band T'
  }
  if ( !refine ) {
    if ( band != 0 && band != sk.count - 1 ) tree_quantize(susp_child_view(sk, tv, band), p, f, 0);
    return band;
  }
  f64 best = -4.0;
  u32 bb = band;
  tree_fields bf{};
  const i64 b0 = static_cast<i64>(band);
  for ( i64 db = -1; db <= 1; ++db ) {
    const i64 cand = b0 + db;
    if ( cand < 0 || cand >= static_cast<i64>(sk.count) ) continue;
    tree_fields tf{};
    const f64 dot = __susp_dot(sk, tv, static_cast<u32>(cand), p, tf, refine);
    if ( dot > best ) {
      best = dot;
      bb = static_cast<u32>(cand);
      bf = tf;
    }
  }
  if ( bb != 0 && bb != sk.count - 1 ) f = bf;
  return bb;
}

};      //  namespace hsc
