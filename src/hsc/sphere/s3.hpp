//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"
#include "../level.hpp"

#include <micron/math.hpp>
#include <micron/math/simd/atrig.hpp>
#include <micron/math/simd/trig.hpp>
#include <micron/types.hpp>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// 4D: spherical codes by Hopf foliations on S^3
// (ref: Miyamoto/Costa/Sa Earp, IEEE T-IT 2021, Sec. III + Algorithms 1/4)
//
// S^3 foliates into flat tori T_eta = S^1_cos(eta) x S^1_sin(eta);
// leaves are at eta_i = pi/4 + i*Deta, Deta = 2 asin(d/2), i in [-t/2, t/2];
// each torus has n internal circles of m equidistant points, consecutive circles shifted by half a step
// codeword counts are derived solely from d_q
//
// all math is f64 with micron kernels; sqrt/round are the __sqrt/__round twins from config.hpp
// (micron::sd_sqrt / hw::round_sd + fixup by default) and floor is mkbits::round_ns -- NEVER
// compiler builtins, which are banned in src/ (see CLAUDE.md and config.hpp's rationale)
//
// WARNING: the decoder's xi1 reconstruction uses j*Dxi1 + k*Dxi1/2: eq. (34) of the paper prints
// k*Dxi2 for the shift term, which directly contradicts its own encoder (Algorithm 1) and breaks
// 62/138 index round-trips at d = 0.5, proven via tests, see tests/s3.cpp
//
// f64 + u64 only, must provide the leaf buffer

namespace hsc
{

inline constexpr f64 k_pi = micron::math::constant_pi<f64>;
inline constexpr f64 k_2pi = k_pi * 2.0;
inline constexpr f64 k_pi_4 = k_pi / 4.0;

// precomputed radii, the torus grid, and the cumulative index offset
struct s3_leaf {
  f64 ce = 0;       // cos(eta_i)
  f64 se = 0;       // sin(eta_i)
  u32 m = 0;        // points per internal circle
  u32 n = 0;        // internal circles (even unless 1)
  u64 off = 0;      // cumulative codeword offset of this leaf
};

struct s3_skeleton {
  const s3_leaf *lv = nullptr;
  u32 count = 0;        // 2*half + 1 leaves
  u32 half = 0;         // floor(t/2)
  f64 d = 0;            // d_eff = min(d_q / 2^24, 2)
  f64 deta = 0;         // 2 asin(d/2)
  u64 m_total = 0;      // M(4, d)
};

// distances above 2 are unachievable between unit vectors
constexpr f64
__s3_d(u32 dq) noexcept
{
  const f64 d = d_of(dq);
  return d > 2.0 ? 2.0 : d;
}

constexpr u32
__s3_half(f64 d) noexcept
{
  const f64 t = micron::math::mkbits::round_ns::floor<f64>(k_pi / (4.0 * micron::asin(d * 0.5)));
  return static_cast<u32>(t) / 2;
}

// worst case over valid streams (dq_min = 1678, d = d_of(dq_min) ~= 1.00017e-4) is 15705 entries
// (~490 KiB); s3_max_leaves carries 4 of slack.  callers guarantee dq >= dq_min and a buffer of
// s3_leaf_count(dq) entries -- hopf.hpp clamps encode-side, dq_valid rejects decode-side, and
// tree_build checks its leaf arena before calling s3_build (which takes no capacity parameter)
constexpr u32
s3_leaf_count(u32 dq) noexcept
{
  return 2 * __s3_half(__s3_d(dq)) + 1;
}

inline constexpr u32 s3_max_leaves = 15709;

// points on one internal circle of radius ce
constexpr u32
__s3_m(f64 d, f64 ce) noexcept
{
  if ( d > 2.0 * ce ) return 1;
  return static_cast<u32>(micron::math::mkbits::round_ns::floor<f64>(k_pi / micron::asin(d / (2.0 * ce))));
}

// internal circles on the torus
constexpr u32
__s3_n(f64 d, f64 ce, f64 se, u32 m) noexcept
{
  if ( d > 2.0 * se ) return 1;
  const f64 n2 = micron::math::mkbits::round_ns::floor<f64>(k_2pi / micron::asin(d / (2.0 * se)));
  const f64 sh = micron::sin(k_pi / (2.0 * static_cast<f64>(m)));
  const f64 arg = (d * d * 0.25) / (se * se) - (ce * ce) / (se * se) * sh * sh;
  f64 n1 = n2;      // half-step shift alone spans distance d, no n1 constraint
  if ( arg > 0.0 ) {
    const f64 r = __sqrt(arg);
    n1 = r > 1.0 ? 1.0 : micron::math::mkbits::round_ns::floor<f64>(k_pi / micron::asin(r));
  }
  const f64 nmin = n1 < n2 ? n1 : n2;
  const u64 ntil = 2 * (static_cast<u64>(nmin) / 2);
  return ntil > 1 ? static_cast<u32>(ntil) : 1;
}

constexpr s3_skeleton
s3_build(u32 dq, s3_leaf *buf) noexcept
{
  const f64 d = __s3_d(dq);
  const f64 deta = 2.0 * micron::asin(d * 0.5);
  const u32 half = __s3_half(d);
  const u32 count = 2 * half + 1;
  u64 off = 0;
  for ( u32 x = 0; x < count; ++x ) {
    const i64 i = static_cast<i64>(x) - static_cast<i64>(half);
    const f64 eta = k_pi_4 + static_cast<f64>(i) * deta;
    f64 se = 0, ce = 0;
    micron::sincos(eta, se, ce);
    const u32 m = __s3_m(d, ce);
    const u32 n = __s3_n(d, ce, se, m);
    buf[x] = s3_leaf{ ce, se, m, n, off };
    off += static_cast<u64>(m) * static_cast<u64>(n);
  }
  return s3_skeleton{ buf, count, half, d, deta, off };
}

constexpr u64
s3_size(const s3_skeleton &sk) noexcept
{
  return sk.m_total;
}

[[gnu::always_inline]] inline void
__sincos2(f64 a, f64 b, f64 *sn, f64 *cs) noexcept
{
#if defined(__micron_x86_avx2) && defined(__micron_x86_fma) && !defined(HSC_SIMD_OFF)
  namespace avx = micron::simd::avx;
  const double in[4] = { static_cast<double>(a), static_cast<double>(b), 0.0, 0.0 };
  micron::simd::d256 vs, vc;
  micron::math::mk::sincos(avx::loadu_f64(in), &vs, &vc);
  avx::storeu_f64(reinterpret_cast<double *>(sn), vs);
  avx::storeu_f64(reinterpret_cast<double *>(cs), vc);
#else
  micron::sincos(a, sn[0], cs[0]);
  micron::sincos(b, sn[1], cs[1]);
#endif
}

constexpr u32
__s3_leaf_of(const s3_skeleton &sk, u64 a) noexcept
{
  u32 lo = 0, hi = sk.count - 1;
  while ( lo < hi ) {
    const u32 mid = (lo + hi + 1) >> 1;
    if ( sk.lv[mid].off <= a )
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// index -> codeword
constexpr void
s3_decode(const s3_skeleton &sk, u64 a, f64 *out) noexcept
{
  const u32 x = __s3_leaf_of(sk, a);
  const s3_leaf &lf = sk.lv[x];
  const u64 r = a - lf.off;
  const u64 j = r % lf.m;
  const u64 k = r / lf.m;
  const f64 dxi1 = k_2pi / static_cast<f64>(lf.m);
  const f64 dxi2 = k_2pi / static_cast<f64>(lf.n);
  const f64 xi1 = static_cast<f64>(j) * dxi1 + static_cast<f64>(k) * dxi1 * 0.5;
  const f64 xi2 = static_cast<f64>(k) * dxi2;
  // two fibre angles are independent, a single packed sincos covers both;
  // note that micron's packed kernel is bit-identical to its scalar one over [0, 2pi)
  f64 sn[4], cs[4];
  if consteval {
    micron::sincos(xi1, sn[0], cs[0]);
    micron::sincos(xi2, sn[1], cs[1]);
  } else {
    __sincos2(xi1, xi2, sn, cs);
  }
  out[0] = lf.ce * cs[0];
  out[1] = lf.ce * sn[0];
  out[2] = lf.se * cs[1];
  out[3] = lf.se * sn[1];
}

// scale invariant (atan2 of ratios), no need to normalize
struct s3_angles {
  f64 eta;      // half-energy split angle
  f64 xi1;      // fibre angles, in [0, 2pi)
  f64 xi2;
};

[[gnu::always_inline]] constexpr s3_angles
s3_all_angles(const f64 *y, f64 na, f64 nb) noexcept
{
  // all three are independent, pass them in parallel
  f64 r[4];
#if defined(HSC_SIMD_OFF)
  r[0] = micron::math::mk::atan2_bl(nb, na);
  r[1] = micron::math::mk::atan2_bl(y[1], y[0]);
  r[2] = micron::math::mk::atan2_bl(y[3], y[2]);
#else
  if consteval {
    r[0] = micron::math::mk::atan2_bl(nb, na);
    r[1] = micron::math::mk::atan2_bl(y[1], y[0]);
    r[2] = micron::math::mk::atan2_bl(y[3], y[2]);
  } else {
    const f64 ys[4] = { nb, y[1], y[3], 1.0 };
    const f64 xs[4] = { na, y[0], y[2], 1.0 };
    micron::math::mk::atan2_bl_x4(ys, xs, r);
  }
#endif
  // atan2 lands in (-pi, pi]; the fibre coordinates want [0, 2pi)
  return s3_angles{ r[0], r[1] < 0.0 ? f64(r[1] + k_2pi) : r[1], r[2] < 0.0 ? f64(r[2] + k_2pi) : r[2] };
}

constexpr u64
__s3_quantize_leaf(const s3_skeleton &sk, u32 x, const s3_angles &ang) noexcept
{
  const s3_leaf &lf = sk.lv[x];
  const f64 xi1 = ang.xi1;
  const f64 xi2 = ang.xi2;
  const f64 dxi1 = k_2pi / static_cast<f64>(lf.m);
  const f64 dxi2 = k_2pi / static_cast<f64>(lf.n);
  const i64 k = static_cast<i64>(__round(xi2 / dxi2)) % static_cast<i64>(lf.n);
  const i64 jr = static_cast<i64>(__round((xi1 - static_cast<f64>(k) * dxi1 * 0.5) / dxi1)) % static_cast<i64>(lf.m);
  // jr + (m & (jr >> 63))
  const i64 j = jr + (static_cast<i64>(lf.m) & (jr >> 63));
  return lf.off + static_cast<u64>(k) * lf.m + static_cast<u64>(j);
}

// point -> index of a nearby codeword (paper Algorithm 4)
constexpr u64
s3_quantize(const s3_skeleton &sk, const f64 *y, u32 refine = 0) noexcept
{
  const f64 na = __sqrt(y[0] * y[0] + y[1] * y[1]);
  const f64 nb = __sqrt(y[2] * y[2] + y[3] * y[3]);
  const s3_angles ang = s3_all_angles(y, na, nb);
  const f64 eta = ang.eta;
  const i64 half = static_cast<i64>(sk.half);
  i64 i = static_cast<i64>(__round((eta - k_pi_4) / sk.deta));
  i = i < -half ? -half : i;      // clamp to the leaf fan; written as two selects so gcc
  i = i > half ? half : i;        // emits cmov instead of a data-dependent branch pair
  const u32 x0 = static_cast<u32>(i + half);
  const u64 a0 = __s3_quantize_leaf(sk, x0, ang);
  if ( refine == 0 ) return a0;

  u64 best = a0;
  f64 bdot = -4.0;
  for ( i64 dx = -1; dx <= 1; ++dx ) {
    const i64 xi = static_cast<i64>(x0) + dx;
    if ( xi < 0 || xi >= static_cast<i64>(sk.count) ) continue;
    const u64 a = __s3_quantize_leaf(sk, static_cast<u32>(xi), ang);
    f64 p[4]{};
    s3_decode(sk, a, p);
    const f64 dot = p[0] * y[0] + p[1] * y[1] + p[2] * y[2] + p[3] * y[3];
    if ( dot > bdot ) {
      bdot = dot;
      best = a;
    }
  }
  return best;
}

};      // namespace hsc
