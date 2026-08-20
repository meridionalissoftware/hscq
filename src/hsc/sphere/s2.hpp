//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"
#include "../level.hpp"
#include "s3.hpp"

#include <micron/math.hpp>
#include <micron/math/simd/atrig.hpp>
#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  S^2 quantizer for quotient mode
//
//  colatitude theta in [0, pi] slices into T = floor(pi/Dth) + 1 bands separated by Dth = 2 asin(d/2), placed symmetrically about the
//  equator
//
//  f64 + u64 only, buffer must be provided

namespace hsc
{

struct s2_band {
  f64 st = 0;       //  sin(theta_b): band circle radius
  f64 ct = 0;       //  cos(theta_b): band height
  u32 m = 0;        //  points on the band
  u64 off = 0;      //  cumulative codeword offset
};

struct s2_skeleton {
  const s2_band *bd = nullptr;
  u32 count = 0;
  f64 d = 0;
  f64 dth = 0;      //  2 asin(d/2)
  f64 th0 = 0;      //  first band colatitude: (pi - (T-1) Dth) / 2
  u64 m_total = 0;
};

constexpr u32
s2_band_count(u32 dq) noexcept
{
  const f64 d = __s3_d(dq);
  const f64 dth = 2.0 * micron::asin(d * 0.5);
  return static_cast<u32>(micron::math::mkbits::round_ns::floor<f64>(k_pi / dth)) + 1;
}

//  worst case over valid streams (dq_min = 1678) is T = 31411 bands; 7 of slack
inline constexpr u32 s2_max_bands = 31418;

constexpr s2_skeleton
s2_build(u32 dq, s2_band *buf) noexcept
{
  const f64 d = __s3_d(dq);
  const f64 dth = 2.0 * micron::asin(d * 0.5);
  const u32 count = static_cast<u32>(micron::math::mkbits::round_ns::floor<f64>(k_pi / dth)) + 1;
  const f64 th0 = (k_pi - static_cast<f64>(count - 1) * dth) * 0.5;
  u64 off = 0;
  for ( u32 b = 0; b < count; ++b ) {
    const f64 th = th0 + static_cast<f64>(b) * dth;
    f64 st = 0, ct = 0;
    micron::sincos(th, st, ct);
    u32 m = 1;
    if ( d <= 2.0 * st ) m = static_cast<u32>(micron::math::mkbits::round_ns::floor<f64>(k_pi / micron::asin(d / (2.0 * st))));
    buf[b] = s2_band{ st, ct, m, off };
    off += m;
  }
  return s2_skeleton{ buf, count, d, dth, th0, off };
}

constexpr u64
s2_size(const s2_skeleton &sk) noexcept
{
  return sk.m_total;
}

constexpr u32
__s2_band_of(const s2_skeleton &sk, u64 a) noexcept
{
  u32 lo = 0, hi = sk.count - 1;
  while ( lo < hi ) {
    const u32 mid = (lo + hi + 1) >> 1;
    if ( sk.bd[mid].off <= a )
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

constexpr void
s2_decode(const s2_skeleton &sk, u64 a, f64 *out) noexcept
{
  const u32 b = __s2_band_of(sk, a);
  const s2_band &bd = sk.bd[b];
  const f64 phi = static_cast<f64>(a - bd.off) * (k_2pi / static_cast<f64>(bd.m));
  f64 sp = 0, cp = 0;
  micron::sincos(phi, sp, cp);
  out[0] = bd.st * cp;
  out[1] = bd.st * sp;
  out[2] = bd.ct;
}

constexpr u64
s2_quantize(const s2_skeleton &sk, const f64 *p) noexcept
{
  const f64 z = p[2] < -1.0 ? -1.0 : (p[2] > 1.0 ? 1.0 : p[2]);
  const f64 th = micron::acos(z);
  i64 b = static_cast<i64>(__round((th - sk.th0) / sk.dth));
  b = b < 0 ? 0 : (b >= static_cast<i64>(sk.count) ? static_cast<i64>(sk.count) - 1 : b);
  const s2_band &bd = sk.bd[b];
  const f64 ph = micron::math::mk::atan2_bl(p[1], p[0]);
  const f64 phi = ph < 0.0 ? f64(ph + k_2pi) : ph;      //  select, not a coin-flip branch
  const i64 j = static_cast<i64>(__round(phi * static_cast<f64>(bd.m) / k_2pi)) % static_cast<i64>(bd.m);
  return bd.off + static_cast<u64>(j);
}

};      //  namespace hsc
