//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"
#include "../error.hpp"
#include "../sphere/s2.hpp"

#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  compression by quotienting the Hopf fiber
//
//  a block of 4 floats is read as (z0, z1) = (y0 + i y1, y2 + i y3) in S^3;
//  when insensitive to a global phase (q ~ q e^{i psi}), storing q elims the fiber coordinate;
//  Hopf map h(z0, z1) = (2 z0 conj(z1), |z0|^2 - |z1|^2) projects onto S^2 = S^3/S^1;
//  S^2 band code stores the class;
//  yielding ~log2(1/d) fewer bits than the full S^3 code at the same d
//
//  h scales as h(lambda q) = lambda^2 h(q), so normalizing the projection by |q|^2 (no sqrt) makes the whole pipeline scale invariant

namespace hsc
{

//  project a (nonzero) 4-float block to its unit S^2 class point; fail(bad_value) on NaN/Inf/0
constexpr max_t
hopf_project(const f32 *z, f64 *p) noexcept
{
  const f64 y0 = static_cast<f64>(z[0]);
  const f64 y1 = static_cast<f64>(z[1]);
  const f64 y2 = static_cast<f64>(z[2]);
  const f64 y3 = static_cast<f64>(z[3]);
  const f64 n2 = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
  if ( !(n2 > 0.0) || !(n2 < 1e300) ) [[unlikely]]
    return fail(error::bad_value);
  p[0] = 2.0 * (y0 * y2 + y1 * y3) / n2;
  p[1] = 2.0 * (y1 * y2 - y0 * y3) / n2;
  p[2] = (y0 * y0 + y1 * y1 - y2 * y2 - y3 * y3) / n2;
  return 0;
}

//  p on S^2 -> the unit (z0, z1) with z0 real >= 0 mapping onto it;
//  p[2] <= -1 + 1e-12 short-circuits to the [0 : z1] class (z0 = 0, z1 = 1), as in quat/oct --
//  near that pole the section loses precision to cancellation in 1 + p[2] (~5e-5 relative at the
//  threshold), but no s2 band center lands in that regime
constexpr void
hopf_lift(const f64 *p, f32 *z) noexcept
{
  if ( p[2] <= -1.0 + 1e-12 ) {
    z[0] = 0.0f;
    z[1] = 0.0f;
    z[2] = 1.0f;
    z[3] = 0.0f;
    return;
  }
  const f64 z0 = __sqrt((1.0 + p[2]) * 0.5);
  z[0] = static_cast<f32>(z0);
  z[1] = 0.0f;
  z[2] = static_cast<f32>(p[0] / (2.0 * z0));
  z[3] = static_cast<f32>(-p[1] / (2.0 * z0));
}

constexpr max_t
quotient_quantize(const s2_skeleton &sk, const f32 *z, u64 &a) noexcept
{
  f64 p[3]{};
  const max_t r = hopf_project(z, p);
  if ( r < 0 ) [[unlikely]]
    return r;
  a = s2_quantize(sk, p);
  return 0;
}

constexpr void
quotient_reconstruct(const s2_skeleton &sk, u64 a, f32 *z) noexcept
{
  f64 p[3]{};
  s2_decode(sk, a, p);
  hopf_lift(p, z);
}

};      //  namespace hsc
