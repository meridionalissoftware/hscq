//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"
#include "../error.hpp"
#include "../sphere/susp.hpp"

#include <micron/math.hpp>
#include <micron/types.hpp>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// compression by quotienting the quaternionic Hopf fiber (mode quat, S^7 -> S^4)
//
// a block of 8 floats is read as two quaternions (q0, q1), each stored vector-first,
// scalar-last: q = (x, y, z, w) = w + x i + y j + z k
// the Hopf map
//   h(q0, q1) = (2 q0 conj(q1), |q0|^2 - |q1|^2)  in  R^4 x R = R^5
// projects onto S^4 = S^7/S^3 and the cap-anchored suspension code stores the class,
// yielding ~3 log2(1/d) fewer bits than the full S^7 code at the same d.
// note that left multiplication is not quotiented
//
// h scales as h(lambda q) = lambda^2 h(q); normalizing by |q|^2 (no sqrt) makes the
// whole pipeline scale invariant, exactly like mode quotient

namespace hsc
{

// |q|^2 over n components as one fixed-association fma chain
constexpr f64
__fma_norm2(const f64 *q, u32 n) noexcept
{
  namespace fu = micron::math::mk::fused;
  f64 s = q[0] * q[0];
  for ( u32 k = 1; k < n; ++k ) s = fu::fma(q[k], q[k], s);
  return s;
}

constexpr void
quat_conj(const f64 *q, f64 *out) noexcept
{
  out[0] = -q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = q[3];
}

// Hamilton product on (x, y, z, w) scalar-last storage; out must not alias a or b
constexpr void
quat_mul(const f64 *a, const f64 *b, f64 *out) noexcept
{
  namespace fu = micron::math::mk::fused;
  out[0] = fu::fma(a[3], b[0], fu::fma(a[0], b[3], fu::fms(a[1], b[2], a[2] * b[1])));
  out[1] = fu::fma(a[3], b[1], fu::fnma(a[0], b[2], fu::fma(a[1], b[3], a[2] * b[0])));
  out[2] = fu::fma(a[3], b[2], fu::fma(a[0], b[1], fu::fnma(a[1], b[0], a[2] * b[3])));
  out[3] = fu::fms(a[3], b[3], fu::fma(a[0], b[0], fu::fma(a[1], b[1], a[2] * b[2])));
}

// project a (nonzero) 8-float block to its unit S^4 class point, height last
// (p[0..3] = 2 q0 conj(q1) / n2, p[4] = (|q0|^2 - |q1|^2) / n2); fail(bad_value) on NaN/Inf/0
constexpr max_t
quat_project(const f32 *z, f64 *p) noexcept
{
  f64 x[4], y[4];
  for ( u32 k = 0; k < 4; ++k ) x[k] = static_cast<f64>(z[k]);
  for ( u32 k = 0; k < 4; ++k ) y[k] = static_cast<f64>(z[4 + k]);
  const f64 nx = __fma_norm2(x, 4);
  const f64 ny = __fma_norm2(y, 4);
  const f64 n2 = nx + ny;
  if ( !(n2 > 0.0) || !(n2 < 1e300) ) [[unlikely]]
    return fail(error::bad_value);
  f64 yc[4], v[4];
  quat_conj(y, yc);
  quat_mul(x, yc, v);
  for ( u32 k = 0; k < 4; ++k ) p[k] = 2.0 * v[k] / n2;
  p[4] = (nx - ny) / n2;
  return 0;
}

// p on S^4 -> the unit (q0, q1) with q0 = (0,0,0,c), c real >= 0, mapping onto it
constexpr void
quat_lift(const f64 *p, f32 *z) noexcept
{
  if ( p[4] <= -1.0 + 1e-12 ) {
    for ( u32 k = 0; k < 7; ++k ) z[k] = 0.0f;
    z[7] = 1.0f;      // the [0 : y] class: q0 = 0, q1 = identity
    return;
  }
  const f64 c = __sqrt((1.0 + p[4]) * 0.5);
  z[0] = 0.0f;
  z[1] = 0.0f;
  z[2] = 0.0f;
  z[3] = static_cast<f32>(c);
  // q1 = conj(v) / (2 c)
  z[4] = static_cast<f32>(-p[0] / (2.0 * c));
  z[5] = static_cast<f32>(-p[1] / (2.0 * c));
  z[6] = static_cast<f32>(-p[2] / (2.0 * c));
  z[7] = static_cast<f32>(p[3] / (2.0 * c));
}

constexpr max_t
quat_quantize(const susp_skeleton &ss, const tree_skeleton &tv, const f32 *z, u64 &a, u32 refine = 0) noexcept
{
  f64 p[5]{};
  const max_t r = quat_project(z, p);
  if ( r < 0 ) [[unlikely]]
    return r;
  tree_fields f{};
  const u32 band = susp_quantize(ss, tv, p, f, refine);
  const bool pole = band == 0 || band == ss.count - 1;
  a = ss.bd[band].off_mod + (pole ? 0 : f.base[0]);      // a dim-4 child rank IS its s3 index
  return 0;
}

constexpr void
quat_reconstruct(const susp_skeleton &ss, const tree_skeleton &tv, u64 a, f32 *z) noexcept
{
  const u32 band = __susp_band_of(ss, a);
  tree_fields f{};
  f.base[0] = a - ss.bd[band].off_mod;
  f64 p[5]{};
  susp_decode(ss, tv, band, f, p);
  quat_lift(p, z);
}

};      // namespace hsc
