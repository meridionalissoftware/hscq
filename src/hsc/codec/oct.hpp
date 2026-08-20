//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"
#include "../error.hpp"
#include "quat.hpp"

#include <micron/math.hpp>
#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  compression by quotienting the octonionic Hopf fiber (mode oct, S^15 -> S^8)
//
//  a block of 16 floats is read as two octonions (o0, o1), each stored vector-first,
//  scalar-last of 8: o = (e1..e7, e0) with basis e1,e2,e3 = i,j,k; e4 = l;
//  e5,e6,e7 = il,jl,kl.  Cayley-Dickson split: o = A + B l with the scalar-last quats
//  A = (o[0], o[1], o[2], o[7]) and B = (o[4], o[5], o[6], o[3]); product convention
//   (A1 + B1 l)(A2 + B2 l) = (A1 A2 - conj(B2) B1) + (B2 A1 + B1 conj(A2)) l
//  which realizes e1 e4 = e5, e2 e4 = e6, e3 e4 = e7
//
//  NOTE: octonions are not associative; two pairs are identified iff they carry the same left-ratio o1 o0^-1 (fibers are graph spheres {(a,
//  q a)}) Hopf map
//    h(o0, o1) = (2 o0 conj(o1), |o0|^2 - |o1|^2)  in  R^8 x R = R^9
//  projects onto S^8 = S^15/S^7 by norm multiplicativity
//
//  scale invariance and the fused-op discipline are exactly as in quat.hpp

namespace hsc
{

constexpr void
oct_conj(const f64 *o, f64 *out) noexcept
{
  for ( u32 k = 0; k < 7; ++k ) out[k] = -o[k];
  out[7] = o[7];
}

//  Cayley-Dickson product on (e1..e7, e0) scalar-last storage; out must not alias a or b
constexpr void
oct_mul(const f64 *a, const f64 *b, f64 *out) noexcept
{
  const f64 a1[4] = { a[0], a[1], a[2], a[7] };
  const f64 b1[4] = { a[4], a[5], a[6], a[3] };
  const f64 a2[4] = { b[0], b[1], b[2], b[7] };
  const f64 b2[4] = { b[4], b[5], b[6], b[3] };
  f64 c[4], t[4], u[4], r1[4], r2[4];
  quat_mul(a1, a2, t);      //  A1 A2
  quat_conj(b2, c);
  quat_mul(c, b1, u);      //  conj(B2) B1
  for ( u32 k = 0; k < 4; ++k ) r1[k] = t[k] - u[k];
  quat_mul(b2, a1, t);      //  B2 A1
  quat_conj(a2, c);
  quat_mul(b1, c, u);      //  B1 conj(A2)
  for ( u32 k = 0; k < 4; ++k ) r2[k] = t[k] + u[k];
  out[0] = r1[0];
  out[1] = r1[1];
  out[2] = r1[2];
  out[7] = r1[3];
  out[4] = r2[0];
  out[5] = r2[1];
  out[6] = r2[2];
  out[3] = r2[3];
}

//  project a (nonzero) 16-float block to its unit S^8 class point (height last)
//  (p[0..7] = 2 o0 conj(o1) / n2, p[8] = (|o0|^2 - |o1|^2) / n2)
constexpr max_t
oct_project(const f32 *z, f64 *p) noexcept
{
  f64 x[8], y[8];
  for ( u32 k = 0; k < 8; ++k ) x[k] = static_cast<f64>(z[k]);
  for ( u32 k = 0; k < 8; ++k ) y[k] = static_cast<f64>(z[8 + k]);
  const f64 nx = __fma_norm2(x, 8);
  const f64 ny = __fma_norm2(y, 8);
  const f64 n2 = nx + ny;
  if ( !(n2 > 0.0) || !(n2 < 1e300) ) [[unlikely]]
    return fail(error::bad_value);
  f64 yc[8], v[8];
  oct_conj(y, yc);
  oct_mul(x, yc, v);
  for ( u32 k = 0; k < 8; ++k ) p[k] = 2.0 * v[k] / n2;
  p[8] = (nx - ny) / n2;
  return 0;
}

//  p on S^8 -> the unit (o0, o1) with o0 = (0,...,0,c), c real >= 0, mapping onto it
constexpr void
oct_lift(const f64 *p, f32 *z) noexcept
{
  if ( p[8] <= -1.0 + 1e-12 ) {
    for ( u32 k = 0; k < 15; ++k ) z[k] = 0.0f;
    z[15] = 1.0f;      //  the [0 : y] class: o0 = 0, o1 = identity
    return;
  }
  const f64 c = __sqrt((1.0 + p[8]) * 0.5);
  for ( u32 k = 0; k < 7; ++k ) z[k] = 0.0f;
  z[7] = static_cast<f32>(c);
  //  o1 = conj(v) / (2 c)
  for ( u32 k = 0; k < 7; ++k ) z[8 + k] = static_cast<f32>(-p[k] / (2.0 * c));
  z[15] = static_cast<f32>(p[7] / (2.0 * c));
}

constexpr max_t
oct_quantize(const susp_skeleton &ss, const tree_skeleton &tv, const f32 *z, u32 &band, tree_fields &f, u32 refine = 0) noexcept
{
  f64 p[9]{};
  const max_t r = oct_project(z, p);
  if ( r < 0 ) [[unlikely]]
    return r;
  band = susp_quantize(ss, tv, p, f, refine);
  return 0;
}

constexpr void
oct_reconstruct(const susp_skeleton &ss, const tree_skeleton &tv, u32 band, const tree_fields &f, f32 *z) noexcept
{
  f64 p[9]{};
  susp_decode(ss, tv, band, f, p);
  oct_lift(p, z);
}

};      //  namespace hsc
