//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../src/hsc/hsc.hpp"

#include <micron/types.hpp>

namespace hc
{

inline constexpr usize k_n = 1 << 16;

struct corpus {
  const char *name;
  const u8 *d;
  usize n;
  u64 mr;
};

inline u8 g_noise[k_n];
inline u8 g_smooth[k_n];
inline u8 g_text[k_n];
inline f32 g_gauss[k_n / 4];
inline f32 g_unit[k_n / 4];
inline u8 g_spiky[k_n];
inline f32 g_onehot[k_n / 4];
inline f32 g_unit16[k_n / 4];       // 16-wide unit blocks: unit-d16 input, and oct's WRONG-lane demo
inline f32 g_fiber8[k_n / 4];       // quaternion pairs with a genuinely random S^3 fiber: quat's home turf
inline f32 g_fiber16[k_n / 4];      // octonion pairs with a random S^7 fiber element: oct's home turf

inline u64 g_seed = 0x9E3779B97F4A7C15ull;

inline u64
xs()
{
  g_seed ^= g_seed << 13;
  g_seed ^= g_seed >> 7;
  g_seed ^= g_seed << 17;
  return g_seed;
}

inline void
generate()
{
  for ( usize i = 0; i < k_n; ++i ) g_noise[i] = static_cast<u8>(xs());
  f64 x = 128.0;
  for ( usize i = 0; i < k_n; ++i ) {
    x += (static_cast<f64>(xs() >> 11) / 9007199254740992.0 - 0.5) * 18.0;
    x = x < 0 ? 0 : (x > 255 ? 255 : x);
    g_smooth[i] = static_cast<u8>(x);
  }
  const char *t = "spherical codes by Hopf foliations compress fixed-rate blocks without a codebook. ";
  for ( usize i = 0; i < k_n; ++i ) g_text[i] = static_cast<u8>(t[i % 83]);
  for ( usize i = 0; i < k_n / 4; ++i ) {

    f64 s = 0;
    for ( i32 k = 0; k < 4; ++k ) s += static_cast<f64>(xs() >> 11) / 9007199254740992.0 - 0.5;
    g_gauss[i] = static_cast<f32>(s);
  }
  for ( usize b = 0; b < k_n / 32; ++b ) {
    f64 v[8], s = 0;
    for ( i32 c = 0; c < 8; ++c ) {
      v[c] = static_cast<f64>(xs() >> 11) / 9007199254740992.0 - 0.5 + 1e-6;
      s += v[c] * v[c];
    }
    const f64 nn = __builtin_sqrt(s);
    for ( i32 c = 0; c < 8; ++c ) g_unit[b * 8 + c] = static_cast<f32>(v[c] / nn);
  }

  for ( usize b = 0; b < k_n / 8; ++b ) {
    const u64 r = xs();
    for ( usize c = 0; c < 8; ++c ) g_spiky[b * 8 + c] = 0x80;
    g_spiky[b * 8 + (r & 7u)] = (r >> 3) & 1u ? u8(0xFF) : u8(0x00);
  }
  for ( usize b = 0; b < k_n / 32; ++b ) {
    const u32 hot = static_cast<u32>(xs() & 7u);
    const f64 sign = (xs() & 1u) ? 1.0 : -1.0;
    f64 v[8], s = 0;
    for ( i32 c = 0; c < 8; ++c ) v[c] = (static_cast<f64>(xs() >> 11) / 9007199254740992.0 - 0.5) * 0.02;
    v[hot] = sign;
    for ( i32 c = 0; c < 8; ++c ) s += v[c] * v[c];
    const f64 nn = __builtin_sqrt(s);
    for ( i32 c = 0; c < 8; ++c ) g_onehot[b * 8 + c] = static_cast<f32>(v[c] / nn);
  }
  for ( usize b = 0; b < k_n / 64; ++b ) {
    f64 v[16], s = 0;
    for ( i32 c = 0; c < 16; ++c ) {
      v[c] = static_cast<f64>(xs() >> 11) / 9007199254740992.0 - 0.5 + 1e-6;
      s += v[c] * v[c];
    }
    const f64 nn = __builtin_sqrt(s);
    for ( i32 c = 0; c < 16; ++c ) g_unit16[b * 16 + c] = static_cast<f32>(v[c] / nn);
  }
  // fiber-symmetric pairs: a random BASE point lifted through a random FIBER element via the
  // exact graph-sphere parametrization (y = s u, x = (v u)/(2s)) -- the data quat/oct exist for
  const auto sphere_pt = [&](u32 n, f64 *p) {
    f64 s = 0;
    for ( u32 c = 0; c <= n; ++c ) {
      p[c] = static_cast<f64>(xs() >> 11) / 9007199254740992.0 - 0.5 + 1e-9;
      s += p[c] * p[c];
    }
    const f64 nn = __builtin_sqrt(s);
    for ( u32 c = 0; c <= n; ++c ) p[c] /= nn;
  };
  for ( usize b = 0; b < k_n / 32; ++b ) {
    f64 p[5], u[4];
    sphere_pt(4, p);
    sphere_pt(3, u);      // 4 components used as a unit quaternion
    const f64 sq = __builtin_sqrt((1.0 - p[4]) * 0.5);
    const f64 cq = __builtin_sqrt((1.0 + p[4]) * 0.5);
    f64 x[4], y[4];
    if ( sq < 1e-6 ) {
      for ( u32 k = 0; k < 4; ++k ) {
        x[k] = cq * u[k];
        y[k] = 0.0;
      }
    } else {
      f64 xv[4];
      hsc::quat_mul(p, u, xv);
      for ( u32 k = 0; k < 4; ++k ) {
        x[k] = xv[k] / (2.0 * sq);
        y[k] = sq * u[k];
      }
    }
    for ( u32 k = 0; k < 4; ++k ) g_fiber8[b * 8 + k] = static_cast<f32>(x[k]);
    for ( u32 k = 0; k < 4; ++k ) g_fiber8[b * 8 + 4 + k] = static_cast<f32>(y[k]);
  }
  for ( usize b = 0; b < k_n / 64; ++b ) {
    f64 p[9], u[8];
    sphere_pt(8, p);
    sphere_pt(7, u);      // 8 components used as a unit octonion
    const f64 sq = __builtin_sqrt((1.0 - p[8]) * 0.5);
    const f64 cq = __builtin_sqrt((1.0 + p[8]) * 0.5);
    f64 x[8], y[8];
    if ( sq < 1e-6 ) {
      for ( u32 k = 0; k < 8; ++k ) {
        x[k] = cq * u[k];
        y[k] = 0.0;
      }
    } else {
      f64 xv[8];
      hsc::oct_mul(p, u, xv);
      for ( u32 k = 0; k < 8; ++k ) {
        x[k] = xv[k] / (2.0 * sq);
        y[k] = sq * u[k];
      }
    }
    for ( u32 k = 0; k < 8; ++k ) g_fiber16[b * 16 + k] = static_cast<f32>(x[k]);
    for ( u32 k = 0; k < 8; ++k ) g_fiber16[b * 16 + 8 + k] = static_cast<f32>(y[k]);
  }
}

inline const corpus corpora[] = {
  { "noise64k", g_noise, k_n, 1 << 8 },
  { "smooth64k", g_smooth, k_n, 1 << 8 },
  { "text64k", g_text, k_n, 1 << 8 },
  { "spiky64k", g_spiky, k_n, 1 << 8 },
};
inline constexpr usize corpora_count = 4;

};      // namespace hc
