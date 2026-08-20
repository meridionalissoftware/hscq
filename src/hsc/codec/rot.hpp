//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"

#include <micron/types.hpp>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// ENERGY BALANCING PREROTATION
// y = (1/sqrt n) H Dx  (header flags bit2)
//
// H is the n * n Walsh-Hadamard matrix (butterfly, adds/subs only, H symmetric, H^2 = nI);
// D is a pinned +-1 diagonal; (1/sqrt n) H is orthogonal AND involutive, so the inverse is D then the same butterfly;
// rows of (1/sqrt n) H D have entries of equal magnitude 1/sqrt(n) (H and D themselves are +-1),
// so every basis vector maps to an exact 50/50 half energy split at every recursion level

namespace hsc
{

// bit c set = negate coordinate c before the butterfly;
// a block of dim n uses the low n bits (prefix property: dim-4 blocks always see bits 0..3)
inline constexpr u64 k_dsign_mask = 0xDC1B77AE0BF34DADull;

// 1/sqrt(2^k) by dim_log2
inline constexpr f64 k_rot_scale[7] = {
  1.0,
  1.0 / __sqrt(f64(2.0)),
  1.0 / __sqrt(f64(4.0)),
  1.0 / __sqrt(f64(8.0)),
  1.0 / __sqrt(f64(16.0)),
  1.0 / __sqrt(f64(32.0)),
  1.0 / __sqrt(f64(64.0)),
};

// in-place Walsh-Hadamard butterfly, adds/subs only (no fma)
constexpr void
__wht(f64 *v, u32 n) noexcept
{
  for ( u32 h = 1; h < n; h <<= 1 ) {
    for ( u32 i = 0; i < n; i += h << 1 ) {
      for ( u32 j = i; j < i + h; ++j ) {
        const f64 a = v[j];
        const f64 b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
    }
  }
}

// encode side, before quantize: signs, butterfly, scale
constexpr void
rot_fwd(f64 *v, u32 dim_log2) noexcept
{
  const u32 n = 1u << dim_log2;
  for ( u32 c = 0; c < n; ++c )
    if ( (k_dsign_mask >> c) & 1u ) v[c] = -v[c];
  __wht(v, n);
  const f64 s = k_rot_scale[dim_log2];
  for ( u32 c = 0; c < n; ++c ) v[c] *= s;
}

// decode side, after reconstruct: butterfly, then scale and signs in one pass
constexpr void
rot_inv(f64 *v, u32 dim_log2) noexcept
{
  const u32 n = 1u << dim_log2;
  __wht(v, n);
  const f64 s = k_rot_scale[dim_log2];
  for ( u32 c = 0; c < n; ++c ) v[c] *= ((k_dsign_mask >> c) & 1u) ? -s : s;
}

};      // namespace hsc
