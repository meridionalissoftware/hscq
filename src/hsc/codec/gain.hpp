//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"

#include <micron/types.hpp>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// scalar gain quantizer
//
// q = round(g / step), step = scale / (2^bits - 1); q == 0 decodes to exact 0.0.
// q == 0 is NOT unreachable: any block with 0 < g < step/2 rounds to it (the gain cliff --
// see quant.hpp's gain_range); non-positive/NaN gain and zero scale also encode to 0

namespace hsc
{

struct gain_quant {
  u32 bits = 8;       // field width, 1..24
  f32 scale = 0;      // full-scale gain (the f32-rounded canonical value)
};

constexpr u32
gq_levels(const gain_quant &gq) noexcept
{
  return (1u << gq.bits) - 1;
}

constexpr u32
gq_encode(const gain_quant &gq, f64 g) noexcept
{
  const f64 s = static_cast<f64>(gq.scale);
  if ( !(g > 0.0) || !(s > 0.0) ) return 0;
  const f64 step = s / static_cast<f64>(gq_levels(gq));
  const f64 q = __round(g / step);
  const f64 top = static_cast<f64>(gq_levels(gq));
  return static_cast<u32>(q < 0.0 ? 0.0 : (q > top ? top : q));
}

constexpr f64
gq_decode(const gain_quant &gq, u32 q) noexcept
{
  if ( q == 0 ) return 0.0;
  const f64 step = static_cast<f64>(gq.scale) / static_cast<f64>(gq_levels(gq));
  return static_cast<f64>(q) * step;
}

constexpr f32
gq_bin_scale(u32 dim_log2) noexcept
{
  const f64 n = static_cast<f64>(1u << dim_log2);
  return static_cast<f32>(127.5 * __sqrt(n));
}

};      // namespace hsc
