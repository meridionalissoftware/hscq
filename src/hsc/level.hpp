//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  rate control

namespace hsc
{

enum class mode : u8 {
  bin = 0,           //  arbitrary bytes: centered u8 blocks, shape-gain
  vec = 1,           //  f32 blocks
  unit = 2,          //  f32 blocks promised unit-norm
  quotient = 3,      //  f32 complex pairs up to global phase: Hopf-projected to S^2, 4 floats only
  quat = 4,          //  f32 quaternion pairs up to the S^3 fiber: Hopf-projected to S^4, 8 floats only
  oct = 5,           //  f32 octonion pairs up to the S^7 fiber: Hopf-projected to S^8, 16 floats only
  hp1 = 4,           //  alias: the quat base is HP^1
  op1 = 5,           //  alias: the oct base is OP^1
};

inline constexpr i32 default_level = 6;
inline constexpr u32 dq_one = 1u << 24;
inline constexpr u32 dq_min = 1678;      //  d = 1e-4, the validity/level floor (NOT a k_grid member:
                                         //  the dyadic grid has 1680/1671 as neighbors and runs 49
                                         //  entries further down to 1318; snap-up keeps those unreachable)
inline constexpr u32 dq_max = 2u << 24;      //  d = 2, one codeword per node

constexpr u32
dq_of(f64 d) noexcept
{
  if ( d <= 0.0 ) return 0;
  if ( d >= 2.0 ) return dq_max;
  return static_cast<u32>(d * static_cast<f64>(dq_one) + 0.5);
}

constexpr f64
d_of(u32 dq) noexcept
{
  return static_cast<f64>(dq) / static_cast<f64>(dq_one);
}

constexpr bool
dq_valid(u32 dq) noexcept
{
  return dq >= dq_min and dq <= dq_max;
}

//  presets: d in { .9 .7 .5 .4 .3 .2 .1 .05 .02 | .01 .005 .002 .001 5e-4 2e-4 1e-4 }
constexpr u32
level_dq(i32 lvl) noexcept
{
  switch ( lvl < 1 ? 1 : (lvl > 16 ? 16 : lvl) ) {
  case 1:
    return 15099494;      //  d = 0.9   M(4) = 16
  case 2:
    return 11744051;      //  d = 0.7   M(4) = 52
  case 3:
    return 8388608;      //  d = 0.5   M(4) = 138
  case 4:
    return 6710886;      //  d = 0.4   M(4) = 284
  case 5:
    return 5033165;      //  d = 0.3   M(4) = 736
  case 6:
    return 3355443;      //  d = 0.2   M(4) = 2588
  case 7:
    return 1677722;      //  d = 0.1   M(4) = 21844
  case 8:
    return 838861;      //  d = 0.05  M(4) = 178758
  case 9:
    return 335544;      //  d = 0.02  M(4) = 2828294
  case 10:
    return 167772;      //  d = 0.01  M(4) = 22704306
  case 11:
    return 83886;      //  d = 0.005  M(4) = 181975364
  case 12:
    return 33554;      //  d = 0.002  M(4) = 2847067414
  case 13:
    return 16777;      //  d = 0.001  M(4) = 22785126711
  case 14:
    return 8389;      //  d = 5e-4   M(4) = 182282509986
  case 15:
    return 3355;      //  d = 2e-4   M(4) = 2850022310650
  default:
    return 1678;      //  d = 1e-4   M(4) = 22780659936258   (== dq_min, the validity floor)
  }
}

inline constexpr i32 max_level = 16;

//  NOTE: must remain usable as an NTTP by hsc::ct
struct hopf_opts {
  mode m = mode::bin;
  i32 level = default_level;      //  used iff d == 0.0
  f64 d = 0.0;                    //  minimum codeword distance; canonicalized to d_q
  u32 dim_log2 = 3;               //  block dimension n = 2^dim_log2, 2..6; the quotient family pins it (quotient 2, quat 3, oct 4)
  u32 gain_bits = 8;              //  modes bin/vec only, 1..24
  u32 refine = 0;                 //  encoder-side quantizer refinement (0/1); never on the wire; quotient ignores it
  bool transform = false;         //  H*D pre-rotation (codec/rot.hpp, flags bit2); the quotient family forces off
};

constexpr u32
opts_dq(const hopf_opts &o) noexcept
{
  return o.d == 0.0 ? level_dq(o.level) : dq_of(o.d);
}

inline constexpr hopf_opts opts_exact_bytes{ .m = mode::bin, .level = 13, .dim_log2 = 2, .gain_bits = 8 };

constexpr i32
bin_exact_level(u32 dim_log2) noexcept
{
  return dim_log2 == 2 ? 13 : 0;
}

constexpr bool
exact_bytes(const hopf_opts &o) noexcept
{
  const i32 lvl = bin_exact_level(o.dim_log2);
  if ( o.m != mode::bin || lvl == 0 || o.transform || o.gain_bits < 8 ) return false;
  return opts_dq(o) <= level_dq(lvl);      //  d_q falls as the level rises: finer is finer
}

};      //  namespace hsc
