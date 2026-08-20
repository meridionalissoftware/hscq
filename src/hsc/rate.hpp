//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "codec/pack.hpp"
#include "codec/scratch.hpp"
#include "format.hpp"
#include "hopf.hpp"
#include "level.hpp"

namespace hsc
{

struct rate_info {
  mode m = mode::bin;
  u32 dim_log2 = 3;
  u32 dq = 0;
  u32 gain_bits = 0;
  u32 shape_bits = 0;
  u32 record_bits = 0;
  u32 block_elems = 0;
  f64 bits_per_elem = 0;
  f64 bits_per_input_byte = 0;
  f64 ratio = 0;
};

inline result<rate_info>
rate(const hopf_opts &o, hopf_scratch &sc) noexcept
{
  const __hopf::enc_params p = __hopf::resolve(o, o.m);
  rate_info ri{};
  ri.m = p.m;
  ri.dim_log2 = p.dim_log2;
  ri.dq = p.dq;
  ri.gain_bits = p.gain_bits;
  if ( p.m == mode::quotient ) {
    const max_t r = sc.build_s2(p.dq);
    if ( r < 0 ) [[unlikely]]
      return as_error(r);
    ri.shape_bits = static_cast<u32>(micron::bit_width(sc.s2.m_total - 1));
  } else if ( p.m == mode::quat || p.m == mode::oct ) {
    const max_t r = sc.build_susp(p.m == mode::quat ? 2 : 3, p.dq);
    if ( r < 0 ) [[unlikely]]
      return as_error(r);
    //  quat may read m_mod as exact: max quat M = 2^58.065 at dq_min (< 2^64; see susp.hpp:28,39)
    ri.shape_bits = p.m == mode::quat ? static_cast<u32>(micron::bit_width(sc.ss.m_mod - 1)) : susp_bits(sc.sp);
  } else {
    const max_t r = sc.build_tree(p.dim_log2, p.dq);
    if ( r < 0 ) [[unlikely]]
      return as_error(r);
    ri.shape_bits = hsc::shape_bits(sc.sk, sc.pt);
  }
  ri.record_bits = ri.gain_bits + ri.shape_bits;
  ri.block_elems = static_cast<u32>(__block_elems(p.m, p.dim_log2));
  ri.bits_per_elem = static_cast<f64>(ri.record_bits) / static_cast<f64>(ri.block_elems);
  ri.bits_per_input_byte = p.m == mode::bin ? ri.bits_per_elem : ri.bits_per_elem * 0.25;
  //  frame-FREE asymptote (record bits only); ratio_of() below is the framed number and differs
  //  for small inputs by the 48-byte header+trailer share
  ri.ratio = ri.bits_per_input_byte > 0.0 ? 8.0 / ri.bits_per_input_byte : 0.0;
  return ri;
}

constexpr usize
input_bytes(usize n_elems, const hopf_opts &o) noexcept
{
  return o.m == mode::bin ? n_elems : n_elems * 4;
}

//  framed ratio: input bytes over bound() INCLUDING the 40+8-byte frame (rate_info.ratio is the
//  frame-free asymptote)
inline f64
ratio_of(usize n_elems, const hopf_opts &o, hopf_scratch &sc) noexcept
{
  const usize z = bound(n_elems, o, sc);
  return z == 0 ? 0.0 : static_cast<f64>(input_bytes(n_elems, o)) / static_cast<f64>(z);
}

inline bool
degenerate(const hopf_opts &o, hopf_scratch &sc) noexcept
{
  const result<rate_info> r = rate(o, sc);
  return r.is_first() ? r.cast<rate_info>().shape_bits == 0 : false;
}

};      //  namespace hsc
