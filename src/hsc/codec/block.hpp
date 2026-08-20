//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"
#include "../error.hpp"
#include "../sphere/tree.hpp"
#include "gain.hpp"

#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  gain = ||v|| through the scalar quantizer;
//  shape = the SCHF fields of v (scale invariant);
//  zero gain reconstructs exact zeros;
//  fixed rate layout
//
//  SCHF guarantees codewords >= d apart; it does not bound the distance from an arbitrary vector to the nearest codeword

namespace hsc
{

struct block_code {
  u32 gain_q = 0;
  tree_fields shape{};
};

//  f64
constexpr void
quantize_vec(const tree_skeleton &sk, const gain_quant &gq, const f64 *v, block_code &out, u32 refine = 0) noexcept
{
  const u32 n = 1u << sk.dim_log2;
  f64 s = 0;
  for ( u32 c = 0; c < n; ++c ) s += v[c] * v[c];
  const f64 g = __sqrt(s);
  out.gain_q = gq_encode(gq, g);
  if ( out.gain_q == 0 ) [[unlikely]] {
    //  zero block never walks the tree, zero out fields explicitly
    out.shape = tree_fields{};
    return;
  }
  //  no pre-clear
  tree_quantize(sk, v, out.shape, refine);
}

constexpr void
reconstruct_vec(const tree_skeleton &sk, const gain_quant &gq, const block_code &in, f64 *v) noexcept
{
  const u32 n = 1u << sk.dim_log2;
  if ( in.gain_q == 0 ) {
    for ( u32 c = 0; c < n; ++c ) v[c] = 0.0;
    return;
  }
  const f64 g = gq_decode(gq, in.gain_q);
  tree_decode(sk, in.shape, v);
  for ( u32 c = 0; c < n; ++c ) v[c] *= g;
}

//  f32, with validation
//  NaN/Inf anywhere -> fail(bad_value)
constexpr max_t
quantize_block(const tree_skeleton &sk, const gain_quant &gq, const f32 *x, block_code &out, u32 refine = 0) noexcept
{
  const u32 n = 1u << sk.dim_log2;
  f64 v[64];
  for ( u32 c = 0; c < n; ++c ) {
    const f64 e = static_cast<f64>(x[c]);
    if ( micron::math::isnan(e) || micron::math::isinf(e) ) [[unlikely]]
      return fail(error::bad_value);
    v[c] = e;
  }
  quantize_vec(sk, gq, v, out, refine);
  return 0;
}

constexpr void
reconstruct_block(const tree_skeleton &sk, const gain_quant &gq, const block_code &in, f32 *x) noexcept
{
  const u32 n = 1u << sk.dim_log2;
  f64 v[64];      //  reconstruct_vec writes v[0..n)
  reconstruct_vec(sk, gq, in, v);
  for ( u32 c = 0; c < n; ++c ) x[c] = static_cast<f32>(v[c]);
}

//  no gain field; a zero (or non-finite) block cannot be represented
constexpr max_t
quantize_unit(const tree_skeleton &sk, const f32 *x, tree_fields &out, u32 refine = 0) noexcept
{
  const u32 n = 1u << sk.dim_log2;
  f64 v[64];
  f64 s = 0;
  for ( u32 c = 0; c < n; ++c ) {
    const f64 e = static_cast<f64>(x[c]);
    if ( micron::math::isnan(e) || micron::math::isinf(e) ) [[unlikely]]
      return fail(error::bad_value);
    v[c] = e;
    s += e * e;
  }
  if ( !(s > 0.0) ) [[unlikely]]
    return fail(error::bad_value);
  tree_quantize(sk, v, out, refine);
  return 0;
}

constexpr void
reconstruct_unit(const tree_skeleton &sk, const tree_fields &in, f32 *x) noexcept
{
  const u32 n = 1u << sk.dim_log2;
  f64 v[64];      //  tree_decode writes v[0..n)
  tree_decode(sk, in, v);
  for ( u32 c = 0; c < n; ++c ) x[c] = static_cast<f32>(v[c]);
}

};      //  namespace hsc
