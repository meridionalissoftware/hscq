//  The typed vector layer (shape-gain blocks) and the scratch cache under it. Guards: the
//  documented per-block L2 error bound, NaN/Inf rejection, exact-zero blocks, the
//  reconstruct->requantize fixed point, the unit lane's no-gain contract, and scratch reuse
//  across parameter changes.

#include "../src/hsc/codec/block.hpp"
#include "../src/hsc/codec/scratch.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

int
main()
{
  hsc::hopf_scratch sc;

  sb::test_case("scratch builds and reuses; rebuild on parameter change");
  {
    sb::require(sc.build_tree(3, hsc::level_dq(5)) >= 0);
    const auto *nodes_before = sc.sk.nodes;
    sb::require(sc.build_tree(3, hsc::level_dq(5)) >= 0);      //  hot: no rebuild
    sb::require(sc.sk.nodes == nodes_before);
    sb::require(sc.build_tree(4, hsc::level_dq(6)) >= 0);      //  different key: rebuilt
    sb::require(sc.built_dim, 4u);
    sb::require(sc.build_tree(3, hsc::level_dq(5)) >= 0);      //  and back
    sb::require(sc.built_dim, 3u);
  }

  sb::test_case("error envelopes: worst <= 2.5 g d + step, RMSE <= 1.5 g_rms d (see block.hpp)");
  {
    //  SCHF packs (codewords >= d apart) but does not cover: uneven half-energy splits clamp to
    //  the leaf fan edge and stack down the recursion. Measured: worst ~ 2.1 g d, RMSE ~ 0.8 g d
    //  at dim 8. These envelopes hold that behavior in place without pretending a covering bound.
    sb::require(sc.build_tree(3, hsc::level_dq(5)) >= 0);      //  dim 8, d = 0.3
    const hsc::gain_quant gq{ 8, 8.0f };
    const f64 d = hsc::d_of(hsc::level_dq(5));
    const f64 step = 8.0 / 255.0;
    tutil::rng g;
    f64 se = 0, sg = 0;
    for ( i32 t = 0; t < 5000; ++t ) {
      f32 x[8], back[8];
      for ( u32 c = 0; c < 8; ++c ) x[c] = static_cast<f32>(g.unit() * 2.0);
      hsc::block_code bc{};
      sb::require(hsc::quantize_block(sc.sk, gq, x, bc) >= 0);
      hsc::reconstruct_block(sc.sk, gq, bc, back);
      f64 e2 = 0, g2 = 0;
      for ( u32 c = 0; c < 8; ++c ) {
        const f64 e = static_cast<f64>(back[c]) - static_cast<f64>(x[c]);
        e2 += e * e;
        g2 += static_cast<f64>(x[c]) * static_cast<f64>(x[c]);
      }
      const f64 gn = __builtin_sqrt(g2);
      sb::require(__builtin_sqrt(e2) <= gn * d * 2.5 + step + 1e-9);
      se += e2;
      sg += g2;
    }
    sb::require(__builtin_sqrt(se / sg) <= d * 1.5);
  }

  sb::test_case("NaN and Inf anywhere in a block are rejected as bad_value");
  {
    const hsc::gain_quant gq{ 8, 8.0f };
    f32 x[8] = { 0.5f, -0.25f, 0.125f, 1.0f, 0.75f, -0.5f, 0.25f, -1.0f };
    hsc::block_code bc{};
    x[3] = hsc::__u2f(0x7FC00000u);      //  qNaN
    sb::require(hsc::as_error(hsc::quantize_block(sc.sk, gq, x, bc)) == hsc::error::bad_value);
    x[3] = hsc::__u2f(0x7F800000u);      //  +Inf
    sb::require(hsc::as_error(hsc::quantize_block(sc.sk, gq, x, bc)) == hsc::error::bad_value);
  }

  sb::test_case("the zero block reconstructs to exact zeros through gain_q == 0");
  {
    const hsc::gain_quant gq{ 8, 8.0f };
    f32 x[8]{}, back[8];
    hsc::block_code bc{};
    sb::require(hsc::quantize_block(sc.sk, gq, x, bc) >= 0);
    sb::require(bc.gain_q, 0u);
    hsc::reconstruct_block(sc.sk, gq, bc, back);
    for ( u32 c = 0; c < 8; ++c ) sb::require(hsc::__f2u(back[c]), 0u);
  }

  sb::test_case("reconstruct -> requantize is a fixed point (gain and shape)");
  {
    const hsc::gain_quant gq{ 8, 8.0f };
    tutil::rng g;
    for ( i32 t = 0; t < 3000; ++t ) {
      f32 x[8], back[8];
      for ( u32 c = 0; c < 8; ++c ) x[c] = static_cast<f32>(g.unit() * 2.0);
      hsc::block_code b0{}, b1{};
      sb::require(hsc::quantize_block(sc.sk, gq, x, b0) >= 0);
      hsc::reconstruct_block(sc.sk, gq, b0, back);
      sb::require(hsc::quantize_block(sc.sk, gq, back, b1) >= 0);
      sb::require(b1.gain_q, b0.gain_q);
      if ( b0.gain_q ) {
        for ( u32 i = 0; i < hsc::tree_inode_count(3); ++i ) sb::require(b1.shape.leaf[i], b0.shape.leaf[i]);
        for ( u32 i = 0; i < hsc::tree_bnode_count(3); ++i ) sb::require(b1.shape.base[i], b0.shape.base[i]);
      }
    }
  }

  sb::test_case("unit lane: quantizes without a gain, rejects the zero block");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 2000; ++t ) {
      f32 x[8];
      f64 n2 = 0;
      for ( u32 c = 0; c < 8; ++c ) {
        x[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(x[c]) * static_cast<f64>(x[c]);
      }
      if ( n2 < 1e-6 ) continue;
      const f64 n = __builtin_sqrt(n2);
      for ( u32 c = 0; c < 8; ++c ) x[c] = static_cast<f32>(static_cast<f64>(x[c]) / n);
      hsc::tree_fields f{};
      sb::require(hsc::quantize_unit(sc.sk, x, f) >= 0);
      f32 back[8];
      hsc::reconstruct_unit(sc.sk, f, back);
      f64 bn = 0;
      for ( u32 c = 0; c < 8; ++c ) bn += static_cast<f64>(back[c]) * static_cast<f64>(back[c]);
      sb::require(bn > 1.0 - 1e-6 && bn < 1.0 + 1e-6);
      //  inside the same non-covering envelope as the gain lane (see block.hpp)
      f64 e2 = 0;
      for ( u32 c = 0; c < 8; ++c ) {
        const f64 e = static_cast<f64>(back[c]) - static_cast<f64>(x[c]);
        e2 += e * e;
      }
      sb::require(__builtin_sqrt(e2) <= hsc::d_of(hsc::level_dq(5)) * 2.5 + 1e-6);
    }
    f32 z[8]{};
    hsc::tree_fields f{};
    sb::require(hsc::as_error(hsc::quantize_unit(sc.sk, z, f)) == hsc::error::bad_value);
  }

  return 1;
}
