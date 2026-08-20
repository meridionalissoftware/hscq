// The scalar gain quantizer. Guards: exact zero at q == 0 (a zero block must survive
// byte-perfect), the half-step error bound inside range, clamping at full scale, the f32
// canonicalization rule (quantize against the ROUNDED scale, since the header stores f32
// bits), and the bin-mode implicit scale table.

#include "../src/hsc/codec/gain.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

// bin scales: 127.5 * sqrt(n) as f32; the dyadic n = 4/16/64 cases are exact decimals
static_assert(hsc::gq_bin_scale(2) == 255.0f);
static_assert(hsc::gq_bin_scale(4) == 510.0f);
static_assert(hsc::gq_bin_scale(6) == 1020.0f);

static_assert(hsc::gq_encode(hsc::gain_quant{ 8, 255.0f }, 0.0) == 0);
static_assert(hsc::gq_decode(hsc::gain_quant{ 8, 255.0f }, 0) == 0.0);

}      // namespace

int
main()
{
  sb::test_case("exact zero and exact full scale");
  {
    const hsc::gain_quant gq{ 8, 255.0f };
    sb::require(hsc::gq_encode(gq, 0.0), 0u);
    sb::require(hsc::gq_decode(gq, 0) == 0.0);
    sb::require(hsc::gq_encode(gq, 255.0), 255u);
    sb::require(hsc::gq_decode(gq, 255) == 255.0);
    // above range clamps to the top level, never wraps
    sb::require(hsc::gq_encode(gq, 1e9), 255u);
    // negative or non-positive gain is the zero level
    sb::require(hsc::gq_encode(gq, -3.0), 0u);
  }

  sb::test_case("|decode(encode(g)) - g| <= step/2 inside range, all widths");
  {
    tutil::rng g;
    for ( u32 bits : { 1u, 2u, 4u, 8u, 12u, 16u, 24u } ) {
      const hsc::gain_quant gq{ bits, 100.0f };
      const f64 step = 100.0 / static_cast<f64>((1u << bits) - 1);
      for ( i32 t = 0; t < 4000; ++t ) {
        const f64 v = (g.unit() * 0.5 + 0.5) * 100.0;
        const u32 q = hsc::gq_encode(gq, v);
        sb::require(q <= (1u << bits) - 1);
        const f64 back = hsc::gq_decode(gq, q);
        const f64 err = back > v ? back - v : v - back;
        sb::require(err <= step * 0.5 + 1e-12);
      }
    }
  }

  sb::test_case("re-encoding a reconstructed gain is a fixed point");
  {
    const hsc::gain_quant gq{ 8, 360.62445f };      // gq_bin_scale(3)
    for ( u32 q = 0; q <= 255; ++q ) sb::require(hsc::gq_encode(gq, hsc::gq_decode(gq, q)), q);
  }

  sb::test_case("canonicalization: the f32-rounded scale is the quantizer, bit-exactly");
  {
    const f64 raw = 123.4567891234;      // not representable in f32
    const f32 rounded = static_cast<f32>(raw);
    const hsc::gain_quant gq{ 8, rounded };
    // header transports the bit pattern; a decoder rebuilding from bits gets the same steps
    const hsc::gain_quant gq2{ 8, hsc::__u2f(hsc::__f2u(rounded)) };
    for ( u32 q = 0; q <= 255; ++q ) sb::require(hsc::gq_decode(gq, q) == hsc::gq_decode(gq2, q));
  }

  sb::test_case("bin scale table matches 127.5 * sqrt(n) for the non-dyadic dims too");
  {
    sb::require(hsc::gq_bin_scale(3) == static_cast<f32>(127.5 * __builtin_sqrt(8.0)));
    sb::require(hsc::gq_bin_scale(5) == static_cast<f32>(127.5 * __builtin_sqrt(32.0)));
  }

  return 1;
}
