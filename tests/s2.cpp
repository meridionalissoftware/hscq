// The S^2 latitude-band quantizer behind quotient mode. Guards: golden cardinalities, full
// index bijection sweeps, brute-force minimum pairwise distance on whole codebooks, pole
// determinism (degenerate m=1 bands ignore longitude), and off-sphere robustness of the
// acos clamp. The Hopf projection/lift pair that feeds this quantizer is tested in
// quotient.cpp, not here.

#include "../src/hsc/sphere/s2.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

constexpr u32 k_cap = 256;

constexpr u64
ct_m(u32 dq)
{
  hsc::s2_band buf[k_cap]{};
  return hsc::s2_build(dq, buf).m_total;
}

// golden cardinalities from the validated prototype, at the exact wire d_q values
static_assert(ct_m(hsc::level_dq(2)) == 22);        // d = 0.7
static_assert(ct_m(hsc::level_dq(3)) == 46);        // d = 0.5
static_assert(ct_m(hsc::level_dq(5)) == 132);       // d = 0.3
static_assert(ct_m(hsc::level_dq(7)) == 1236);      // d = 0.1

static hsc::s2_band g_bd[hsc::s2_max_bands];

}      // namespace

int
main()
{
  sb::test_case("full bijection sweep: quantize(decode(a)) == a for every index");
  {
    for ( i32 lvl : { 1, 2, 3, 4, 5, 6, 7 } ) {
      const hsc::s2_skeleton sk = hsc::s2_build(hsc::level_dq(lvl), g_bd);
      for ( u64 a = 0; a < sk.m_total; ++a ) {
        f64 p[3]{};
        hsc::s2_decode(sk, a, p);
        sb::require(hsc::s2_quantize(sk, p), a);
      }
    }
  }

  sb::test_case("brute-force minimum pairwise distance >= d on whole codebooks");
  {
    for ( i32 lvl : { 2, 3, 5 } ) {      // M = 22 / 46 / 132
      const hsc::s2_skeleton sk = hsc::s2_build(hsc::level_dq(lvl), g_bd);
      const u64 M = sk.m_total;
      static f64 pts[132][3];
      for ( u64 a = 0; a < M; ++a ) hsc::s2_decode(sk, a, pts[a]);
      f64 dmin2 = 16.0;
      for ( u64 a = 0; a < M; ++a )
        for ( u64 b = a + 1; b < M; ++b ) {
          f64 s = 0;
          for ( i32 c = 0; c < 3; ++c ) {
            const f64 t = pts[a][c] - pts[b][c];
            s += t * t;
          }
          if ( s < dmin2 ) dmin2 = s;
        }
      sb::require(__builtin_sqrt(dmin2) >= sk.d - 1e-9);
    }
  }

  sb::test_case("decoded codewords are unit vectors");
  {
    const hsc::s2_skeleton sk = hsc::s2_build(hsc::level_dq(7), g_bd);
    tutil::rng g;
    for ( i32 t = 0; t < 2000; ++t ) {
      f64 p[3]{};
      hsc::s2_decode(sk, g.next() % sk.m_total, p);
      const f64 n2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
      sb::require(n2 > 1.0 - 1e-12 && n2 < 1.0 + 1e-12);
    }
  }

  sb::test_case("poles quantize deterministically regardless of longitude");
  {
    const hsc::s2_skeleton sk = hsc::s2_build(hsc::level_dq(3), g_bd);
    const f64 north[3] = { 0.0, 0.0, 1.0 };
    const f64 south[3] = { 0.0, 0.0, -1.0 };
    const u64 an = hsc::s2_quantize(sk, north);
    const u64 as = hsc::s2_quantize(sk, south);
    sb::require(an < sk.m_total);
    sb::require(as < sk.m_total);
    sb::require(an != as);
    // a whisper off the pole in any longitude still lands on the same codeword neighborhood
    for ( i32 t = 0; t < 16; ++t ) {
      const f64 phi = static_cast<f64>(t) * hsc::k_2pi / 16.0;
      const f64 p[3] = { 1e-9 * micron::cos(phi), 1e-9 * micron::sin(phi), 1.0 };
      sb::require(hsc::s2_quantize(sk, p), an);
    }
  }

  sb::test_case("off-sphere inputs clamp instead of NaN");
  {
    const hsc::s2_skeleton sk = hsc::s2_build(hsc::level_dq(3), g_bd);
    const f64 over[3] = { 0.0, 0.0, 1.5 };
    const f64 under[3] = { 0.0, 0.0, -1.5 };
    sb::require(hsc::s2_quantize(sk, over) < sk.m_total);
    sb::require(hsc::s2_quantize(sk, under) < sk.m_total);
  }

  return 1;
}
