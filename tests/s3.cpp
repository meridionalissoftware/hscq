//  The 4D SCHF core. Guards: golden cardinalities (pinned at d_eff = d_q/2^24, cross-checked
//  against the validated Python prototype), FULL index bijection sweeps, brute-force minimum
//  pairwise distance on whole codebooks, the eq. (34) typo regression, degenerate-leaf
//  construction, refine monotonicity -- and the drift tripwire: the runtime-built skeleton's
//  stream-defining INTEGERS must equal the consteval-built ones (micron kernels vs gcc
//  constant folds may differ in f64 ulps; they may never differ in an m, n, offset or M).

#include "../src/hsc/sphere/s3.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

constexpr u32 k_small = 160;      //  leaf capacity covering every preset (worst: level 9 -> 79 leaves)

constexpr u64
ct_m(u32 dq)
{
  hsc::s3_leaf buf[k_small]{};
  return hsc::s3_build(dq, buf).m_total;
}

//  golden cardinalities: the consteval path must reproduce the validated values exactly
static_assert(ct_m(hsc::level_dq(1)) == 16);
static_assert(ct_m(hsc::level_dq(2)) == 52);
static_assert(ct_m(hsc::level_dq(3)) == 138);
static_assert(ct_m(hsc::level_dq(4)) == 284);
static_assert(ct_m(hsc::level_dq(5)) == 736);
static_assert(ct_m(hsc::level_dq(6)) == 2588);
static_assert(ct_m(hsc::level_dq(7)) == 21844);
static_assert(ct_m(hsc::level_dq(8)) == 178758);
static_assert(ct_m(hsc::level_dq(9)) == 2828294);

//  consteval-baked integer skeletons for the drift tripwire
struct ct_table {
  u32 count = 0;
  u32 m[k_small]{};
  u32 n[k_small]{};
  u64 off[k_small]{};
  u64 m_total = 0;
};

constexpr ct_table
bake(u32 dq)
{
  hsc::s3_leaf buf[k_small]{};
  const hsc::s3_skeleton sk = hsc::s3_build(dq, buf);
  ct_table t{};
  t.count = sk.count;
  t.m_total = sk.m_total;
  for ( u32 i = 0; i < sk.count; ++i ) {
    t.m[i] = buf[i].m;
    t.n[i] = buf[i].n;
    t.off[i] = buf[i].off;
  }
  return t;
}

constexpr u32 k_levels[9] = { hsc::level_dq(1), hsc::level_dq(2), hsc::level_dq(3), hsc::level_dq(4), hsc::level_dq(5),
                              hsc::level_dq(6), hsc::level_dq(7), hsc::level_dq(8), hsc::level_dq(9) };
constexpr ct_table k_baked[9] = { bake(k_levels[0]), bake(k_levels[1]), bake(k_levels[2]), bake(k_levels[3]), bake(k_levels[4]),
                                  bake(k_levels[5]), bake(k_levels[6]), bake(k_levels[7]), bake(k_levels[8]) };

//  the paper's literal eq. (34): xi1_hat = j*Dxi1 + k*Dxi2 -- kept only to prove it wrong
u64
quantize_paper_literal(const hsc::s3_skeleton &sk, const f64 *y)
{
  const f64 na = __builtin_sqrt(y[0] * y[0] + y[1] * y[1]);
  const f64 nb = __builtin_sqrt(y[2] * y[2] + y[3] * y[3]);
  const f64 eta = micron::atan2(nb, na);
  const i64 half = static_cast<i64>(sk.half);
  i64 i = static_cast<i64>(__builtin_round((eta - hsc::k_pi_4) / sk.deta));
  i = i < -half ? -half : (i > half ? half : i);
  const hsc::s3_leaf &lf = sk.lv[static_cast<u32>(i + half)];
  f64 xi1 = micron::atan2(y[1], y[0]);
  if ( xi1 < 0.0 ) xi1 += hsc::k_2pi;
  f64 xi2 = micron::atan2(y[3], y[2]);
  if ( xi2 < 0.0 ) xi2 += hsc::k_2pi;
  const f64 dxi1 = hsc::k_2pi / static_cast<f64>(lf.m);
  const f64 dxi2 = hsc::k_2pi / static_cast<f64>(lf.n);
  const i64 k = static_cast<i64>(__builtin_round(xi2 / dxi2)) % static_cast<i64>(lf.n);
  const i64 jr = static_cast<i64>(__builtin_round((xi1 - static_cast<f64>(k) * dxi2 * 0.5) / dxi1)) % static_cast<i64>(lf.m);
  const i64 j = jr < 0 ? jr + static_cast<i64>(lf.m) : jr;
  return lf.off + static_cast<u64>(k) * lf.m + static_cast<u64>(j);
}

static hsc::s3_leaf g_lv[hsc::s3_max_leaves];

}      //  namespace

int
main()
{
  sb::test_case("runtime skeleton integers == consteval skeleton integers (drift tripwire)");
  {
    for ( u32 li = 0; li < 9; ++li ) {
      const hsc::s3_skeleton sk = hsc::s3_build(k_levels[li], g_lv);
      sb::require(sk.count == k_baked[li].count);
      sb::require(sk.m_total == k_baked[li].m_total);
      for ( u32 i = 0; i < sk.count; ++i ) {
        sb::require(g_lv[i].m == k_baked[li].m[i]);
        sb::require(g_lv[i].n == k_baked[li].n[i]);
        sb::require(g_lv[i].off == k_baked[li].off[i]);
      }
    }
  }

  sb::test_case("full bijection sweep: quantize(decode(a)) == a for every index");
  {
    for ( i32 lvl : { 1, 2, 3, 4, 5, 6 } ) {
      const hsc::s3_skeleton sk = hsc::s3_build(hsc::level_dq(lvl), g_lv);
      for ( u64 a = 0; a < sk.m_total; ++a ) {
        f64 p[4]{};
        hsc::s3_decode(sk, a, p);
        sb::require(hsc::s3_quantize(sk, p), a);
      }
    }
  }

  sb::test_case("brute-force minimum pairwise distance >= d on whole codebooks");
  {
    for ( i32 lvl : { 1, 2, 3, 4, 5 } ) {      //  M up to 736
      const hsc::s3_skeleton sk = hsc::s3_build(hsc::level_dq(lvl), g_lv);
      const u64 M = sk.m_total;
      static f64 pts[736][4];
      for ( u64 a = 0; a < M; ++a ) hsc::s3_decode(sk, a, pts[a]);
      f64 dmin2 = 16.0;
      for ( u64 a = 0; a < M; ++a )
        for ( u64 b = a + 1; b < M; ++b ) {
          f64 s = 0;
          for ( i32 c = 0; c < 4; ++c ) {
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
    const hsc::s3_skeleton sk = hsc::s3_build(hsc::level_dq(6), g_lv);
    tutil::rng g;
    for ( i32 t = 0; t < 2000; ++t ) {
      f64 p[4]{};
      hsc::s3_decode(sk, g.next() % sk.m_total, p);
      const f64 n2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3];
      sb::require(n2 > 1.0 - 1e-12 && n2 < 1.0 + 1e-12);
    }
  }

  sb::test_case("eq. (34) as printed breaks round-trips; the corrected form does not");
  {
    const hsc::s3_skeleton sk = hsc::s3_build(hsc::level_dq(3), g_lv);      //  d = 0.5, M = 138
    u64 broken = 0;
    for ( u64 a = 0; a < sk.m_total; ++a ) {
      f64 p[4]{};
      hsc::s3_decode(sk, a, p);
      if ( quantize_paper_literal(sk, p) != a ) ++broken;
    }
    sb::require(broken > 0);      //  Python prototype measured 62/138; any nonzero proves the typo
  }

  sb::test_case("degenerate leaves (eta reaching 0 / pi/2) build and round-trip");
  {
    const u32 dq = hsc::dq_of(2.0 * micron::sin(hsc::k_pi / 16.0));      //  d ~ 0.39018
    const hsc::s3_skeleton sk = hsc::s3_build(dq, g_lv);
    sb::require(sk.m_total > 0);
    for ( u64 a = 0; a < sk.m_total; ++a ) {
      f64 p[4]{};
      hsc::s3_decode(sk, a, p);
      sb::require(hsc::s3_quantize(sk, p), a);
    }
  }

  sb::test_case("d clamps: dq at/above 2.0 gives the single-codeword code");
  {
    const hsc::s3_skeleton sk = hsc::s3_build(hsc::dq_max, g_lv);
    sb::require(sk.m_total, 1ull);
    sb::require(sk.count, 1u);
    f64 p[4]{};
    hsc::s3_decode(sk, 0, p);
    sb::require(hsc::s3_quantize(sk, p), 0ull);
  }

  sb::test_case("refine never loses to the plain quantizer (candidate set includes it)");
  {
    const hsc::s3_skeleton sk = hsc::s3_build(hsc::level_dq(4), g_lv);
    tutil::rng g;
    for ( i32 t = 0; t < 4000; ++t ) {
      f64 y[4] = { g.unit(), g.unit(), g.unit(), g.unit() };
      const f64 n2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
      if ( n2 < 1e-12 ) continue;
      f64 p0[4]{}, p1[4]{};
      hsc::s3_decode(sk, hsc::s3_quantize(sk, y, 0), p0);
      hsc::s3_decode(sk, hsc::s3_quantize(sk, y, 1), p1);
      const f64 d0 = p0[0] * y[0] + p0[1] * y[1] + p0[2] * y[2] + p0[3] * y[3];
      const f64 d1 = p1[0] * y[0] + p1[1] * y[1] + p1[2] * y[2] + p1[3] * y[3];
      sb::require(d1 >= d0 - 1e-12);
    }
  }

  return 1;
}
