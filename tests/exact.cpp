// The bit-exactness gate. Every fast or vectorized path in hsc carries a scalar constexpr twin,
// because tests/comptime.cpp asserts consteval streams are byte-identical to runtime streams in
// BOTH directions -- so a twin that differs by one ulp is a format bug, not a rounding detail.
// This file pins each twin against its reference directly, one level below comptime.cpp, so a
// mismatch names the kernel instead of surfacing as a mysterious stream diff.
//
// Currently pinned:
//   __round        ==  __builtin_round  (the libm call it replaces), over the ties, the sign
//                                        cases, the specials, and hsc's live argument domain.
//   __sqrt         ==  __builtin_sqrt   (consteval == runtime, and both == the correctly-rounded
//                                        reference). __sqrt exists because __builtin_sqrt is NOT a
//                                        bare sqrtsd under duck's flags: with -fmath-errno live it
//                                        carries an errno guard and a TAIL CALL to libm, inside the
//                                        codec's inner loops. IEEE-754 sqrt is uniquely defined, so
//                                        the consteval fold and the hardware instruction must agree.
//   atan2_bl_x4    ==  4x atan2_bl      (the packed quantizer kernel vs its consteval twin)
//   atan2_bl       ==  micron::atan2    (so the branchless rewrite changed no stream at all)
//   mk::sincos     ==  micron::sincos   (packed decode kernel vs its consteval twin, on [0,2pi))
//   quat_mul/oct_mul consteval == runtime (the -ffp-contract tripwire: the products are pinned
//                                        fused-op chains, so contraction can never split the sides)
//
// The last three are what let tests/comptime.cpp keep asserting consteval == runtime through a
// vectorized codec: if any of them drifts by one ulp, a stream silently changes shape.
//
// The test points are built with micron::math::ldexp / nextafter, not the __builtin_ forms: gcc
// expands __builtin_round inline (roundsd) but lowers those two to libm CALLS, and the suite links
// -ffreestanding -nostdlib against a corelib that deliberately exports no such symbols
// (micron/math/__gcc_math_syms.hpp says so in as many words). Both are exact bit manipulations, so
// the generated arguments are identical either way -- this is scaffolding, never the thing pinned.

#include "../src/hsc/codec/oct.hpp"
#include "../src/hsc/config.hpp"
#include "../src/hsc/sphere/s3.hpp"
#include "tutil.hpp"

#include <micron/math/mk.hpp>
#include <micron/math/simd/atrig.hpp>
#include <micron/math/simd/trig.hpp>

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

constexpr u64
ubits(f64 v) noexcept
{
  return __builtin_bit_cast(u64, v);
}

// consteval side: __round must fold to __builtin_round exactly (this is the twin contract)
static_assert(ubits(hsc::__round(0.5)) == ubits(1.0));
static_assert(ubits(hsc::__round(-0.5)) == ubits(-1.0));
static_assert(ubits(hsc::__round(1.5)) == ubits(2.0));
static_assert(ubits(hsc::__round(2.5)) == ubits(3.0));      // away from zero, NOT banker's
static_assert(ubits(hsc::__round(-2.5)) == ubits(-3.0));
static_assert(ubits(hsc::__round(0.49999999999999994)) == ubits(0.0));      // the classic +0.5 trap
static_assert(ubits(hsc::__round(0.0)) == ubits(0.0));
static_assert(ubits(hsc::__round(-0.0)) == ubits(-0.0));      // sign of zero is preserved
static_assert(ubits(hsc::__round(-0.25)) == ubits(-0.0));

// consteval side: __sqrt must fold exactly. IEEE-754 square root is correctly rounded and unique,
// so there is only one right answer and both sides of the twin have to produce it.
static_assert(ubits(hsc::__sqrt(0.0)) == ubits(0.0));
static_assert(ubits(hsc::__sqrt(-0.0)) == ubits(-0.0));      // sign of zero is preserved
static_assert(hsc::__sqrt(1.0) == 1.0);
static_assert(hsc::__sqrt(4.0) == 2.0);
static_assert(hsc::__sqrt(2.25) == 1.5);
static_assert(hsc::__sqrt(1e10) == 100000.0);
static_assert(ubits(hsc::__sqrt(2.0)) == ubits(__builtin_sqrt(2.0)));      // the irrational case

u64 g_seed = 0x9E3779B97F4A7C15ull;

u64
xs() noexcept
{
  g_seed ^= g_seed << 13;
  g_seed ^= g_seed >> 7;
  g_seed ^= g_seed << 17;
  return g_seed;
}

// consteval-baked quat/oct products over a fixed integer-derived grid: main() recomputes the
// SAME rows at runtime and requires bit identity. This is the localized -ffp-contract tripwire:
// the build passes no contraction flag, consteval never contracts, and the product kernels are
// written as explicit fused-op chains precisely so both sides round identically. If a compiler
// or flag change ever splits them, this case names the kernel instead of a mysterious stream diff.
inline constexpr usize k_alg_rows = 64;

struct alg_baked {
  f64 qa[k_alg_rows][4];
  f64 qb[k_alg_rows][4];
  f64 qo[k_alg_rows][4];
  f64 oa[k_alg_rows][8];
  f64 ob[k_alg_rows][8];
  f64 oo[k_alg_rows][8];
};

consteval alg_baked
bake_alg()
{
  alg_baked t{};
  u64 s = 0xA5A5DEADBEEF1234ull;
  auto nx = [&]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    // integer-exact map to [-2, 2): identical consteval and runtime by construction
    return static_cast<f64>(s >> 11) * 0x1p-53 * 4.0 - 2.0;
  };
  for ( usize r = 0; r < k_alg_rows; ++r ) {
    for ( u32 k = 0; k < 4; ++k ) t.qa[r][k] = nx();
    for ( u32 k = 0; k < 4; ++k ) t.qb[r][k] = nx();
    hsc::quat_mul(t.qa[r], t.qb[r], t.qo[r]);
    for ( u32 k = 0; k < 8; ++k ) t.oa[r][k] = nx();
    for ( u32 k = 0; k < 8; ++k ) t.ob[r][k] = nx();
    hsc::oct_mul(t.oa[r], t.ob[r], t.oo[r]);
  }
  return t;
}

inline constexpr alg_baked k_alg = bake_alg();

// consteval-baked __sqrt over a fixed integer-derived grid; main() recomputes the SAME rows at
// runtime and requires bit identity. __sqrt is an `if consteval` twin (the consteval branch folds
// __builtin_sqrt, the runtime branch issues vsqrtsd), so this is the case that actually pins the
// split -- a randomized sweep only ever exercises one side of it.
inline constexpr usize k_sqrt_rows = 512;

struct sqrt_baked {
  f64 in[k_sqrt_rows];
  f64 out[k_sqrt_rows];
};

consteval sqrt_baked
bake_sqrt()
{
  sqrt_baked t{};
  u64 s = 0xD1B54A32D192ED03ull;
  for ( usize r = 0; r < k_sqrt_rows; ++r ) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    // integer-exact map to [0, 4): identical consteval and runtime by construction
    t.in[r] = static_cast<f64>(s >> 11) * 0x1p-53 * 4.0;
    t.out[r] = hsc::__sqrt(t.in[r]);
  }
  return t;
}

inline constexpr sqrt_baked k_sqrt = bake_sqrt();

}      // namespace

int
main()
{
  sb::test_case("__round == __builtin_round: exact halves, both signs, every magnitude");
  {
    u64 bad = 0;
    for ( i64 k = -300000; k <= 300000; ++k ) {
      const f64 a = static_cast<f64>(k) + 0.5;
      const f64 b = static_cast<f64>(k) - 0.5;
      if ( ubits(hsc::__round(a)) != ubits(__builtin_round(a)) ) ++bad;
      if ( ubits(hsc::__round(b)) != ubits(__builtin_round(b)) ) ++bad;
    }
    sb::require(bad, 0ull);
  }

  sb::test_case("__round == __builtin_round: neighbours of every tie across the exponent range");
  {
    u64 bad = 0;
    // |x| < 2^52 is hsc's whole live domain: every round() argument in the codec is a ratio of
    // an angle to a grid step (bounded by a leaf/band count) or a byte-scale value.
    for ( i32 e = -20; e <= 51; ++e ) {
      for ( i64 k = -400; k <= 400; ++k ) {
        const f64 base = static_cast<f64>(k) * micron::math::ldexp<f64>(1.0, e) * 0.5;
        f64 x = base;
        for ( i32 d = 0; d < 3; ++d ) {
          if ( ubits(hsc::__round(x)) != ubits(__builtin_round(x)) ) ++bad;
          const f64 y = -x;
          if ( ubits(hsc::__round(y)) != ubits(__builtin_round(y)) ) ++bad;
          x = micron::math::nextafter<f64>(x, 1e308);
        }
        x = base;
        for ( i32 d = 0; d < 3; ++d ) {
          if ( ubits(hsc::__round(x)) != ubits(__builtin_round(x)) ) ++bad;
          x = micron::math::nextafter<f64>(x, -1e308);
        }
      }
    }
    sb::require(bad, 0ull);
  }

  sb::test_case("__round == __builtin_round: randomized sweep over hsc's argument domain");
  {
    u64 bad = 0;
    for ( u64 i = 0; i < 3000000; ++i ) {
      const f64 m = static_cast<f64>(static_cast<i64>(xs() >> 11)) / 9007199254740992.0 - 0.5;
      const f64 x = m * static_cast<f64>(1u << (xs() % 30u));
      if ( ubits(hsc::__round(x)) != ubits(__builtin_round(x)) ) ++bad;
    }
    sb::require(bad, 0ull);
  }

  sb::test_case("__round == __builtin_round: specials and already-integral values");
  {
    const f64 inf = __builtin_inf();
    sb::require(ubits(hsc::__round(inf)), ubits(__builtin_round(inf)));
    sb::require(ubits(hsc::__round(-inf)), ubits(__builtin_round(-inf)));
    sb::require(__builtin_isnan(hsc::__round(__builtin_nan(""))));
    // integral inputs must come back untouched, sign of zero included
    u64 bad = 0;
    for ( i64 k = -100000; k <= 100000; ++k ) {
      const f64 x = static_cast<f64>(k);
      if ( ubits(hsc::__round(x)) != ubits(__builtin_round(x)) ) ++bad;
    }
    sb::require(bad, 0ull);
    // the denormal floor and the smallest normals round to a signed zero, not to a trap
    sb::require(ubits(hsc::__round(5e-324)), ubits(__builtin_round(5e-324)));
    sb::require(ubits(hsc::__round(-5e-324)), ubits(__builtin_round(-5e-324)));
  }

  sb::test_case("__sqrt: consteval == runtime, bit for bit (the twin contract)");
  {
    u64 bad = 0;
    for ( usize r = 0; r < k_sqrt_rows; ++r )
      if ( ubits(hsc::__sqrt(k_sqrt.in[r])) != ubits(k_sqrt.out[r]) ) ++bad;
    sb::require(bad, 0ull);
  }

  sb::test_case("__sqrt == __builtin_sqrt over hsc's domain (sums of squares, so x >= 0)");
  {
    u64 bad = 0;
    for ( u64 i = 0; i < 2000000; ++i ) {
      const f64 m = static_cast<f64>(static_cast<i64>(xs() >> 11)) / 9007199254740992.0;
      const f64 x = m * micron::math::ldexp<f64>(1.0, static_cast<i32>(xs() % 80u) - 40);
      if ( ubits(hsc::__sqrt(x)) != ubits(__builtin_sqrt(x)) ) ++bad;
    }
    sb::require(bad, 0ull);

    // perfect squares must come back exact, not one ulp low
    for ( i64 k = 0; k <= 100000; ++k ) {
      const f64 x = static_cast<f64>(k);
      if ( ubits(hsc::__sqrt(x * x)) != ubits(x) ) ++bad;
    }
    sb::require(bad, 0ull);
  }

  sb::test_case("__sqrt: specials, signed zero, and the denormal floor");
  {
    const f64 inf = __builtin_inf();
    sb::require(ubits(hsc::__sqrt(0.0)), ubits(0.0));
    sb::require(ubits(hsc::__sqrt(-0.0)), ubits(-0.0));      // sqrt(-0) is -0, not +0
    sb::require(ubits(hsc::__sqrt(inf)), ubits(__builtin_sqrt(inf)));
    sb::require(__builtin_isnan(hsc::__sqrt(__builtin_nan(""))));
    sb::require(ubits(hsc::__sqrt(5e-324)), ubits(__builtin_sqrt(5e-324)));
    sb::require(ubits(hsc::__sqrt(2.2250738585072014e-308)), ubits(__builtin_sqrt(2.2250738585072014e-308)));
  }

  sb::test_case("packed atan2_bl_x4 == its scalar twin, bit for bit (random + edges)");
  {
    namespace mk = micron::math::mk;
    u64 bad = 0;
    f64 y[4], x[4], r[4];
    for ( u64 i = 0; i < 400000; ++i ) {
      for ( u32 k = 0; k < 4; ++k ) {
        const f64 my = static_cast<f64>(static_cast<i64>(xs() >> 11)) / 9007199254740992.0 - 0.5;
        const f64 mx = static_cast<f64>(static_cast<i64>(xs() >> 11)) / 9007199254740992.0 - 0.5;
        y[k] = my * micron::math::ldexp<f64>(1.0, static_cast<i32>(xs() % 60u) - 30);
        x[k] = mx * micron::math::ldexp<f64>(1.0, static_cast<i32>(xs() % 60u) - 30);
      }
      mk::atan2_bl_x4(y, x, r);
      for ( u32 k = 0; k < 4; ++k )
        if ( ubits(r[k]) != ubits(mk::atan2_bl(y[k], x[k])) ) ++bad;
    }
    sb::require(bad, 0ull);

    // every edge the packed kernel screens for, and mixtures of them inside one vector
    const f64 sp[10] = { 0.0, -0.0, 1.0, -1.0, __builtin_inf(), -__builtin_inf(), 1e300, -1e-300, 5e-324, -3.5 };
    u64 bad2 = 0;
    for ( u32 a = 0; a < 10; ++a )
      for ( u32 b = 0; b < 10; ++b )
        for ( u32 c = 0; c < 10; ++c ) {
          y[0] = sp[a];
          x[0] = sp[b];
          y[1] = sp[c];
          x[1] = sp[a];
          y[2] = sp[b];
          x[2] = sp[c];
          y[3] = sp[c];
          x[3] = sp[b];
          mk::atan2_bl_x4(y, x, r);
          for ( u32 k = 0; k < 4; ++k ) {
            const f64 e = mk::atan2_bl(y[k], x[k]);
            if ( __builtin_isnan(e) && __builtin_isnan(r[k]) ) continue;
            if ( ubits(r[k]) != ubits(e) ) ++bad2;
          }
        }
    sb::require(bad2, 0ull);
  }

  sb::test_case("branchless atan2_bl == micron::atan2 (the rewrite is stream-neutral)");
  {
    u64 bad = 0;
    for ( u64 i = 0; i < 1000000; ++i ) {
      const f64 my = static_cast<f64>(static_cast<i64>(xs() >> 11)) / 9007199254740992.0 - 0.5;
      const f64 mx = static_cast<f64>(static_cast<i64>(xs() >> 11)) / 9007199254740992.0 - 0.5;
      const f64 y = my * micron::math::ldexp<f64>(1.0, static_cast<i32>(xs() % 50u) - 25);
      const f64 x = mx * micron::math::ldexp<f64>(1.0, static_cast<i32>(xs() % 50u) - 25);
      if ( ubits(micron::math::mk::atan2_bl(y, x)) != ubits(micron::atan2(y, x)) ) ++bad;
    }
    sb::require(bad, 0ull);
  }

  sb::test_case("packed sincos == scalar sincos over [0, 2pi), the decoder's whole domain");
  {
    u64 bad = 0;
    for ( u64 i = 0; i < 400000; ++i ) {
      f64 a[2];
      a[0] = static_cast<f64>(xs() >> 11) / 9007199254740992.0 * hsc::k_2pi;
      a[1] = static_cast<f64>(xs() >> 11) / 9007199254740992.0 * hsc::k_2pi;
      f64 sn[4], cs[4];
      hsc::__sincos2(a[0], a[1], sn, cs);
      for ( u32 k = 0; k < 2; ++k ) {
        f64 rs = 0, rc = 0;
        micron::sincos(a[k], rs, rc);
        if ( ubits(sn[k]) != ubits(rs) || ubits(cs[k]) != ubits(rc) ) ++bad;
      }
    }
    sb::require(bad, 0ull);
  }

  sb::test_case("quat_mul/oct_mul: runtime bit-identical to the consteval-baked grid");
  {
    u64 bad = 0;
    for ( usize r = 0; r < k_alg_rows; ++r ) {
      f64 qo[4], oo[8];
      hsc::quat_mul(k_alg.qa[r], k_alg.qb[r], qo);
      for ( u32 k = 0; k < 4; ++k )
        if ( ubits(qo[k]) != ubits(k_alg.qo[r][k]) ) ++bad;
      hsc::oct_mul(k_alg.oa[r], k_alg.ob[r], oo);
      for ( u32 k = 0; k < 8; ++k )
        if ( ubits(oo[k]) != ubits(k_alg.oo[r][k]) ) ++bad;
    }
    sb::require(bad, 0ull);
  }

  return 1;
}
