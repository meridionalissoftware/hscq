// Quotient mode geometry. Guards: the Hopf projection/section pair (h(lift(p)) == p on every
// codeword), TRUE phase invariance (quantize(e^{i psi} z) == quantize(z) across the whole
// fiber), the canonical-representative convention (Re z0 >= 0, Im z0 == 0; south pole pins to
// (0, 1)), scale invariance, the quotient-metric error bound, and bad-input rejection.

#include "../src/hsc/codec/quotient.hpp"
#include "../src/hsc/codec/scratch.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

// squared quotient distance between unit quaternions: dq^2 = 2 - 2 |<a, b>_C|
f64
quotient_dist(const f32 *a, const f32 *b)
{
  const f64 re
      = static_cast<f64>(a[0]) * b[0] + static_cast<f64>(a[1]) * b[1] + static_cast<f64>(a[2]) * b[2] + static_cast<f64>(a[3]) * b[3];
  const f64 im
      = static_cast<f64>(a[1]) * b[0] - static_cast<f64>(a[0]) * b[1] + static_cast<f64>(a[3]) * b[2] - static_cast<f64>(a[2]) * b[3];
  const f64 m = __builtin_sqrt(re * re + im * im);
  const f64 s = 2.0 - 2.0 * (m > 1.0 ? 1.0 : m);
  return __builtin_sqrt(s > 0.0 ? s : 0.0);
}

void
phase_rotate(const f32 *z, f64 psi, f32 *out)
{
  const f64 c = micron::cos(psi), s = micron::sin(psi);
  out[0] = static_cast<f32>(z[0] * c - z[1] * s);
  out[1] = static_cast<f32>(z[0] * s + z[1] * c);
  out[2] = static_cast<f32>(z[2] * c - z[3] * s);
  out[3] = static_cast<f32>(z[2] * s + z[3] * c);
}

}      // namespace

int
main()
{
  hsc::hopf_scratch sc;
  sb::require(sc.build_s2(hsc::level_dq(6)) >= 0);      // d = 0.2, M = 304-class codebook family

  sb::test_case("the section is a true section: h(lift(p)) == p on every S^2 codeword");
  {
    for ( u64 a = 0; a < sc.s2.m_total; ++a ) {
      f64 p[3]{}, back[3]{};
      hsc::s2_decode(sc.s2, a, p);
      f32 z[4];
      hsc::hopf_lift(p, z);
      // the lift is unit
      f64 n2 = 0;
      for ( i32 c = 0; c < 4; ++c ) n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      sb::require(n2 > 1.0 - 1e-6 && n2 < 1.0 + 1e-6);
      // canonical representative: z0 real, nonnegative
      sb::require(static_cast<f64>(z[0]) >= 0.0);
      sb::require(hsc::__f2u(z[1]) == 0u || z[1] == 0.0f);
      sb::require(hsc::hopf_project(z, back) >= 0);
      f64 e2 = 0;
      for ( i32 c = 0; c < 3; ++c ) {
        const f64 e = back[c] - p[c];
        e2 += e * e;
      }
      sb::require(__builtin_sqrt(e2) < 1e-6);      // f32 lift round-trips within f32 grain
    }
  }

  sb::test_case("phase invariance: the whole fiber quantizes to one class");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 800; ++t ) {
      f32 z[4];
      f64 n2 = 0;
      for ( i32 c = 0; c < 4; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      u64 a0 = 0;
      sb::require(hsc::quotient_quantize(sc.s2, z, a0) >= 0);
      for ( i32 k = 1; k < 12; ++k ) {
        f32 zr[4];
        phase_rotate(z, static_cast<f64>(k) * hsc::k_2pi / 12.0, zr);
        u64 a = 0;
        sb::require(hsc::quotient_quantize(sc.s2, zr, a) >= 0);
        sb::require(a, a0);
      }
    }
  }

  sb::test_case("scale invariance: lambda * z quantizes identically");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 500; ++t ) {
      f32 z[4] = { static_cast<f32>(g.unit()), static_cast<f32>(g.unit()), static_cast<f32>(g.unit()), static_cast<f32>(g.unit()) };
      f64 n2 = 0;
      for ( i32 c = 0; c < 4; ++c ) n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      if ( n2 < 1e-6 ) continue;
      u64 a0 = 0, a1 = 0, a2 = 0;
      sb::require(hsc::quotient_quantize(sc.s2, z, a0) >= 0);
      f32 zs[4];
      for ( i32 c = 0; c < 4; ++c ) zs[c] = z[c] * 37.5f;
      sb::require(hsc::quotient_quantize(sc.s2, zs, a1) >= 0);
      for ( i32 c = 0; c < 4; ++c ) zs[c] = z[c] * 0.001f;
      sb::require(hsc::quotient_quantize(sc.s2, zs, a2) >= 0);
      sb::require(a1, a0);
      sb::require(a2, a0);
    }
  }

  sb::test_case("quotient-metric error stays within ~d (worst) for unit inputs");
  {
    tutil::rng g;
    const f64 d = hsc::d_of(hsc::level_dq(6));
    f64 se = 0;
    i32 cnt = 0;
    for ( i32 t = 0; t < 4000; ++t ) {
      f32 z[4];
      f64 n2 = 0;
      for ( i32 c = 0; c < 4; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      const f64 n = __builtin_sqrt(n2);
      for ( i32 c = 0; c < 4; ++c ) z[c] = static_cast<f32>(static_cast<f64>(z[c]) / n);
      u64 a = 0;
      sb::require(hsc::quotient_quantize(sc.s2, z, a) >= 0);
      f32 zh[4];
      hsc::quotient_reconstruct(sc.s2, a, zh);
      const f64 dq = quotient_dist(z, zh);
      sb::require(dq <= d * 1.05 + 1e-6);      // measured worst ~ d; 5% slack for band corners
      se += dq * dq;
      ++cnt;
    }
    // RMS tracks ~ d/4 (the fiber halves the base metric); generous ceiling at d/2
    sb::require(__builtin_sqrt(se / cnt) < d * 0.5);
  }

  sb::test_case("the reconstruct -> requantize class is a fixed point");
  {
    for ( u64 a = 0; a < sc.s2.m_total; ++a ) {
      f32 z[4];
      hsc::quotient_reconstruct(sc.s2, a, z);
      u64 back = 0;
      sb::require(hsc::quotient_quantize(sc.s2, z, back) >= 0);
      sb::require(back, a);
    }
  }

  sb::test_case("zero and non-finite blocks are rejected as bad_value");
  {
    f32 z[4]{};
    u64 a = 0;
    sb::require(hsc::as_error(hsc::quotient_quantize(sc.s2, z, a)) == hsc::error::bad_value);
    f32 zn[4] = { 1.0f, 0.0f, hsc::__u2f(0x7FC00000u), 0.0f };
    sb::require(hsc::as_error(hsc::quotient_quantize(sc.s2, zn, a)) == hsc::error::bad_value);
  }

  return 1;
}
