// Quat mode geometry. Guards: the quaternionic Hopf projection/section pair (h(lift(p)) == p
// on every S^4 codeword), TRUE fiber invariance (quantize((q0 g, q1 g)) == quantize((q0, q1))
// for unit quaternions g -- simultaneous RIGHT multiplication), the canonical-representative
// convention (q0 = (0,0,0,c), c >= 0; south pole pins to (0, identity)), scale invariance,
// the quotient-metric error bound, refine-never-worse, and bad-input rejection.
// Components are VECTOR-FIRST, scalar last: q = (x, y, z, w).

#include "../src/hsc/codec/quat.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

f64
quat_dist(const f32 *a, const f32 *b)
{
  f64 x1[4], y1[4], x2[4], y2[4];
  for ( u32 k = 0; k < 4; ++k ) {
    x1[k] = static_cast<f64>(a[k]);
    y1[k] = static_cast<f64>(a[4 + k]);
    x2[k] = static_cast<f64>(b[k]);
    y2[k] = static_cast<f64>(b[4 + k]);
  }
  f64 c[4], t[4], u[4], q[4];
  hsc::quat_conj(x1, c);
  hsc::quat_mul(c, x2, t);
  hsc::quat_conj(y1, c);
  hsc::quat_mul(c, y2, u);
  for ( u32 k = 0; k < 4; ++k ) q[k] = t[k] + u[k];
  const f64 m = __builtin_sqrt(hsc::__fma_norm2(q, 4));
  const f64 s = 2.0 - 2.0 * (m > 1.0 ? 1.0 : m);
  return __builtin_sqrt(s > 0.0 ? s : 0.0);
}

void
fiber_rotate(const f32 *z, const f64 *g, f32 *out)
{
  f64 x[4], y[4], xr[4], yr[4];
  for ( u32 k = 0; k < 4; ++k ) {
    x[k] = static_cast<f64>(z[k]);
    y[k] = static_cast<f64>(z[4 + k]);
  }
  hsc::quat_mul(x, g, xr);
  hsc::quat_mul(y, g, yr);
  for ( u32 k = 0; k < 4; ++k ) {
    out[k] = static_cast<f32>(xr[k]);
    out[4 + k] = static_cast<f32>(yr[k]);
  }
}

void
random_unit_quat(tutil::rng &g, f64 *q)
{
  f64 n2 = 0;
  for ( u32 k = 0; k < 4; ++k ) {
    q[k] = g.unit();
    n2 += q[k] * q[k];
  }
  const f64 inv = 1.0 / __builtin_sqrt(n2 > 1e-12 ? n2 : 1.0);
  for ( u32 k = 0; k < 4; ++k ) q[k] *= inv;
}

static hsc::tree_node g_nodes[512];
static hsc::tree_row g_rows[1024];
static hsc::s3_leaf g_leaves[8192];
static hsc::susp_band g_bands[64];

}      // namespace

int
main()
{
  hsc::tree_arena ar{ g_nodes, 512, 0, g_rows, 1024, 0, g_leaves, 8192, 0 };
  hsc::susp_skeleton ss{};
  sb::require(hsc::susp_build(hsc::level_dq(6), 2, ar, g_bands, ss) >= 0);
  const hsc::tree_skeleton tv = hsc::tree_view(ar, 0, 2, hsc::level_dq(6));

  sb::test_case("the section is a true section: h(lift(p)) == p on every S^4 codeword");
  {
    for ( u64 a = 0; a < ss.m_mod; ++a ) {
      const u32 band = hsc::__susp_band_of(ss, a);
      hsc::tree_fields f{};
      f.base[0] = a - ss.bd[band].off_mod;
      f64 p[5]{}, back[5]{};
      hsc::susp_decode(ss, tv, band, f, p);
      f32 z[8];
      hsc::quat_lift(p, z);

      f64 n2 = 0;
      for ( i32 c = 0; c < 8; ++c ) n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      sb::require(n2 > 1.0 - 1e-6 && n2 < 1.0 + 1e-6);

      sb::require(hsc::__f2u(z[0]) == 0u);
      sb::require(hsc::__f2u(z[1]) == 0u);
      sb::require(hsc::__f2u(z[2]) == 0u);
      sb::require(static_cast<f64>(z[3]) >= 0.0);
      sb::require(hsc::quat_project(z, back) >= 0);
      f64 e2 = 0;
      for ( i32 c = 0; c < 5; ++c ) {
        const f64 e = back[c] - p[c];
        e2 += e * e;
      }
      sb::require(__builtin_sqrt(e2) < 1e-6);
    }

    f32 z[8];
    hsc::quat_reconstruct(ss, tv, ss.m_mod - 1, z);
    for ( i32 c = 0; c < 7; ++c ) sb::require(z[c] == 0.0f);
    sb::require(z[7] == 1.0f);
  }

  sb::test_case("fiber invariance: the whole S^3 fiber quantizes to one class");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 800; ++t ) {
      f32 z[8];
      f64 n2 = 0;
      for ( i32 c = 0; c < 8; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      u64 a0 = 0;
      sb::require(hsc::quat_quantize(ss, tv, z, a0) >= 0);
      for ( i32 k = 0; k < 12; ++k ) {
        f64 gq[4];
        random_unit_quat(g, gq);
        f32 zr[8];
        fiber_rotate(z, gq, zr);
        u64 a = 0;
        sb::require(hsc::quat_quantize(ss, tv, zr, a) >= 0);
        sb::require(a, a0);
      }
    }
  }

  sb::test_case("scale invariance: lambda * (q0, q1) quantizes identically");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 500; ++t ) {
      f32 z[8];
      f64 n2 = 0;
      for ( i32 c = 0; c < 8; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      u64 a0 = 0, a1 = 0, a2 = 0;
      sb::require(hsc::quat_quantize(ss, tv, z, a0) >= 0);
      f32 zs[8];
      for ( i32 c = 0; c < 8; ++c ) zs[c] = z[c] * 37.5f;
      sb::require(hsc::quat_quantize(ss, tv, zs, a1) >= 0);
      for ( i32 c = 0; c < 8; ++c ) zs[c] = z[c] * 0.001f;
      sb::require(hsc::quat_quantize(ss, tv, zs, a2) >= 0);
      sb::require(a1, a0);
      sb::require(a2, a0);
    }
  }

  sb::test_case("quotient-metric error bound for unit inputs");
  {
    tutil::rng g;
    const f64 d = hsc::d_of(hsc::level_dq(6));
    f64 se = 0, worst = 0;
    i32 cnt = 0;
    for ( i32 t = 0; t < 4000; ++t ) {
      f32 z[8];
      f64 n2 = 0;
      for ( i32 c = 0; c < 8; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      const f64 n = __builtin_sqrt(n2);
      for ( i32 c = 0; c < 8; ++c ) z[c] = static_cast<f32>(static_cast<f64>(z[c]) / n);
      u64 a = 0;
      sb::require(hsc::quat_quantize(ss, tv, z, a) >= 0);
      f32 zh[8];
      hsc::quat_reconstruct(ss, tv, a, zh);
      const f64 dq = quat_dist(z, zh);
      if ( dq > worst ) worst = dq;
      se += dq * dq;
      ++cnt;
    }

    sb::require(worst <= d * 1.0 + 1e-6);
    sb::require(__builtin_sqrt(se / cnt) < d * 0.4);
  }

  sb::test_case("the reconstruct -> requantize class is a fixed point");
  {
    for ( u64 a = 0; a < ss.m_mod; ++a ) {
      f32 z[8];
      hsc::quat_reconstruct(ss, tv, a, z);
      u64 back = 0;
      sb::require(hsc::quat_quantize(ss, tv, z, back) >= 0);
      sb::require(back, a);
    }
  }

  sb::test_case("refine never worsens the quotient-metric error");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 2000; ++t ) {
      f32 z[8];
      f64 n2 = 0;
      for ( i32 c = 0; c < 8; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      u64 a0 = 0, a1 = 0;
      sb::require(hsc::quat_quantize(ss, tv, z, a0, 0) >= 0);
      sb::require(hsc::quat_quantize(ss, tv, z, a1, 1) >= 0);
      f32 z0[8], z1[8];
      hsc::quat_reconstruct(ss, tv, a0, z0);
      hsc::quat_reconstruct(ss, tv, a1, z1);
      sb::require(quat_dist(z, z1) <= quat_dist(z, z0) + 1e-12);
    }
  }

  sb::test_case("zero and non-finite blocks are rejected as bad_value");
  {
    f32 z[8]{};
    u64 a = 0;
    sb::require(hsc::as_error(hsc::quat_quantize(ss, tv, z, a)) == hsc::error::bad_value);
    f32 zn[8] = { 1.0f, 0.0f, 0.0f, 0.0f, hsc::__u2f(0x7FC00000u), 0.0f, 0.0f, 0.0f };
    sb::require(hsc::as_error(hsc::quat_quantize(ss, tv, zn, a)) == hsc::error::bad_value);
  }

  return 1;
}
