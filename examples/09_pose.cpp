//  09_pose.cpp
//  See also:
//    examples/05_modes.cpp  — all six modes side by side, including quat/oct rate tables
//
//  Build (from the repo root):
//    duck batch examples.duck && ./bin/09_pose

#include "_ex_common.hpp"

#include <micron/std.hpp>

namespace
{

constexpr usize k_blocks = 512;

u64 g_seed = 0xC0FFEE1234ABCDEFull;

u64
xs()
{
  g_seed ^= g_seed << 13;
  g_seed ^= g_seed >> 7;
  g_seed ^= g_seed << 17;
  return g_seed;
}

void
unit_quat(f64 *q)
{
  f64 s = 0;
  for ( u32 k = 0; k < 4; ++k ) {
    q[k] = static_cast<f64>(xs() >> 11) / 9007199254740992.0 - 0.5 + 1e-9;
    s += q[k] * q[k];
  }
  const f64 n = micron::math::fsqrt(s);
  for ( u32 k = 0; k < 4; ++k ) q[k] /= n;
}

//  rotation angle between two unit quaternions, in degrees (double cover folded by |dot|)
f64
angle_deg(const f64 *a, const f64 *b)
{
  f64 d = 0;
  for ( u32 k = 0; k < 4; ++k ) d += a[k] * b[k];
  d = d < 0 ? -d : d;
  d = d > 1.0 ? 1.0 : d;
  return 2.0 * micron::acos(d) * 180.0 / hsc::k_pi;
}

}      //  namespace

int
main()
{
  hsc::hopf_scratch sc;

  //  the stream: r[b] is the signal (a relative orientation), g[b] the per-record gauge
  static f64 g_r[k_blocks][4];
  static f64 g_g[k_blocks][4];
  mc::vector<f32> src;
  src.reserve(k_blocks * 8 + 1);
  for ( usize b = 0; b < k_blocks; ++b ) {
    unit_quat(g_r[b]);
    unit_quat(g_g[b]);
    f64 x[4];
    hsc::quat_mul(g_r[b], g_g[b], x);      //  q0 = r * g, q1 = g  ->  q0 * conj(q1) = r
    for ( u32 k = 0; k < 4; ++k ) src.push_back(static_cast<f32>(x[k]));
    for ( u32 k = 0; k < 4; ++k ) src.push_back(static_cast<f32>(g_g[b][k]));
  }

  ex::head("rate: what the fiber costs unit mode and quat mode does not pay");
  for ( i32 lvl : { 3, 5, 6, 7 } ) {
    auto ru = hsc::rate({ .m = hsc::mode::unit, .level = lvl, .dim_log2 = 3 }, sc);
    auto rq = hsc::rate({ .m = hsc::mode::quat, .level = lvl }, sc);
    if ( ru.is_first() && rq.is_first() )
      mc::echo("L", static_cast<u32>(lvl), ":  unit d8 ", ru.cast<hsc::rate_info>().record_bits, " bits/block   quat ",
               rq.cast<hsc::rate_info>().record_bits, " bits/block   saved ",
               ru.cast<hsc::rate_info>().record_bits - rq.cast<hsc::rate_info>().record_bits, " (the S^3 gauge)");
  }

  const hsc::hopf_opts o{ .m = hsc::mode::quat, .level = 6 };
  auto zr = hsc::hopf(hsc::floats{ src.begin(), src.size() }, o, sc);
  if ( !zr.is_first() ) {
    mc::echo("encode failed: ", hsc::error_name(zr.cast<hsc::error>()));
    return 1;
  }
  const hsc::fhsc &z = zr.cast<hsc::fhsc>();
  mc::echo("");
  mc::echo(k_blocks, " pose records, ", k_blocks * 8 * 4, " bytes -> ", z.size(), " bytes");

  ex::head("gauge invariance: re-gauge the whole stream, get the identical bytes");
  {
    //  replace every record's gauge with a fresh one; the physics did not change
    mc::vector<f32> regauged;
    regauged.reserve(k_blocks * 8 + 1);
    for ( usize b = 0; b < k_blocks; ++b ) {
      f64 g2[4], x[4];
      unit_quat(g2);
      hsc::quat_mul(g_r[b], g2, x);
      for ( u32 k = 0; k < 4; ++k ) regauged.push_back(static_cast<f32>(x[k]));
      for ( u32 k = 0; k < 4; ++k ) regauged.push_back(static_cast<f32>(g2[k]));
    }
    auto z2 = hsc::hopf(hsc::floats{ regauged.begin(), regauged.size() }, o, sc);
    bool same = z2.is_first() && z2.cast<hsc::fhsc>().size() == z.size();
    if ( same )
      for ( usize i = 0; i < z.size(); ++i )
        if ( z.begin()[i] != z2.cast<hsc::fhsc>().begin()[i] ) same = false;
    mc::echo(same ? "byte-identical: the gauge never touched the wire" : "streams differ (bug!)");
  }

  ex::head("what survives, what dies");
  {
    mc::vector<f32> back;
    back.reserve(src.size() + 1);
    back.resize(src.size());
    if ( !hsc::unhopf(hsc::bytes{ z.begin(), z.size() }, hsc::wfloats{ back.begin(), back.size() }, sc).is_first() ) return 1;
    f64 rel_sum = 0, rel_max = 0, abs_sum = 0;
    for ( usize b = 0; b < k_blocks; ++b ) {
      f64 x[4], y[4], yc[4], rr[4];
      for ( u32 k = 0; k < 4; ++k ) {
        x[k] = static_cast<f64>(back[b * 8 + k]);
        y[k] = static_cast<f64>(back[b * 8 + 4 + k]);
      }
      //  decoded relative rotation, renormalized (the pair is the canonical representative)
      hsc::quat_conj(y, yc);
      hsc::quat_mul(x, yc, rr);
      f64 n = micron::math::fsqrt(hsc::__fma_norm2(rr, 4));
      for ( u32 k = 0; k < 4; ++k ) rr[k] /= n > 0 ? n : 1.0;
      const f64 er = angle_deg(g_r[b], rr);
      rel_sum += er;
      if ( er > rel_max ) rel_max = er;
      //  the decoded q0 against the original q0 = r * g: the gauge is gone, so this is large
      f64 x0[4];
      hsc::quat_mul(g_r[b], g_g[b], x0);
      f64 nx = micron::math::fsqrt(hsc::__fma_norm2(x, 4));
      for ( u32 k = 0; k < 4; ++k ) x[k] /= nx > 0 ? nx : 1.0;
      abs_sum += angle_deg(x0, x);
    }
    ex::line3("relative rotation error, mean deg = ", rel_sum / static_cast<f64>(k_blocks));
    ex::line3("relative rotation error, max  deg = ", rel_max);
    ex::line3("ABSOLUTE orientation error, mean deg = ", abs_sum / static_cast<f64>(k_blocks));
    mc::echo("");
    mc::echo("the ratio q0 * conj(q1) came back to quantizer precision; q0 itself came back as");
    mc::echo("the canonical representative -- its absolute orientation is GONE. If both absolute");
    mc::echo("orientations were signal, quat deleted three dimensions you needed (oct: seven).");
    mc::echo("That is why the porcelain refuses this family for tensors; see 05_modes.");
  }

  return 0;
}
