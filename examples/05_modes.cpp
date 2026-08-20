// 05_modes.cpp
// See also:
//   examples/04_embeddings.cpp  — the lane you actually want for a table of vectors
//   examples/06_transform.cpp   — the pre-rotation, for data that sits on the axes
//   examples/09_pose.cpp        — the quat lane end to end, on gauge-carrying orientation data
//
// hsc has six input modes and they are not interchangeable
//   bin       arbitrary bytes, centered on 127.5, shape + gain
//   vec       f32 blocks, shape + gain               <- the lane for weights and embeddings
//   unit      f32 blocks PROMISED unit-norm: no gain field at all
//   quotient  4-float complex pairs up to a global phase, through the Hopf map onto S^2
//   quat      8-float quaternion pairs up to the S^3 fiber, onto S^4 (vector-first, scalar-last)
//   oct       16-float octonion pairs up to the S^7 fiber, onto S^8
//
// Build (from the repo root):
//   duck batch examples.duck && ./bin/05_modes

#include "_ex_common.hpp"

#include <micron/std.hpp>

namespace
{

constexpr usize k_rows = 128;
constexpr usize k_cols = 64;

};      // namespace

int
main()
{
  hsc::hopf_scratch sc;
  mc::vector<f32> w = ex::weights(k_rows, k_cols);
  hsc::tensor t = hsc::tensor::of(w, k_cols);

  // a) vec: the lane for tensors
  // every block carries its own gain, so a row keeps its internal shape as well as its direction
  ex::head("vec");

  auto v = hsc::quantize(t, { .level = 6, .dim_log2 = 3 }, sc);
  if ( !v.is_first() ) {
    mc::echo("vec failed: ", hsc::error_name(v.cast<hsc::error>()));
    return 1;
  }
  const hsc::qstream &vq = v.cast<hsc::qstream>();
  mc::vector<f32> vb;
  vb.reserve(t.elems() + 1);
  vb.resize(t.elems());
  hsc::dequantize(vq, hsc::wfloats{ vb.begin(), vb.size() }, sc);
  const hsc::qerror ve = hsc::measure(t, hsc::floats{ vb.begin(), vb.size() });
  mc::echo("record  = ", vq.p.gain_bits, " gain + ", vq.p.shape_bits, " shape = ", vq.p.record_bits, " bits/block");
  ex::line3("bits/w  = ", vq.p.bits_per_weight);
  ex::line3("rel rmse= ", ve.rel_rmse);

  // b) unit
  ex::head("unit");

  mc::echo("cols = ", k_cols, ", dim = 8, so a row is 8 blocks, not one");
  auto bad = hsc::plan_for(t, { .m = hsc::mode::unit, .level = 6, .dim_log2 = 3 }, sc);
  mc::echo("porcelain says: ", bad.is_first() ? "accepted" : hsc::error_name(bad.cast<hsc::error>()));

  // the raw verb will do it anyway, because the plumbing does not know these were rows
  // here is what it costs: every block comes back at norm 1, so the row's profile is flattened
  auto raw = hsc::hopf(hsc::floats{ w.begin(), w.size() }, hsc::hopf_opts{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 3 }, sc);
  if ( raw.is_first() ) {
    mc::vector<f32> ub;
    ub.reserve(t.elems() + 1);
    ub.resize(t.elems());
    const hsc::fhsc &rz = raw.cast<hsc::fhsc>();
    if ( hsc::unhopf(hsc::bytes{ rz.begin(), rz.size() }, hsc::wfloats{ ub.begin(), ub.size() }, sc).is_first() ) {
      const hsc::qerror ue = hsc::measure(t, hsc::floats{ ub.begin(), ub.size() });
      mc::echo("plumbing  = ", rz.size(), " bytes (vs ", vq.size(), " for vec: the gain field is what you saved)");
      ex::line3("rel rmse  = ", ue.rel_rmse, "   <- against vec's, above");
      ex::line3("cosine    = ", ue.cos);
      mc::echo("the direction survived; the MAGNITUDES did not. For a row wider than one block");
      mc::echo("that is not compression, it is deletion -- which is why plan_for refuses it.");
    }
  }

  // and where unit is right: a table whose row is exactly one block, already normalized
  mc::echo("");
  mc::vector<f32> e8 = ex::embeddings(256, 8);
  hsc::tensor t8 = hsc::tensor::of(e8, 8);
  auto ok = hsc::quantize(t8, { .m = hsc::mode::unit, .level = 6, .dim_log2 = 3 }, sc);
  if ( ok.is_first() ) {
    const hsc::qstream &oq = ok.cast<hsc::qstream>();
    mc::echo("256 unit-norm 8-vectors, unit lane: record = ", oq.p.record_bits, " bits (no gain field)");
    auto cmp = hsc::plan_for(t8, { .level = 6, .dim_log2 = 3 }, sc);
    ex::line3("unit lane = ", oq.p.bits_per_weight, " bits/weight");
    if ( cmp.is_first() ) ex::line3("vec lane  = ", cmp.cast<hsc::qplan>().bits_per_weight, " bits/weight, for the same data");
  }

  // c) quotient: a symmetry, quotiented
  ex::head("quotient");

  mc::vector<f32> u4 = ex::embeddings(512, 4);
  const hsc::hopf_opts uo{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 2 };
  const hsc::hopf_opts qo{ .m = hsc::mode::quotient, .level = 6 };
  auto ru = hsc::rate(uo, sc);
  auto rq = hsc::rate(qo, sc);
  if ( ru.is_first() && rq.is_first() ) {
    mc::echo("unit     on S^3: ", ru.cast<hsc::rate_info>().record_bits, " bits/block");
    mc::echo("quotient on S^2: ", rq.cast<hsc::rate_info>().record_bits, " bits/block");
    mc::echo("saved          : ", ru.cast<hsc::rate_info>().record_bits - rq.cast<hsc::rate_info>().record_bits,
             " bits -- exactly the fibre");
  }
  auto qz = hsc::hopf(hsc::floats{ u4.begin(), u4.size() }, qo, sc);
  if ( qz.is_first() ) {
    mc::vector<f32> qb;
    qb.reserve(u4.size() + 1);
    qb.resize(u4.size());
    const hsc::fhsc &z = qz.cast<hsc::fhsc>();
    if ( hsc::unhopf(hsc::bytes{ z.begin(), z.size() }, hsc::wfloats{ qb.begin(), qb.size() }, sc).is_first() ) {
      const hsc::qerror qe = hsc::measure(hsc::floats{ u4.begin(), u4.size() }, hsc::floats{ qb.begin(), qb.size() });
      ex::line3("coordinate-wise rel rmse = ", qe.rel_rmse, "   and this number is MEANINGLESS here:");
      mc::echo("the decode returns a canonical representative of the class, not your original");
      mc::echo("phase, so a coordinate metric scores a rotation you asked it to discard.");
    }
  }
  mc::echo("");
  auto refused = hsc::plan_for(t, { .m = hsc::mode::quotient }, sc);
  mc::echo("quotient on a weight matrix: ", refused.is_first() ? "accepted" : hsc::error_name(refused.cast<hsc::error>()));

  // d) quat: the quaternionic sibling
  ex::head("quat");

  {
    const hsc::hopf_opts u8o{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 3 };
    const hsc::hopf_opts q8o{ .m = hsc::mode::quat, .level = 6 };
    auto ru8 = hsc::rate(u8o, sc);
    auto rq8 = hsc::rate(q8o, sc);
    if ( ru8.is_first() && rq8.is_first() ) {
      mc::echo("unit on S^7: ", ru8.cast<hsc::rate_info>().record_bits, " bits/block");
      mc::echo("quat on S^4: ", rq8.cast<hsc::rate_info>().record_bits, " bits/block");
      mc::echo("saved      : ", ru8.cast<hsc::rate_info>().record_bits - rq8.cast<hsc::rate_info>().record_bits,
               " bits -- the whole S^3 fiber");
    }
    mc::vector<f32> qp;
    qp.reserve(8 * 64 + 1);
    for ( usize i = 0; i < 8 * 64; ++i ) qp.push_back(static_cast<f32>(micron::sin(static_cast<f64>(i) * 0.71) + 1.1));
    auto z1 = hsc::hopf(hsc::floats{ qp.begin(), qp.size() }, q8o, sc);
    f64 g[4] = { 0.5, -0.5, 0.5, 0.5 };      // a unit quaternion
    mc::vector<f32> qr;
    qr.reserve(qp.size() + 1);
    for ( usize b = 0; b < qp.size() / 8; ++b ) {
      f64 x[4], y[4], xr[4], yr[4];
      for ( u32 k = 0; k < 4; ++k ) {
        x[k] = static_cast<f64>(qp[b * 8 + k]);
        y[k] = static_cast<f64>(qp[b * 8 + 4 + k]);
      }
      hsc::quat_mul(x, g, xr);
      hsc::quat_mul(y, g, yr);
      for ( u32 k = 0; k < 4; ++k ) qr.push_back(static_cast<f32>(xr[k]));
      for ( u32 k = 0; k < 4; ++k ) qr.push_back(static_cast<f32>(yr[k]));
    }
    auto z2 = hsc::hopf(hsc::floats{ qr.begin(), qr.size() }, q8o, sc);
    if ( z1.is_first() && z2.is_first() ) {
      const hsc::fhsc &a = z1.cast<hsc::fhsc>();
      const hsc::fhsc &b = z2.cast<hsc::fhsc>();
      bool same = a.size() == b.size();
      for ( usize i = 0; same && i < a.size(); ++i ) same = a.begin()[i] == b.begin()[i];
      mc::echo("gauge-rotated input -> ", same ? "byte-identical stream" : "DIFFERENT stream (bug!)");
    }
    auto refq = hsc::plan_for(t, { .m = hsc::mode::quat }, sc);
    mc::echo("quat on a weight matrix: ", refq.is_first() ? "accepted" : hsc::error_name(refq.cast<hsc::error>()));
  }

  // e) oct: the octonionic sibling
  ex::head("oct");

  {
    const hsc::hopf_opts u16o{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 4 };
    const hsc::hopf_opts o16o{ .m = hsc::mode::oct, .level = 6 };
    auto ru16 = hsc::rate(u16o, sc);
    auto ro16 = hsc::rate(o16o, sc);
    if ( ru16.is_first() && ro16.is_first() ) {
      mc::echo("unit on S^15: ", ru16.cast<hsc::rate_info>().record_bits, " bits/block");
      mc::echo("oct  on S^8 : ", ro16.cast<hsc::rate_info>().record_bits, " bits/block");
      mc::echo("saved       : ", ru16.cast<hsc::rate_info>().record_bits - ro16.cast<hsc::rate_info>().record_bits,
               " bits -- the whole S^7 fiber");
    }
    auto refo = hsc::plan_for(t, { .m = hsc::mode::oct }, sc);
    mc::echo("oct on a weight matrix: ", refo.is_first() ? "accepted" : hsc::error_name(refo.cast<hsc::error>()));
  }

  // f) bin: bytes, and the one lane the porcelain does not wrap
  // hopf() on bytes cannot fail and needs no plan
  ex::head("bin");

  mc::vector<u8> raw8;
  raw8.reserve(4096);
  for ( usize i = 0; i < 4096; ++i ) raw8.push_back(static_cast<u8>(128 + 60.0 * micron::sin(static_cast<f64>(i) * 0.02)));
  hsc::fhsc bz = hsc::hopf(raw8, hsc::hopf_opts{ .level = 6, .dim_log2 = 3 });
  mc::echo("4096 bytes -> ", bz.size(), " bytes");
  ex::line3("bits/byte  = ", static_cast<f64>(bz.size()) * 8.0 / 4096.0);
  auto pf = hsc::plan_for(t, { .m = hsc::mode::bin }, sc);
  mc::echo("bin through the porcelain: ", pf.is_first() ? "accepted" : hsc::error_name(pf.cast<hsc::error>()));
  mc::echo("(the porcelain is the f32 lane; bytes have no rows, so hsc::hopf is the whole API)");

  return 0;
}
