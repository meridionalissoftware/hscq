//  06_transform.cpp
//  See also:
//    examples/02_tensor.cpp   — the plan, field by field
//    examples/05_modes.cpp    — the six input modes
//
//  Build (from the repo root):
//    duck batch examples.duck && ./bin/06_transform

#include "_ex_common.hpp"

#include <micron/std.hpp>

namespace
{

constexpr usize k_rows = 256;
constexpr usize k_cols = 64;

mc::vector<f32>
spiky(usize rows, usize cols)
{
  ex::rng g;
  mc::vector<f32> v;
  v.reserve(rows * cols + 1);
  v.resize(rows * cols);
  for ( usize i = 0; i < rows * cols; ++i ) v[i] = static_cast<f32>(g.unit() * 0.02);
  for ( usize r = 0; r < rows; ++r )
    for ( usize b = 0; b * 8 < cols; ++b ) v[r * cols + b * 8 + (r + b) % 8] = static_cast<f32>((r % 2) ? 1.0 : -1.0);
  return v;
}

void
run(const char *label, const hsc::tensor &t, hsc::hopf_scratch &sc, u32 dl, i32 lvl)
{
  mc::vector<f32> back;
  back.reserve(t.elems() + 1);
  back.resize(t.elems());
  f64 off = 0, on = 0;
  usize bytes_off = 0, bytes_on = 0;
  for ( u32 pass = 0; pass < 2; ++pass ) {
    hsc::target tg{};
    tg.level = lvl;
    tg.dim_log2 = dl;
    tg.transform = pass == 1;
    const auto pr = hsc::plan_for(t, tg, sc);
    if ( !pr.is_first() ) return;
    const auto zr = hsc::quantize(t, pr.cast<hsc::qplan>(), sc);
    if ( !zr.is_first() ) return;
    const hsc::qstream &q = zr.cast<hsc::qstream>();
    if ( !hsc::dequantize(q, hsc::wfloats{ back.begin(), back.size() }, sc).is_first() ) return;
    const hsc::qerror e = hsc::measure(t, hsc::floats{ back.begin(), back.size() });
    (pass ? on : off) = e.rel_rmse;
    (pass ? bytes_on : bytes_off) = q.size();
  }
  ex::pad(label, 13);
  mc::echon("dim ");
  ex::padnum(1u << dl, 4);
  mc::echon("L");
  ex::padnum(static_cast<u64>(lvl), 4);
  ex::padf3(off, 9);
  ex::padf3(on, 9);
  ex::padf3(off / on, 8);
  mc::echo(bytes_off == bytes_on ? "same bytes" : "RATE CHANGED");
}

};      //  namespace

int
main()
{
  hsc::hopf_scratch sc;

  //  a) the case it was built for
  ex::head("axis-concentrated rows");

  mc::vector<f32> sp = spiky(k_rows, k_cols);
  const hsc::tensor ts = hsc::tensor::of(sp, k_cols);
  mc::echo("data         cell         off      on       gain    rate");
  run("one-hot", ts, sc, 3, 6);
  run("one-hot", ts, sc, 4, 5);
  run("one-hot", ts, sc, 2, 6);

  //  b) the case it hurts
  //  smooth, correlated rows already sit near the dense leaves; rotating them moves them off
  ex::head("smooth rows");

  mc::vector<f32> w = ex::weights(k_rows, k_cols);
  const hsc::tensor tw = hsc::tensor::of(w, k_cols);
  mc::echo("data         cell         off      on       gain    rate");
  run("smooth f32", tw, sc, 3, 6);
  run("smooth f32", tw, sc, 4, 5);
  mc::echo("-> on f32 rows the rotation is roughly a wash: a small win at one cell, a small loss");
  mc::echo("   at another. It is the BYTE lane where it clearly hurts:");
  mc::echo("");

  //  bin mode on smooth bytes is the measured regression that keeps the default off
  mc::vector<u8> sb;
  sb.reserve(65536);
  for ( usize i = 0; i < 65536; ++i ) sb.push_back(static_cast<u8>(128 + 100.0 * micron::sin(static_cast<f64>(i) * 0.003)));
  mc::vector<u8> ob;
  ob.reserve(65536 + 1);
  ob.resize(65536);
  for ( u32 dl : { 3u, 4u } ) {
    f64 err[2] = { 0, 0 };
    for ( u32 pass = 0; pass < 2; ++pass ) {
      const hsc::hopf_opts o{ .level = dl == 3 ? 6 : 5, .dim_log2 = dl, .transform = pass == 1 };
      const hsc::fhsc z = hsc::hopf(sb, o);
      if ( !hsc::unhopf(hsc::bytes{ z.begin(), z.size() }, hsc::wbytes{ ob.begin(), ob.size() }, sc).is_first() ) continue;
      f64 se = 0;
      for ( usize i = 0; i < 65536; ++i ) {
        const f64 d = static_cast<f64>(ob[i]) - static_cast<f64>(sb[i]);
        se += d * d;
      }
      err[pass] = micron::math::fsqrt(se / 65536.0);
    }
    ex::pad("smooth bytes", 13);
    mc::echon("dim ");
    ex::padnum(1u << dl, 4);
    mc::echon("L");
    ex::padnum(dl == 3 ? 6 : 5, 4);
    ex::padf3(err[0], 9);
    ex::padf3(err[1], 9);
    ex::padf3(err[0] / err[1], 8);
    mc::echo("per-byte rmse");
  }

  //  c) the point about rate
  ex::head("rate is unchanged");

  hsc::target a{};
  a.level = 6;
  a.dim_log2 = 3;
  hsc::target b = a;
  b.transform = true;
  const auto pa = hsc::plan_for(ts, a, sc);
  const auto pb = hsc::plan_for(ts, b, sc);
  if ( pa.is_first() && pb.is_first() ) {
    mc::echo("transform off: ", pa.cast<hsc::qplan>().bytes, " bytes, record ", pa.cast<hsc::qplan>().record_bits, " bits");
    mc::echo("transform on : ", pb.cast<hsc::qplan>().bytes, " bytes, record ", pb.cast<hsc::qplan>().record_bits, " bits");
    mc::echo("the flag lives in the header (bit 2), so a decoder needs no help from you.");
  }

  mc::echo("");
  mc::echo("Rule of thumb: turn it on when your rows are sparse, spiky or one-hot -- 2x to 3x lower");
  mc::echo("error at identical rate. Leave it off for smooth data: a wash on f32 rows and a clear");
  mc::echo("loss on smooth bytes, which is the measurement that keeps the default OFF. It is");
  mc::echo("excluded from quotient mode entirely -- that lane's Hopf map needs the complex");
  mc::echo("structure the rotation breaks. And it is analog-side only: streams stay the same size.");
  return 0;
}
