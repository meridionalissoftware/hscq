//  01_quickstart.cpp
//  See also:
//    examples/02_tensor.cpp      — the porcelain end to end, field by field
//    examples/03_budget.cpp      — what a rate costs, without running the codec
//    examples/04_embeddings.cpp  — a quantized table you can probe one row at a time
//    examples/05_modes.cpp       — bin / vec / unit / quotient, and which one you want
//
//  hsc is header-only and micron-only
//    #include <hsc/hsc.hpp>
//  Build (from the repo root):
//    duck batch examples.duck && ./bin/01_quickstart
#include "_ex_common.hpp"

#include <micron/std.hpp>

int
main()
{
  //  a) arbitrary bytes
  //  hopf() on bytes cannot fail: there is no input it rejects and no ratio it misses
  ex::head("bytes");

  mc::vector<u8> data;
  data.reserve(4096);
  for ( usize i = 0; i < 4096; ++i ) data.push_back(static_cast<u8>(128 + 40.0 * micron::sin(static_cast<f64>(i) * 0.05)));

  hsc::fhsc z = hsc::hopf(data, hsc::hopf_opts{ .level = 6, .dim_log2 = 3 });
  mc::echo("in    = ", data.size(), " bytes");
  mc::echo("out   = ", z.size(), " bytes");
  ex::line3("ratio = ", static_cast<f64>(data.size()) / static_cast<f64>(z.size()), "x");

  auto back = hsc::unhopf(z);      //  result<fhsc>
  mc::echo("decodes: ", back.is_first() ? "yes" : hsc::error_name(back.cast<hsc::error>()));

  //  b) f32 vectors
  //  NOTE: typed lanes _can_ fail (NaN, a zero-norm block where unit-norm was promised)
  //  they return result<T> instead
  ex::head("floats");

  mc::vector<f32> v = ex::weights(64, 32);
  auto zv = hsc::hopf(v, hsc::hopf_opts{ .m = hsc::mode::vec, .level = 6, .dim_log2 = 3 });
  mc::echo("2048 f32 -> ", zv.is_first() ? zv.cast<hsc::fhsc>().size() : usize(0), " bytes");

  //  c) a tensor, which is what you are usually actually holding
  //  this is the porcelain: define the shape of your data and desired outputs
  ex::head("tensor");

  mc::vector<f32> w = ex::weights(256, 384);
  hsc::tensor t = hsc::tensor::of(w, 384);      //  [256, 384] row-major (always)

  auto q = hsc::quantize(t, { .bits_per_weight = 3.0 });
  if ( !q.is_first() ) {
    mc::echo("quantize failed: ", hsc::error_name(q.cast<hsc::error>()));
    return 1;
  }
  const hsc::qstream &qz = q.cast<hsc::qstream>();
  mc::echo("tensor  = ", t.rows, " x ", t.cols, " f32 = ", t.elems() * 4, " bytes");
  mc::echo("encoded = ", qz.size(), " bytes");
  ex::line3("        = ", qz.p.bits_per_weight, " bits/weight");
  ex::line3("        = ", qz.p.ratio, "x vs fp32");

  //  d) costs
  //  rel_rmse is ||err|| / ||x||, the same convention BENCHMARKS.md reports
  ex::head("quality");

  auto back2 = hsc::dequantize(qz);      //  result<fhsc32>, owning
  if ( !back2.is_first() ) {
    mc::echo("dequantize failed: ", hsc::error_name(back2.cast<hsc::error>()));
    return 1;
  }
  const hsc::fhsc32 &got = back2.cast<hsc::fhsc32>();
  const hsc::qerror e = hsc::measure(t, hsc::floats{ got.begin(), got.size() });
  ex::line3("rel rmse  = ", e.rel_rmse);
  ex::line3("predicted = ", qz.p.est_rel_rmse, "   (the fitted law, before any data was read)");
  ex::line3("cosine    = ", e.cos);
  ex::line3("psnr      = ", e.psnr_db, " dB");

  //  e) compile time, same wire bytes
  //  hsc::ct runs the full codec in constant evaluation, and tests/comptime.cpp proves the
  //  consteval stream byte-identical to the runtime one
  ex::head("comptime");

  static constexpr hsc::ct::str body{ "hopfed at compile time" };
  static constexpr hsc::hopf_opts opts{ .level = 6, .dim_log2 = 3 };
  constexpr auto stream = hsc::ct::hopf<body, opts>();
  constexpr auto plain = hsc::ct::unhopf<stream>();
  static_assert(plain.len == body.len);
  mc::echo("baked ", body.len, " bytes into ", stream.len, " at compile time");

  return 0;
}
