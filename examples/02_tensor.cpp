//  02_tensor.cpp
//  See also:
//    examples/03_budget.cpp      — choosing a rate without running the codec
//    examples/04_embeddings.cpp  — the container, and reading one row out of a table
//    examples/05_modes.cpp       — why vec is the lane for tensors
//  The porcelain:
//      target  ->  plan_for()  ->  qplan  ->  quantize()  ->  qstream  ->  dequantize()
//      (want)      (solve)         (exact)    (encode)        (bytes+shape)
//
//  Build (from the repo root):
//    duck batch examples.duck && ./bin/02_tensor

#include "_ex_common.hpp"

#include <micron/std.hpp>

int
main()
{
  mc::vector<f32> w = ex::weights(512, 256);
  hsc::tensor t = hsc::tensor::of(w, 256);
  hsc::hopf_scratch sc;      //  caches the (dim, d_q) skeleton across calls; hold one, reuse it

  //  a) say what you want
  //  exactly one of these drives the solve, in this priority:
  //    an explicit (level, dim_log2) pair  >  bits_per_weight  >  rel_rmse  >  the shape's default
  ex::head("target");

  hsc::target want{};
  want.bits_per_weight = 2.5;      //  a hard budget: the artifact will not exceed it
  mc::echo("budget = 2.5 bits/weight over a ", t.rows, " x ", t.cols, " tensor");

  //  b) solve it
  ex::head("plan");

  auto pr = hsc::plan_for(t, want, sc);
  if ( !pr.is_first() ) {
    mc::echo("no cell meets that budget: ", hsc::error_name(pr.cast<hsc::error>()));
    return 1;
  }
  const hsc::qplan &p = pr.cast<hsc::qplan>();

  mc::echo("dim          = ", 1u << p.dim_log2, "        elements per block");
  mc::echo("level        = ", static_cast<i32>(p.level), "        preset index into the d ladder");
  mc::echo("d_q          = ", p.dq, "  the minimum codeword distance, fixed point");
  mc::echo("gain_bits    = ", p.gain_bits, "        per-block magnitude field");
  mc::echo("shape_bits   = ", p.shape_bits, "       ceil(log2 M): the whole cost of the DIRECTION");
  mc::echo("record_bits  = ", p.record_bits, "       gain + shape, constant for every block");
  mc::echo("blocks/row   = ", p.blocks_per_row);
  mc::echo("padded cols  = ", p.padded_cols, "      ", p.padded_cols == p.cols ? "(nothing padded)" : "(row tail padded)");
  mc::echo("blocks       = ", p.blocks);
  mc::echo("bytes        = ", p.bytes, "    exact, framing and pad bits included");
  ex::line3("bits/weight  = ", p.bits_per_weight, "    what the artifact costs");
  ex::line3("payload only = ", p.payload_bits_per_weight, "    framing excluded (BENCHMARKS.md's column)");
  ex::line3("ratio        = ", p.ratio, "x vs fp32");
  ex::line3("est rel rmse = ", p.est_rel_rmse, "    FITTED, not a bound -- measure it below");
  mc::echo("degenerate   = ", p.degenerate ? "yes" : "no");

  //  c) encode
  ex::head("encode");

  auto qr = hsc::quantize(t, p, sc);
  if ( !qr.is_first() ) {
    mc::echo("encode failed: ", hsc::error_name(qr.cast<hsc::error>()));
    return 1;
  }
  const hsc::qstream &q = qr.cast<hsc::qstream>();
  mc::echo("planned ", p.bytes, " bytes, wrote ", q.size(), " -> ", q.size() == p.bytes ? "exact" : "MISMATCH");

  //  what came out is an ORDINARY hsc stream. the plumbing reads it with no help from us
  auto probe = hsc::hopf_probe(q.view());
  if ( probe.is_first() ) {
    const hsc::hopf_info &fi = probe.cast<hsc::hopf_info>();
    mc::echo("hopf_probe:  mode=", static_cast<u32>(fi.m), " dim_log2=", fi.dim_log2, " n_elems=", fi.n_elems, " nblocks=", fi.nblocks);
  }
  mc::echo("crc verifies: ", hsc::verify(q.view()).is_first() ? "yes" : "no");

  //  d) decode, and measure what it actually cost
  ex::head("decode");

  mc::vector<f32> back;
  back.reserve(t.elems() + 1);
  back.resize(t.elems());
  auto dr = hsc::dequantize(q, hsc::wfloats{ back.begin(), back.size() }, sc);
  if ( !dr.is_first() ) {
    mc::echo("decode failed: ", hsc::error_name(dr.cast<hsc::error>()));
    return 1;
  }

  const hsc::qerror e = hsc::measure(t, hsc::floats{ back.begin(), back.size() });
  ex::line3("rel rmse  = ", e.rel_rmse, "   ||err|| / ||x||");
  ex::line3("predicted = ", p.est_rel_rmse);
  ex::line3("ratio p/m = ", p.est_rel_rmse / e.rel_rmse, "   how much the fitted law was off by");
  ex::line3("cosine    = ", e.cos);
  ex::line3("psnr      = ", e.psnr_db, " dB");
  mc::echo("elements  = ", e.elems);

  //  e) the same thing, in one line
  //  for when you do not care to look at the plan
  ex::head("one-liner");

  auto one = hsc::quantize(t, { .rel_rmse = 0.2 });      //  a quality goal instead of a budget
  if ( one.is_first() ) {
    const hsc::qstream &o = one.cast<hsc::qstream>();
    mc::echon("rel_rmse <= 0.2 costs ");
    ex::put3(o.p.bits_per_weight);
    mc::echo(" bits/weight at dim ", 1u << o.p.dim_log2, " L", static_cast<i32>(o.p.level));
  }

  return 0;
}
