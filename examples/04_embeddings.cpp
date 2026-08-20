//  04_embeddings.cpp
//  See also:
//    examples/02_tensor.cpp   — the plan, field by field
//    examples/03_budget.cpp   — what a rate costs before you spend it
//    examples/05_modes.cpp    — why `unit` is the WRONG lane for a 384-wide row
//
//  Build (from the repo root):
//    duck batch examples.duck && ./bin/04_embeddings

#include "_ex_common.hpp"

#include <micron/std.hpp>

namespace
{

constexpr usize k_rows = 2000;
constexpr usize k_cols = 384;
constexpr usize k_topk = 10;

//  exhaustive cosine top-k. Rows are unit-norm, so cosine is a dot product
void
topk(const f32 *table, usize rows, usize cols, const f32 *query, usize *out, usize k)
{
  for ( usize i = 0; i < k; ++i ) out[i] = 0;
  f64 best[k_topk];
  for ( usize i = 0; i < k; ++i ) best[i] = -2.0;
  for ( usize r = 0; r < rows; ++r ) {
    f64 d = 0;
    for ( usize c = 0; c < cols; ++c ) d += static_cast<f64>(table[r * cols + c]) * static_cast<f64>(query[c]);
    if ( d <= best[k - 1] ) continue;
    usize at = k - 1;
    while ( at > 0 && best[at - 1] < d ) {
      best[at] = best[at - 1];
      out[at] = out[at - 1];
      --at;
    }
    best[at] = d;
    out[at] = r;
  }
}

};      //  namespace

int
main()
{
  mc::vector<f32> table = ex::embeddings(k_rows, k_cols);
  for ( usize r = 0; r < 40; ++r )
    for ( usize c = 0; c < k_cols; ++c ) table[r * k_cols + c] *= 500.0f;

  hsc::tensor t = hsc::tensor::of(table, k_cols);
  hsc::hopf_scratch sc;

  //  a) pack the table
  //  an HSCQ blob is self-describing: rows, cols and chunking in 32 bytes, then ordinary hsc
  //  streams. Nothing else has to be stored beside it
  ex::head("pack");

  auto pk = hsc::pack(t, { .bits_per_weight = 3.0 }, sc);
  if ( !pk.is_first() ) {
    mc::echo("pack failed: ", hsc::error_name(pk.cast<hsc::error>()));
    return 1;
  }
  const hsc::fhsc &blob = pk.cast<hsc::fhsc>();
  const hsc::bytes bv{ blob.begin(), blob.size() };

  auto qi = hsc::qprobe(bv);
  if ( !qi.is_first() ) {
    mc::echo("qprobe failed: ", hsc::error_name(qi.cast<hsc::error>()));
    return 1;
  }
  const hsc::qinfo &info = qi.cast<hsc::qinfo>();
  mc::echo("table       = ", info.rows, " x ", info.cols, " f32 = ", t.elems() * 4, " bytes");
  mc::echo("packed      = ", blob.size(), " bytes in ", info.chunks, " chunks of ", info.chunk_rows, " rows");
  ex::line3("bits/weight = ", info.p.bits_per_weight);
  ex::line3("ratio       = ", info.p.ratio, "x vs fp32");
  mc::echo("dim         = ", 1u << info.p.dim_log2, "  record = ", info.p.record_bits, " bits/block");
  ex::line3("framing     = ", static_cast<f64>(info.chunks * 48) * 100.0 / static_cast<f64>(blob.size()), "% of the blob");

  //  b) read one vector without touching the rest
  //  returned floats are bit-identical to what a whole table decode would yield for that row
  ex::head("random access");

  hsc::qreader rd{ bv, sc };
  if ( rd.status() != hsc::error::ok ) {
    mc::echo("reader failed: ", hsc::error_name(rd.status()));
    return 1;
  }
  mc::vector<f32> one;
  one.reserve(k_cols + 1);
  one.resize(k_cols);
  for ( usize r : { usize(0), usize(1), usize(999), k_rows - 1 } ) {
    if ( !rd.row(r, hsc::wfloats{ one.begin(), k_cols }).is_first() ) return 1;
    mc::echon("row ");
    ex::padnum(r, 6);
    mc::echon("cos to the original = ");
    ex::put3(ex::cosine(one.begin(), table.begin() + r * k_cols, k_cols));
    mc::echo("");
  }

  //  c) the whole table back, and what it cost
  ex::head("quality");

  mc::vector<f32> back;
  back.reserve(t.elems() + 1);
  back.resize(t.elems());
  if ( !hsc::unpack(bv, hsc::wfloats{ back.begin(), back.size() }, sc).is_first() ) return 1;
  const hsc::qerror e = hsc::measure(t, hsc::floats{ back.begin(), back.size() });
  ex::line3("rel rmse = ", e.rel_rmse);
  ex::line3("cosine   = ", e.cos);
  ex::line3("psnr     = ", e.psnr_db, " dB");

  //  d) chunking against one big stream
  ex::head("why chunks");

  const hsc::grange gr = hsc::gain_range(t, info.p);
  ex::line3("block norm ratio  = ", gr.ratio, "   (loudest block over quietest)");
  mc::echo("gain_bits         = ", info.p.gain_bits, "  -> resolves about ", (1u << info.p.gain_bits), " of range");
  mc::echo("blocks a single full-scale would ZERO: ", gr.zeroed, " of ", gr.blocks);

  auto flat = hsc::quantize(t, { .bits_per_weight = 3.0 }, sc);
  if ( flat.is_first() ) {
    const hsc::qstream &f = flat.cast<hsc::qstream>();
    mc::vector<f32> fb;
    fb.reserve(t.elems() + 1);
    fb.resize(t.elems());
    if ( hsc::dequantize(f, hsc::wfloats{ fb.begin(), fb.size() }, sc).is_first() ) {
      const usize q0 = 40 * k_cols;
      const hsc::qerror fq
         = hsc::measure(hsc::floats{ table.begin() + q0, t.elems() - q0 }, hsc::floats{ fb.begin() + q0, t.elems() - q0 });
      const hsc::qerror cq
         = hsc::measure(hsc::floats{ table.begin() + q0, t.elems() - q0 }, hsc::floats{ back.begin() + q0, t.elems() - q0 });
      mc::echo("");
      mc::echo("one stream : ", f.size(), " bytes");
      ex::line3("             quiet-row rel rmse ", fq.rel_rmse, "   <- 1.000 means they went to ZERO");
      mc::echo("chunked    : ", blob.size(), " bytes  (+", blob.size() - f.size(), " of framing)");
      ex::line3("             quiet-row rel rmse ", cq.rel_rmse);
      mc::echo("");
      mc::echo("Below the cliff the two lanes are within about 1% of each other and the bare stream");
      mc::echo("is the better deal -- it is smaller and simpler. gain_range() tells you which side");
      mc::echo("you are on. Raising gain_bits moves the cliff instead, at 1/dim bits per weight.");
      mc::echo("");
      mc::echo("What defeats chunking: outliers spread one per chunk. Then every chunk inherits the");
      mc::echo("high full-scale and the win goes to zero. chunk_rows is the knob for that.");
    }
  }

  //  e) does retrieval survive?
  ex::head("recall@10");

  usize hits = 0, total = 0;
  usize ref[k_topk], got[k_topk];
  for ( usize qn = 0; qn < 50; ++qn ) {
    const usize qrow = (qn * 37) % k_rows;
    topk(table.begin(), k_rows, k_cols, table.begin() + qrow * k_cols, ref, k_topk);
    topk(back.begin(), k_rows, k_cols, table.begin() + qrow * k_cols, got, k_topk);
    for ( usize i = 0; i < k_topk; ++i ) {
      ++total;
      for ( usize j = 0; j < k_topk; ++j )
        if ( ref[i] == got[j] ) {
          ++hits;
          break;
        }
    }
  }
  ex::line3("recall@10 = ", static_cast<f64>(hits) / static_cast<f64>(total), "   (50 queries)");
  ex::line3("storage   = ", info.p.bits_per_weight, " bits/weight, against 32 for fp32");

  return 0;
}
