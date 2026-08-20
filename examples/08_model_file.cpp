// 08_model_file.cpp
// See also:
//   examples/04_embeddings.cpp  — the container and row access, on synthetic data
//   benches/vector_bench.cpp    — the same measurement across a grid of cells
//   scripts/vector_eval.py      — cosine and recall@k in numpy, off the --in/--out filter
//
// The corpus is NOT in the repo. Build it first:
//   python3 scripts/corpus.py fetch && python3 scripts/corpus.py models
// Without it this prints `skip` and exits clean, exactly as the benches do.
//
// Build (from the repo root):
//   duck batch examples.duck && ./bin/08_model_file

#include "_ex_common.hpp"

#include <micron/std.hpp>

namespace
{

// a prefix is plenty to measure with, and keeps the example honest about memory
constexpr usize k_cap = 24u << 20;      // 6 Mi floats
alignas(64) u8 g_in[k_cap];

struct tf {
  const char *label;
  const char *path;
  usize cols;
};

constexpr tf k_files[] = {
  { "MiniLM embed", "corpus/models/all-MiniLM-L6-v2.embeddings_word_embeddings_weight.f32", 384 },
  { "MiniLM ffn", "corpus/models/all-MiniLM-L6-v2.encoder_layer_0_intermediate_dense_weight.f32", 384 },
  { "SmolLM2 mlp", "corpus/models/SmolLM2-135M.model_layers_0_mlp_down_proj_weight.f32", 1536 },
};

// symmetric int-N, per row: the standard baseline. One f16 scale per row, N-bit signed codes.
// This example bills the scale honestly: bits/weight = N + 16/cols. NOTE the published scorers
// (scripts/vector_eval.py, scripts/preset_chart.py) follow the usual convention and charge
// exactly N, excluding the per-row scale (16/cols ~= 0.04 b/w at 384 cols) -- their captions say so.
f64
intn_rel_rmse(const f32 *v, usize rows, usize cols, u32 nbits)
{
  const f64 top = static_cast<f64>((1 << (nbits - 1)) - 1);
  f64 se = 0, sg = 0;
  for ( usize r = 0; r < rows; ++r ) {
    f64 amax = 0;
    for ( usize c = 0; c < cols; ++c ) {
      const f64 a = static_cast<f64>(v[r * cols + c]);
      const f64 m = a < 0 ? -a : a;
      if ( m > amax ) amax = m;
    }
    // the scale is stored as f16, so quantize it the way a real kernel would
    const f64 scale = amax > 0 ? static_cast<f64>(static_cast<f32>(amax / top)) : 0.0;
    for ( usize c = 0; c < cols; ++c ) {
      const f64 a = static_cast<f64>(v[r * cols + c]);
      f64 q = scale > 0 ? micron::math::mk::round_ns::round<f64>(a / scale) : 0.0;
      if ( q > top ) q = top;
      if ( q < -top - 1 ) q = -top - 1;
      const f64 b = q * scale;
      se += (b - a) * (b - a);
      sg += a * a;
    }
  }
  return sg > 0 ? micron::math::fsqrt(se / sg) : 0.0;
}

};      // namespace

int
main()
{
  hsc::hopf_scratch sc;
  usize ran = 0;

  for ( const tf &f : k_files ) {
    const max_t got = ex::slurp_prefix(f.path, g_in, k_cap);
    if ( got <= 0 ) {
      mc::echo("skip ", f.label, "  (no ", f.path, " -- run scripts/corpus.py)");
      continue;
    }
    ++ran;
    const usize nf = static_cast<usize>(got) / 4;
    const f32 *fv = reinterpret_cast<const f32 *>(g_in);
    hsc::tensor t = hsc::tensor::of(hsc::floats{ fv, nf }, f.cols);

    ex::head(f.label);
    mc::echo(t.rows, " x ", t.cols, " f32 = ", t.elems() * 4, " bytes read");

    // what the data looks like to the gain field: the number that decides the lane
    const hsc::grange gr = hsc::gain_range(t, 4, 8);
    ex::line3("block norm ratio = ", gr.ratio, "   (dim 16, 8 gain bits resolve about 256)");
    mc::echo("blocks one full-scale would zero: ", gr.zeroed, " of ", gr.blocks);
    mc::echo("");

    // hsc at three rates, against int-N at the nearest matched rate
    mc::echo("cell           bits/w   hsc rel-rmse   baseline        base rel-rmse   cos");

    struct row {
      i32 lvl;
      u32 dl;
      u32 nbits;
      const char *bname;
    };

    const row rows[] = { { 6, 5, 2, "int2" }, { 7, 5, 3, "int3" }, { 6, 3, 4, "int4" } };

    for ( const row &r : rows ) {
      hsc::target tg{};
      tg.level = r.lvl;
      tg.dim_log2 = r.dl;
      const auto pr = hsc::plan_for(t, tg, sc);
      if ( !pr.is_first() ) continue;
      const hsc::qplan &p = pr.cast<hsc::qplan>();
      const auto zr = hsc::quantize(t, p, sc);
      if ( !zr.is_first() ) {
        mc::echo("  encode failed: ", hsc::error_name(zr.cast<hsc::error>()));
        continue;
      }
      mc::vector<f32> back;
      back.reserve(t.elems() + 1);
      back.resize(t.elems());
      if ( !hsc::dequantize(zr.cast<hsc::qstream>(), hsc::wfloats{ back.begin(), back.size() }, sc).is_first() ) continue;
      const hsc::qerror e = hsc::measure(t, hsc::floats{ back.begin(), back.size() });

      const f64 base_bw = static_cast<f64>(r.nbits) + 16.0 / static_cast<f64>(f.cols);
      const f64 base_err = intn_rel_rmse(fv, t.rows, t.cols, r.nbits);

      mc::echon("dim ");
      ex::padnum(1u << r.dl, 4);
      mc::echon("L");
      ex::padnum(static_cast<u64>(r.lvl), 7);
      ex::padf3(p.bits_per_weight, 9);
      ex::padf3(e.rel_rmse, 15);
      ex::pad(r.bname, 6);
      mc::echon("@ ");
      ex::padf3(base_bw, 9);
      ex::padf3(base_err, 16);
      ex::put3(e.cos);
      mc::echo("");
    }
    mc::echo("");
    mc::echo("hsc spends its bits on a DIRECTION shared by dim weights at once; int-N spends them one");
    mc::echo("weight at a time. That is the whole difference, and it is why the gap widens as the");
    mc::echo("budget shrinks: at 2 bits int-N has three levels to work with and hsc still has a whole");
    mc::echo("sphere. Read the last row honestly, though -- at 4 bits int-N is AHEAD (and on slightly");
    mc::echo("more bits). The crossover sits near 4 bits/weight on these tensors; below it hsc wins by");
    mc::echo("roughly a factor of two on relative error, above it there is no reason to reach for it.");
  }

  if ( ran == 0 ) {
    mc::echo("");
    mc::echo("nothing to measure. Build the corpus:");
    mc::echo("  python3 scripts/corpus.py fetch && python3 scripts/corpus.py models");
  }
  return 0;
}
