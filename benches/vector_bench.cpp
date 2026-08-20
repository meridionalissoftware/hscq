//  hsc on real open-weight model tensors: what a fixed-rate spherical code costs a weight matrix
//  and an embedding table. scripts/corpus.py exports tensors from safetensors as raw f32 into
//  corpus/models/; this bench runs them through the vec and unit lanes and reports bits/weight
//  against the int8/int4/int2 rates a quantizer would charge for the same storage.
//
//  Two modes:
//    ./bin/vector_bench                       grid over corpus/models/manifest-listed tensors
//    ./bin/vector_bench --in a.f32 --out b.f32 --mode unit --dim 8 --level 6
//                                             round-trip one file and write the decoded f32 back,
//                                             so scripts/vector_eval.py can do cosine and recall@k
//                                             in numpy instead of here
//
//  PER TENSOR, NEVER ONE BIG STREAM. vec mode takes gmax over the WHOLE input in a first pass
//  (hopf.hpp), so concatenating tensors lets one outlier weight crush the gain resolution of every
//  other tensor. Each file is encoded on its own; that is what makes the numbers mean anything.
//
//  THE QUOTIENT FAMILY (quotient, quat, oct) IS EXCLUDED, deliberately. Those modes quotient out
//  a fiber symmetry on complex/quaternion/octonion pairs; weights and embeddings carry no such
//  symmetry, so the modes would discard 1/3/7 REAL dimensions per pair rather than redundancy,
//  and their decodes return canonical representatives that no coordinate-wise error metric can
//  fairly score.
//
//    duck build benches/vector_bench.cpp --perf --fp -i ../micron -i ../micron/src
//  NOTE the include order: hsc before bbench (__bitwise macro, see hopf_bench.cpp).
#include "_files.hpp"

#include "_bench_common.hpp"

#ifndef HSC_CORPUS_DIR
#define HSC_CORPUS_DIR "corpus"
#endif

static constexpr usize k_cap = 48u << 20;      //  12 Mi floats: enough for a 30k x 384 embedding table
static constexpr usize k_zcap = 32u << 20;

static u8 g_in[k_cap];
static u8 g_z[k_zcap];
static u8 g_out[k_cap];

//  tensors the grid runs when no --in is given; scripts/corpus.py writes these names
struct tfile {
  const char *label;
  const char *file;
};

static constexpr tfile k_tensors[] = {
  { "MiniLM embed", "models/all-MiniLM-L6-v2.embeddings_word_embeddings_weight.f32" },            //  30522 x 384
  { "MiniLM mlp", "models/all-MiniLM-L6-v2.encoder_layer_0_intermediate_dense_weight.f32" },      //  1536 x 384
  { "SmolLM embed", "models/SmolLM2-135M.model_embed_tokens_weight.f32" },                        //  49152 x 576, read as a prefix
  { "SmolLM mlp", "models/SmolLM2-135M.model_layers_0_mlp_gate_proj_weight.f32" },                //  1536 x 576
};
static constexpr usize k_ntensor = sizeof(k_tensors) / sizeof(k_tensors[0]);

struct cfg {
  u32 dl;
  i32 lvl;
};

static constexpr cfg k_cfgs[] = {
  { 3, 3 }, { 3, 5 }, { 3, 6 }, { 3, 7 }, { 3, 9 },      //
  { 4, 3 }, { 4, 5 }, { 4, 6 }, { 4, 7 }, { 4, 9 },      //
  { 5, 3 }, { 5, 5 }, { 5, 6 }, { 5, 7 }, { 5, 9 },      //
  { 6, 3 }, { 6, 5 }, { 6, 6 }, { 6, 7 }, { 6, 9 },      //
};
static constexpr usize k_ncfg = sizeof(k_cfgs) / sizeof(k_cfgs[0]);

static bool
streq(const char *a, const char *b) noexcept
{
  while ( *a && *a == *b ) {
    ++a;
    ++b;
  }
  return *a == *b;
}

static i64
to_i(const char *s) noexcept
{
  i64 v = 0;
  bool neg = false;
  if ( *s == '-' ) {
    neg = true;
    ++s;
  }
  while ( *s >= '0' && *s <= '9' ) v = v * 10 + (*s++ - '0');
  return neg ? -v : v;
}

//  L2-normalize every dim-element block in place; returns the number of blocks left at zero norm
static usize
normalize_blocks(f32 *v, usize n, u32 dim) noexcept
{
  usize zero = 0;
  for ( usize b = 0; b + dim <= n; b += dim ) {
    f64 s = 0;
    for ( u32 c = 0; c < dim; ++c ) s += static_cast<f64>(v[b + c]) * static_cast<f64>(v[b + c]);
    if ( s <= 0.0 ) {
      v[b] = 1.0f;      //  unit mode rejects a zero-norm block (bad_value); pin it to a basis vector
      ++zero;
      continue;
    }
    const f64 inv = 1.0 / __builtin_sqrt(s);
    for ( u32 c = 0; c < dim; ++c ) v[b + c] = static_cast<f32>(static_cast<f64>(v[b + c]) * inv);
  }
  return zero;
}

static void
grid_header() noexcept
{
  mb::line ln;
  ln.s_lj_at("tensor", 15);
  ln.s_lj_at("lane", 21);
  ln.s_lj_at("cfg", 30);
  ln.s_at("out(B)", 41);
  ln.s_at("bits/w", 50);
  ln.s_at("vs fp32", 59);
  ln.s_at("rel rmse", 69);
  ln.s_at("cos", 77);
  micron::io::println(ln.str());
  micron::io::println("---------------------------------------------------------------------------------");
}

static const char *
cfg_name(const cfg &c) noexcept
{
  static char nm[32];
  u32 k = 0;
  nm[k++] = 'd';
  const u32 dim = 1u << c.dl;
  if ( dim >= 10 ) nm[k++] = static_cast<char>('0' + (dim / 10) % 10);
  nm[k++] = static_cast<char>('0' + dim % 10);
  nm[k++] = ' ';
  nm[k++] = 'L';
  if ( c.lvl >= 10 ) nm[k++] = static_cast<char>('0' + c.lvl / 10);
  nm[k++] = static_cast<char>('0' + c.lvl % 10);
  nm[k] = '\0';
  return nm;
}

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  filter mode: one file in, one file out, so numpy can score it

static int
filter(const char *in, const char *out, hsc::mode m, u32 dl, i32 lvl, bool norm)
{
  const max_t got = hf::slurp_into(in, g_in, k_cap);
  if ( got <= 0 ) {
    micron::io::println("vector_bench: cannot read ", in, " (", got == -2 ? "larger than buffer" : "missing", ")");
    return 2;
  }
  const usize nf = static_cast<usize>(got) / 4;
  f32 *fin = reinterpret_cast<f32 *>(g_in);
  const u32 dim = 1u << dl;
  const usize whole = (nf / dim) * dim;      //  unit/quotient reject a partial block
  if ( norm || m == hsc::mode::unit ) normalize_blocks(fin, whole, dim);

  hsc::hopf_scratch sc;
  const hsc::hopf_opts o{ .m = m, .level = lvl, .dim_log2 = dl };
  const max_t zn = hsc::hopf_into(hsc::floats{ fin, whole }, o, g_z, sizeof(g_z), sc);
  if ( zn < 0 ) {
    micron::io::println("vector_bench: encode failed: ", hsc::error_name(hsc::as_error(zn)));
    return 3;
  }
  auto r = hsc::unhopf(hsc::bytes{ g_z, static_cast<usize>(zn) }, hsc::wfloats{ reinterpret_cast<f32 *>(g_out), whole }, sc);
  if ( !r.is_first() ) {
    micron::io::println("vector_bench: decode failed: ", hsc::error_name(r.cast<hsc::error>()));
    return 4;
  }
  if ( hf::spill(out, g_out, whole * 4) < 0 ) {
    micron::io::println("vector_bench: cannot write ", out);
    return 5;
  }
  //  the reference input as the encoder actually saw it (normalized, whole blocks only), so the
  //  scorer compares like with like
  char ref[hf::k_path_max];
  usize k = 0;
  for ( const char *p = out; *p && k + 5 < hf::k_path_max; ++p ) ref[k++] = *p;
  ref[k++] = '.';
  ref[k++] = 'r';
  ref[k++] = 'e';
  ref[k++] = 'f';
  ref[k] = '\0';
  (void)hf::spill(ref, g_in, whole * 4);
  micron::io::println("elems=", whole, " stream=", static_cast<u64>(zn),
                      " bits_per_weight_x1000=", static_cast<u64>(static_cast<f64>(zn - 48) * 8000.0 / static_cast<f64>(whole)));
  return 0;
}

int
main(int argc, char **argv)
{
  const char *in = nullptr;
  const char *out = nullptr;
  const char *root = HSC_CORPUS_DIR;
  hsc::mode m = hsc::mode::unit;
  u32 dl = 3;
  i32 lvl = 6;
  bool norm = false;
  for ( int i = 1; i < argc; ++i ) {
    if ( streq(argv[i], "--in") && i + 1 < argc )
      in = argv[++i];
    else if ( streq(argv[i], "--out") && i + 1 < argc )
      out = argv[++i];
    else if ( streq(argv[i], "--root") && i + 1 < argc )
      root = argv[++i];
    else if ( streq(argv[i], "--dim") && i + 1 < argc )
      dl = static_cast<u32>(to_i(argv[++i]));
    else if ( streq(argv[i], "--level") && i + 1 < argc )
      lvl = static_cast<i32>(to_i(argv[++i]));
    else if ( streq(argv[i], "--normalize") )
      norm = true;
    else if ( streq(argv[i], "--mode") && i + 1 < argc ) {
      ++i;
      m = streq(argv[i], "vec") ? hsc::mode::vec : (streq(argv[i], "unit") ? hsc::mode::unit : hsc::mode::vec);
    }
  }
  if ( in && out ) return filter(in, out, m, dl, lvl, norm);

  mb::pin_cpu0();
  mb::print_preamble("hsc on open-weight model tensors (vec and unit lanes; quotient excluded on purpose)");
  micron::io::println("corpus root: ", root, "   fp32 = 32 bits/weight, fp16 = 16, int8 = 8, int4 = 4, int2 = 2");
  micron::io::println("bits/w is the RATE hsc charges; rel rmse and cos are what it costs. Rate is fixed:");
  micron::io::println("it depends on (dim, level) alone, so compare a row against the int-N with the same bits/w.");
  micron::io::println("");

  hsc::hopf_scratch sc;
  for ( usize ti = 0; ti < k_ntensor; ++ti ) {
    const max_t got = hf::slurp_prefix_at(root, k_tensors[ti].file, g_in, k_cap);
    if ( got <= 0 ) {
      micron::io::println(k_tensors[ti].label, ": skip (missing: ", k_tensors[ti].file, ")");
      continue;
    }
    const usize nf_all = static_cast<usize>(got) / 4;
    micron::io::println("");
    micron::io::println("[", k_tensors[ti].label, "]  ", nf_all, " f32 read (", static_cast<u64>(got), " B)");
    grid_header();
    for ( u32 lane = 0; lane < 2; ++lane ) {
      const hsc::mode lm = lane == 0 ? hsc::mode::vec : hsc::mode::unit;
      u32 loaded_dim = 0;      //  the unit lane normalizes in place, and the block size sets the norms
      if ( lane == 1 ) loaded_dim = 0;
      for ( usize ci = 0; ci < k_ncfg; ++ci ) {
        const u32 dim = 1u << k_cfgs[ci].dl;
        const usize nf = (nf_all / dim) * dim;
        if ( lm == hsc::mode::unit && loaded_dim != dim ) {
          (void)hf::slurp_prefix_at(root, k_tensors[ti].file, g_in, k_cap);
          normalize_blocks(reinterpret_cast<f32 *>(g_in), nf, dim);
          loaded_dim = dim;
        }
        f32 *fin = reinterpret_cast<f32 *>(g_in);
        const hsc::hopf_opts o{ .m = lm, .level = k_cfgs[ci].lvl, .dim_log2 = k_cfgs[ci].dl };
        const max_t zn = hsc::hopf_into(hsc::floats{ fin, nf }, o, g_z, sizeof(g_z), sc);
        mb::line ln;
        ln.s_lj_at(k_tensors[ti].label, 15);
        ln.s_lj_at(lm == hsc::mode::vec ? "vec" : "unit", 21);
        ln.s_lj_at(cfg_name(k_cfgs[ci]), 30);
        if ( zn < 0 ) {
          ln.s_at(hsc::error_name(hsc::as_error(zn)), 41);
          micron::io::println(ln.str());
          continue;
        }
        auto r = hsc::unhopf(hsc::bytes{ g_z, static_cast<usize>(zn) }, hsc::wfloats{ reinterpret_cast<f32 *>(g_out), nf }, sc);
        if ( !r.is_first() ) {
          ln.s_at("decode-fail", 41);
          micron::io::println(ln.str());
          continue;
        }
        const f32 *fo = reinterpret_cast<const f32 *>(g_out);
        f64 se = 0, sg = 0, dot = 0, so = 0;
        for ( usize i = 0; i < nf; ++i ) {
          const f64 a = static_cast<f64>(fin[i]), b = static_cast<f64>(fo[i]);
          se += (b - a) * (b - a);
          sg += a * a;
          dot += a * b;
          so += b * b;
        }
        const f64 bpw = static_cast<f64>(zn - 48) * 8.0 / static_cast<f64>(nf);
        ln.u_at(static_cast<u64>(zn), 41);
        ln.f2_at(mb::to_fmt2(bpw), 50);
        ln.f2_at(mb::to_fmt2(bpw > 0 ? 32.0 / bpw : 0.0), 59);
        ln.f2_at(mb::to_fmt2(sg > 0 ? __builtin_sqrt(se / sg) : 0.0), 69);
        ln.f2_at(mb::to_fmt2((sg > 0 && so > 0) ? dot / __builtin_sqrt(sg * so) : 0.0), 77);
        micron::io::println(ln.str());
      }
    }
  }

  micron::io::println("");
  return 0;
}
