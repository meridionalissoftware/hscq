

#include "_files.hpp"

#include "_bench_common.hpp"

#ifndef HSC_CORPUS_DIR
#define HSC_CORPUS_DIR "corpus"
#endif

static constexpr usize k_cap = 4u << 20;
static constexpr usize k_work = 1u << 19;

static u8 g_in[k_cap];
static u8 g_z[2 * k_cap];
static u8 g_out[k_cap];

struct cfg {
  u32 dl;
  i32 lvl;
};

static constexpr cfg k_cfgs[] = {
  { 2, 1 },  { 2, 2 },  { 2, 3 }, { 2, 4 }, { 2, 5 }, { 2, 6 }, { 2, 7 }, { 2, 8 }, { 2, 9 }, { 2, 10 }, { 2, 11 }, { 2, 12 },
  { 2, 13 }, { 2, 14 }, { 3, 1 }, { 3, 2 }, { 3, 3 }, { 3, 4 }, { 3, 5 }, { 3, 6 }, { 3, 7 }, { 3, 8 },  { 3, 9 },  { 3, 10 },
  { 3, 11 }, { 3, 12 }, { 4, 1 }, { 4, 2 }, { 4, 3 }, { 4, 4 }, { 4, 5 }, { 4, 6 }, { 4, 7 }, { 4, 8 },  { 4, 9 },  { 4, 10 },
  { 4, 11 }, { 5, 1 },  { 5, 2 }, { 5, 3 }, { 5, 4 }, { 5, 5 }, { 5, 6 }, { 5, 7 }, { 5, 8 }, { 5, 9 },  { 5, 10 }, { 6, 1 },
  { 6, 2 },  { 6, 3 },  { 6, 4 }, { 6, 5 }, { 6, 6 }, { 6, 7 }, { 6, 8 }, { 6, 9 },
};
static constexpr usize k_ncfg = sizeof(k_cfgs) / sizeof(k_cfgs[0]);

struct cell {
  u64 out_bytes;
  f64 ratio;
  f64 bits_byte;
  f64 rmse;
  f64 psnr;
  u32 shape_bits;
  bool ok;
};

static cell g_cell[hf::corpus_count][k_ncfg];
static usize g_size[hf::corpus_count];

static f64
psnr_of(f64 rmse) noexcept
{
  if ( rmse < 1e-9 ) return 99.99;
  return 20.0 * micron::log10(255.0 / rmse);
}

static void
print_f3(const char *tag, f64 v) noexcept
{
  const bool neg = v < 0;
  if ( neg ) v = -v;
  const u64 w = static_cast<u64>(v);
  const u64 f = static_cast<u64>((v - static_cast<f64>(w)) * 1000.0 + 0.5);
  micron::io::print(tag, neg ? "-" : "", w, ".", static_cast<u32>(f / 100), static_cast<u32>((f / 10) % 10), static_cast<u32>(f % 10));
}

static void
grid_header() noexcept
{
  mb::line ln;
  ln.s_lj_at("corpus", 13);
  ln.s_lj_at("cfg", 24);
  ln.s_at("shape", 30);
  ln.s_at("out(B)", 40);
  ln.s_at("ratio", 48);
  ln.s_at("bits/B", 57);
  ln.s_at("rmse", 65);
  ln.s_at("psnr", 73);
  ln.s_at("dg", 77);
  micron::io::println(ln.str());
  micron::io::println("--------------------------------------------------------------------------------");
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

int
main(int argc, char **argv)
{
  const char *root = argc > 1 ? argv[1] : HSC_CORPUS_DIR;
  mb::pin_cpu0();
  mb::print_preamble("hsc compression ratio and distortion on real data (bin mode)");
  micron::io::println("corpus root: ", root, "   working prefix: 512 KiB per file");
  micron::io::println("FIXED RATE: the ratio column is constant down each config -- it does not depend on the data.");
  micron::io::println("What the data decides is rmse/psnr. dg=1 marks shape_bits==0 (gain-only: loss, not compression).");
  micron::io::println("");

  hsc::hopf_scratch sc;

  for ( usize ci = 0; ci < k_ncfg; ++ci ) {
    const hsc::hopf_opts o{ .level = k_cfgs[ci].lvl, .dim_log2 = k_cfgs[ci].dl };
    auto ri = hsc::rate(o, sc);
    const u32 sbits = ri.is_first() ? ri.cast<hsc::rate_info>().shape_bits : 0u;
    for ( usize fi = 0; fi < hf::corpus_count; ++fi ) {
      cell &cl = g_cell[fi][ci];
      cl.ok = false;
      cl.shape_bits = sbits;
      const max_t got = hf::slurp_at(root, hf::corpus_files[fi].file, g_in, k_cap);
      if ( got <= 0 ) {
        g_size[fi] = 0;
        continue;
      }
      const usize n = static_cast<usize>(got) < k_work ? static_cast<usize>(got) : k_work;
      g_size[fi] = n;
      const usize zn = hsc::hopf_into(hsc::bytes{ g_in, n }, o, g_z, sizeof(g_z), sc);
      if ( zn == 0 ) continue;
      auto r = hsc::unhopf(hsc::bytes{ g_z, zn }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
      if ( !r.is_first() ) continue;
      f64 se = 0;
      for ( usize i = 0; i < n; ++i ) {
        const f64 e = static_cast<f64>(g_out[i]) - static_cast<f64>(g_in[i]);
        se += e * e;
      }
      cl.rmse = __builtin_sqrt(se / static_cast<f64>(n));
      cl.psnr = psnr_of(cl.rmse);
      cl.out_bytes = zn;
      //  two deliberate conventions in one row: ratio is FRAMED (whole stream, end-to-end truth),
      //  bits_byte is PAYLOAD (record bits only, the number the rate model states) -- so
      //  ratio != 8/bits_byte by the 48-byte frame share; the report caption says so too
      cl.ratio = static_cast<f64>(n) / static_cast<f64>(zn);
      cl.bits_byte = static_cast<f64>(zn - 48) * 8.0 / static_cast<f64>(n);
      cl.ok = true;

      if ( ri.is_first() && zn != hsc::bound(n, o, sc) ) micron::io::println("!! RATE MODEL MISMATCH at ", cfg_name(k_cfgs[ci]));
    }
  }

  micron::io::println("-- fixed-rate grid --");
  micron::io::println("");
  for ( usize fi = 0; fi < hf::corpus_count; ++fi ) {
    if ( g_size[fi] == 0 ) {
      micron::io::println(hf::corpus_files[fi].label, ": skip (no ", hf::corpus_files[fi].file, " under ", root, ")");
      continue;
    }
    micron::io::println("");
    micron::io::println("[", hf::corpus_files[fi].label, "]  ", hf::corpus_files[fi].note, "   n=", g_size[fi], " B");
    grid_header();
    for ( usize ci = 0; ci < k_ncfg; ++ci ) {
      const cell &cl = g_cell[fi][ci];
      mb::line ln;
      ln.s_lj_at(hf::corpus_files[fi].label, 13);
      ln.s_lj_at(cfg_name(k_cfgs[ci]), 24);
      ln.u_at(cl.shape_bits, 30);
      if ( !cl.ok ) {
        ln.s_at("fail", 40);
        micron::io::println(ln.str());
        continue;
      }
      ln.u_at(cl.out_bytes, 40);
      ln.f2_at(mb::to_fmt2(cl.ratio), 48);
      ln.f2_at(mb::to_fmt2(cl.bits_byte), 57);
      ln.f2_at(mb::to_fmt2(cl.rmse), 65);
      ln.f2_at(mb::to_fmt2(cl.psnr), 73);
      ln.u_at(cl.shape_bits == 0 ? 1u : 0u, 77);
      micron::io::println(ln.str());
    }
  }

  micron::io::println("");
  micron::io::println("-- quality-matched: the cheapest config meeting a distortion target --");
  micron::io::println("   (degenerate cells excluded; this is the only data-dependent ratio in the report)");
  micron::io::println("");
  {
    mb::line ln;
    ln.s_lj_at("corpus", 13);
    ln.s_at("target", 22);
    ln.pad_to(28, 0);
    ln.s_lj_at("best cfg", 36);
    ln.s_at("ratio", 44);
    ln.s_at("bits/B", 53);
    ln.s_at("psnr", 61);
    micron::io::println(ln.str());
    micron::io::println("--------------------------------------------------------------");
  }
  for ( usize fi = 0; fi < hf::corpus_count; ++fi ) {
    if ( g_size[fi] == 0 ) continue;
    for ( f64 target : { 20.0, 30.0, 40.0 } ) {
      usize best = k_ncfg;
      for ( usize ci = 0; ci < k_ncfg; ++ci ) {
        const cell &cl = g_cell[fi][ci];
        if ( !cl.ok || cl.shape_bits == 0 || cl.psnr < target ) continue;
        if ( best == k_ncfg || cl.bits_byte < g_cell[fi][best].bits_byte ) best = ci;
      }
      mb::line ln;
      ln.s_lj_at(hf::corpus_files[fi].label, 13);
      ln.u_at(static_cast<u64>(target), 20);
      ln.s(" dB");
      ln.pad_to(28, 0);
      if ( best == k_ncfg ) {
        ln.s_lj_at("  none in grid", 40);
        micron::io::println(ln.str());
        continue;
      }
      ln.s_lj_at(cfg_name(k_cfgs[best]), 36);
      ln.f2_at(mb::to_fmt2(g_cell[fi][best].ratio), 44);
      ln.f2_at(mb::to_fmt2(g_cell[fi][best].bits_byte), 53);
      ln.f2_at(mb::to_fmt2(g_cell[fi][best].psnr), 61);
      micron::io::println(ln.str());
    }
  }

  micron::io::println("");
  micron::io::println("-- rate reference: bits per block by (dim, level), data-independent by construction --");
  {
    mb::line ln;
    ln.s_lj_at("cfg", 12);
    ln.s_at("shape", 20);
    ln.s_at("gain", 27);
    ln.s_at("record", 35);
    ln.s_at("bits/B", 44);
    ln.s_at("ratio", 52);
    micron::io::println(ln.str());
  }
  for ( usize ci = 0; ci < k_ncfg; ++ci ) {
    const hsc::hopf_opts o{ .level = k_cfgs[ci].lvl, .dim_log2 = k_cfgs[ci].dl };
    auto r = hsc::rate(o, sc);
    if ( !r.is_first() ) continue;
    const hsc::rate_info ri = r.cast<hsc::rate_info>();
    mb::line ln;
    ln.s_lj_at(cfg_name(k_cfgs[ci]), 12);
    ln.u_at(ri.shape_bits, 20);
    ln.u_at(ri.gain_bits, 27);
    ln.u_at(ri.record_bits, 35);
    ln.f2_at(mb::to_fmt2(ri.bits_per_input_byte), 44);
    ln.f2_at(mb::to_fmt2(ri.ratio), 52);
    micron::io::println(ln.str());
  }

  micron::io::println("");
  (void)print_f3;
  return 0;
}
