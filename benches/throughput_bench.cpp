// hsc throughput — the headline table: cyc/byte AND MB/s, encode next to decode, across the
// whole (mode x dim x level x transform) grid, plus hot-vs-cold scratch, a size sweep, a
// ragged (non-block-multiple) tail, and a sustained soak with thermal drift.
//   duck build benches/throughput_bench.cpp --perf --fp -i ../micron -i ../micron/src && ./bin/throughput_bench
#include "_corpus.hpp"

#include "_bench_common.hpp"

// The grid is wide, so each cell runs a deliberately short measurement: bench_one still takes
// the median of 7, it just uses fewer reps per sample than the single-cell benches do.
static constexpr u64 k_reps = 12;      // reps per perf-counter sample
static constexpr u64 k_treps = 6;      // reps per wall-clock sample

static constexpr usize k_big = 1 << 19;      // 512 KiB, the top of the size sweep
static u8 g_in[k_big];
static u8 g_z[2 << 20];
static u8 g_out[2 << 20];

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%
// one (encode, decode) pair -> one throughput row

static void
row_bytes(const char *corpus, const char *impl, const u8 *d, usize n, const hsc::hopf_opts &o, hsc::hopf_scratch &sc) noexcept
{
  auto enc = [&]() {
    const usize w = hsc::hopf_into(hsc::bytes{ d, n }, o, g_z, sizeof(g_z), sc);
    mb::sink_size(w);
    mb::clobber(g_z);
  };
  enc();      // build the skeleton outside every measured loop
  const usize zn = hsc::hopf_into(hsc::bytes{ d, n }, o, g_z, sizeof(g_z), sc);

  auto dec = [&]() {
    auto w = hsc::unhopf(hsc::bytes{ g_z, zn }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
    mb::sink_bool(w.is_first());
    mb::clobber(g_out);
  };
  dec();

  const mb::row re = mb::bench_one(corpus, impl, n, n, enc, k_reps);
  const f64 ens = mb::time_one_ns(enc, k_treps);
  const mb::row rd = mb::bench_one(corpus, impl, n, n, dec, k_reps);
  const f64 dns = mb::time_one_ns(dec, k_treps);

  mb::print_tput_row(mb::tput{ corpus, impl, zn, zn ? (f64)n / (f64)zn : 0.0, re.cyc_per_op / (f64)n, mb::mbps(n, ens),
                               rd.cyc_per_op / (f64)n, mb::mbps(n, dns), re.ipc, re.bmiss_rate, ens > 0.0 ? re.cyc_per_op / ens : 0.0 });
}

static void
row_floats(const char *corpus, const char *impl, const f32 *d, usize nf, const hsc::hopf_opts &o, hsc::hopf_scratch &sc) noexcept
{
  const usize n = nf * 4;      // bytes of input, so MB/s is comparable with the byte lanes
  auto enc = [&]() {
    const max_t w = hsc::hopf_into(hsc::floats{ d, nf }, o, g_z, sizeof(g_z), sc);
    mb::sink_size(static_cast<usize>(w));
    mb::clobber(g_z);
  };
  enc();
  const max_t zw = hsc::hopf_into(hsc::floats{ d, nf }, o, g_z, sizeof(g_z), sc);
  if ( zw < 0 ) return;
  const usize zn = static_cast<usize>(zw);

  auto dec = [&]() {
    auto w = hsc::unhopf(hsc::bytes{ g_z, zn }, hsc::wfloats{ (f32 *)g_out, sizeof(g_out) / 4 }, sc);
    mb::sink_bool(w.is_first());
    mb::clobber(g_out);
  };
  dec();

  const mb::row re = mb::bench_one(corpus, impl, n, n, enc, k_reps);
  const f64 ens = mb::time_one_ns(enc, k_treps);
  const mb::row rd = mb::bench_one(corpus, impl, n, n, dec, k_reps);
  const f64 dns = mb::time_one_ns(dec, k_treps);

  mb::print_tput_row(mb::tput{ corpus, impl, zn, zn ? (f64)n / (f64)zn : 0.0, re.cyc_per_op / (f64)n, mb::mbps(n, ens),
                               rd.cyc_per_op / (f64)n, mb::mbps(n, dns), re.ipc, re.bmiss_rate, ens > 0.0 ? re.cyc_per_op / ens : 0.0 });
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%

static char g_name[64];

static const char *
name_for(const char *pfx, u32 dim, i32 lvl, bool tf) noexcept
{
  u32 p = 0;
  for ( const char *q = pfx; *q; ++q ) g_name[p++] = *q;
  g_name[p++] = 'd';
  if ( dim >= 10 ) g_name[p++] = static_cast<char>('0' + dim / 10);
  g_name[p++] = static_cast<char>('0' + dim % 10);
  g_name[p++] = ' ';
  g_name[p++] = 'L';
  g_name[p++] = static_cast<char>('0' + lvl);
  if ( tf ) {
    g_name[p++] = ' ';
    g_name[p++] = 'r';
    g_name[p++] = 'o';
    g_name[p++] = 't';
  }
  g_name[p] = '\0';
  return g_name;
}

int
main()
{
  mb::pin_cpu0();
  mb::print_preamble("hsc throughput — cyc/byte and MB/s, encode and decode");
  hc::generate();
  mb::spin_up();      // reach the turbo clock before the first wall-clock cell (see GHz column)

  hsc::hopf_scratch sc;

  // %%%% the dim x level grid, bin mode, on the two extreme corpora
  micron::io::println("-- bin: dimension x level (noise = incompressible worst case, smooth = typical) --");
  mb::print_tput_header();
  for ( u32 dl = 2; dl <= 6; ++dl ) {
    for ( i32 lvl : { 3, 6 } ) {
      const hsc::hopf_opts o{ .level = lvl, .dim_log2 = dl };
      row_bytes("noise64k", name_for("bin ", 1u << dl, lvl, false), hc::g_noise, hc::k_n, o, sc);
    }
  }
  micron::io::println("");

  // %%%% the level sweep at the reference dimension
  micron::io::println("-- bin d8: level sweep (rate knob) --");
  mb::print_tput_header();
  for ( i32 lvl : { 1, 3, 5, 6, 7, 9 } ) {
    const hsc::hopf_opts o{ .level = lvl, .dim_log2 = 3 };
    row_bytes("noise64k", name_for("bin ", 8, lvl, false), hc::g_noise, hc::k_n, o, sc);
  }
  micron::io::println("");

  // %%%% corpus sensitivity + the H*D transform, on and off
  micron::io::println("-- bin: corpus x transform --");
  mb::print_tput_header();
  for ( usize i = 0; i < hc::corpora_count; ++i ) {
    const hc::corpus &c = hc::corpora[i];
    for ( bool tf : { false, true } ) {
      const hsc::hopf_opts o{ .level = 6, .dim_log2 = 3, .transform = tf };
      row_bytes(c.name, name_for("bin ", 8, 6, tf), c.d, c.n, o, sc);
    }
  }
  micron::io::println("");

  // %%%% the typed float lanes
  micron::io::println("-- float lanes: vec / unit / quotient (MB/s over INPUT bytes) --");
  mb::print_tput_header();
  {
    const usize nf = hc::k_n / 4;
    for ( u32 dl = 2; dl <= 6; ++dl ) {
      const hsc::hopf_opts ov{ .m = hsc::mode::vec, .level = 6, .dim_log2 = dl };
      row_floats("gauss16k", name_for("vec ", 1u << dl, 6, false), hc::g_gauss, nf, ov, sc);
    }
    for ( u32 dl = 2; dl <= 6; ++dl ) {
      const hsc::hopf_opts ou{ .m = hsc::mode::unit, .level = 6, .dim_log2 = dl };
      row_floats("unit16k", name_for("unit ", 1u << dl, 6, false), hc::g_unit, nf, ou, sc);
    }
    for ( bool tf : { false, true } ) {
      const hsc::hopf_opts ov{ .m = hsc::mode::vec, .level = 6, .dim_log2 = 3, .transform = tf };
      row_floats("onehot16k", name_for("vec ", 8, 6, tf), hc::g_onehot, nf, ov, sc);
    }
    for ( i32 lvl : { 3, 6, 7 } ) {
      const hsc::hopf_opts oq{ .m = hsc::mode::quotient, .level = lvl };
      row_floats("unit16k", name_for("quot ", 4, lvl, false), hc::g_unit, nf, oq, sc);
    }
    for ( i32 lvl : { 3, 6, 7 } ) {
      const hsc::hopf_opts oq4{ .m = hsc::mode::quat, .level = lvl };
      row_floats("fiber16k", name_for("quat ", 8, lvl, false), hc::g_fiber8, nf, oq4, sc);
    }
    for ( i32 lvl : { 3, 6, 7 } ) {
      const hsc::hopf_opts oq8{ .m = hsc::mode::oct, .level = lvl };
      row_floats("fiber16k", name_for("oct ", 16, lvl, false), hc::g_fiber16, nf, oq8, sc);
    }
  }
  micron::io::println("");

  // %%%% refine and gain_bits, which no bench has ever touched through the public API
  micron::io::println("-- encoder knobs: refine, gain_bits --");
  mb::print_tput_header();
  {
    const hsc::hopf_opts r0{ .level = 6, .dim_log2 = 3, .refine = 0 };
    const hsc::hopf_opts r1{ .level = 6, .dim_log2 = 3, .refine = 1 };
    row_bytes("noise64k", "bin d8 L6 refine0", hc::g_noise, hc::k_n, r0, sc);
    row_bytes("noise64k", "bin d8 L6 refine1", hc::g_noise, hc::k_n, r1, sc);
    for ( u32 gb : { 4u, 8u, 16u } ) {
      const hsc::hopf_opts o{ .level = 6, .dim_log2 = 3, .gain_bits = gb };
      const char *nm = gb == 4 ? "bin d8 L6 g4" : (gb == 8 ? "bin d8 L6 g8" : "bin d8 L6 g16");
      row_bytes("noise64k", nm, hc::g_noise, hc::k_n, o, sc);
    }
  }
  micron::io::println("");

  // %%%% size sweep + a ragged tail (n not a multiple of the block size)
  micron::io::println("-- size sweep (1 KiB .. 512 KiB) and a ragged tail --");
  mb::print_tput_header();
  {
    for ( usize i = 0; i < k_big; ++i ) g_in[i] = hc::g_noise[i % hc::k_n];
    const hsc::hopf_opts o{ .level = 6, .dim_log2 = 3 };
    for ( usize n = 1024; n <= k_big; n <<= 2 ) {
      const char *nm = n == 1024 ? "1KiB" : (n == 4096 ? "4KiB" : (n == 16384 ? "16KiB" : (n == 65536 ? "64KiB" : "256KiB")));
      row_bytes(nm, "bin d8 L6", g_in, n, o, sc);
    }
    row_bytes("ragged", "bin d8 L6 (n=65533)", g_in, 65533, o, sc);
    const hsc::hopf_opts o64{ .level = 6, .dim_log2 = 6 };
    row_bytes("ragged", "bin d64 L6 (n=65533)", g_in, 65533, o64, sc);
  }
  micron::io::println("");

  // %%%% cold scratch: what a caller pays who does not hold a hopf_scratch
  micron::io::println("-- cold scratch (skeleton rebuilt per call) vs hot --");
  mb::print_tput_header();
  {
    for ( u32 dl : { 3u, 4u, 6u } ) {
      const hsc::hopf_opts o{ .level = 6, .dim_log2 = dl };
      auto enc = [&]() {
        hsc::hopf_scratch cold;
        const usize w = hsc::hopf_into(hsc::bytes{ hc::g_noise, hc::k_n }, o, g_z, sizeof(g_z), cold);
        mb::sink_size(w);
        mb::clobber(g_z);
      };
      enc();
      const usize zn = hsc::hopf_into(hsc::bytes{ hc::g_noise, hc::k_n }, o, g_z, sizeof(g_z), sc);
      auto dec = [&]() {
        hsc::hopf_scratch cold;
        auto w = hsc::unhopf(hsc::bytes{ g_z, zn }, hsc::wbytes{ g_out, sizeof(g_out) }, cold);
        mb::sink_bool(w.is_first());
        mb::clobber(g_out);
      };
      dec();
      const mb::row re = mb::bench_one("noise64k", "cold", hc::k_n, hc::k_n, enc, 4);
      const f64 ens = mb::time_one_ns(enc, 3);
      const mb::row rd = mb::bench_one("noise64k", "cold", hc::k_n, hc::k_n, dec, 4);
      const f64 dns = mb::time_one_ns(dec, 3);
      mb::print_tput_row(mb::tput{ "noise64k", name_for("cold ", 1u << dl, 6, false), zn, (f64)hc::k_n / (f64)zn,
                                   re.cyc_per_op / (f64)hc::k_n, mb::mbps(hc::k_n, ens), rd.cyc_per_op / (f64)hc::k_n,
                                   mb::mbps(hc::k_n, dns), re.ipc, re.bmiss_rate, ens > 0.0 ? re.cyc_per_op / ens : 0.0 });
    }
  }
  micron::io::println("");

  // %%%% sustained rate + thermal droop (turbo is on and the governor is schedutil here)
  micron::io::println("-- sustained soak: median MB/s, first vs last decile, drift --");
  mb::print_soak_header();
  {
    const hsc::hopf_opts o{ .level = 6, .dim_log2 = 3 };
    (void)hsc::hopf_into(hsc::bytes{ hc::g_noise, hc::k_n }, o, g_z, sizeof(g_z), sc);
    const usize zn = hsc::hopf_into(hsc::bytes{ hc::g_noise, hc::k_n }, o, g_z, sizeof(g_z), sc);
    mb::print_soak_row("encode noise64k", "bin d8 L6",
                       mb::soak_one(
                           [&]() {
                             mb::sink_size(hsc::hopf_into(hsc::bytes{ hc::g_noise, hc::k_n }, o, g_z, sizeof(g_z), sc));
                             mb::clobber(g_z);
                           },
                           hc::k_n, 2000));
    mb::print_soak_row("decode noise64k", "bin d8 L6",
                       mb::soak_one(
                           [&]() {
                             auto w = hsc::unhopf(hsc::bytes{ g_z, zn }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
                             mb::sink_bool(w.is_first());
                             mb::clobber(g_out);
                           },
                           hc::k_n, 2000));
  }

  mb::print_epilogue();
  return 0;
}
