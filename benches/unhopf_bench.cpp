// hsc decoder — decode throughput (cyc/op, cyc/byte of decompressed output) per corpus, dim
// and level, scratch hot.
//   duck build benches/unhopf_bench.cpp --perf --fp -i ../micron -i ../micron/src && ./bin/unhopf_bench
#include "_corpus.hpp"

#include "_bench_common.hpp"

static u8 g_z[1 << 20];
static u8 g_out[1 << 20];

int
main()
{
  mb::pin_cpu0();
  mb::print_preamble("hsc unhopf — decode throughput (cyc/op, cyc/byte of output)");
  hc::generate();
  mb::print_header();

  hsc::hopf_scratch sc;

  for ( usize i = 0; i < hc::corpora_count; ++i ) {
    const hc::corpus &c = hc::corpora[i];

    struct cfg {
      const char *name;
      u32 dl;
      i32 lvl;
      bool tf;      // H*D transform (rows without it value-init false)
    };

    const cfg cfgs[] = {
      { "un d8 L3", 3, 3 }, { "un d8 L6", 3, 6 }, { "un d16 L6", 4, 6 }, { "un d8 L6 rot", 3, 6, true }, { "un d16 L6 rot", 4, 6, true }
    };
    for ( const cfg &f : cfgs ) {
      const hsc::hopf_opts o{ .level = f.lvl, .dim_log2 = f.dl, .transform = f.tf };
      const usize zn = hsc::hopf_into(hsc::bytes{ c.d, c.n }, o, g_z, sizeof(g_z), sc);
      mb::row r = mb::bench_one(
          c.name, f.name, c.n, c.n,
          [&]() {
            auto w = hsc::unhopf(hsc::bytes{ g_z, zn }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
            mb::sink_bool(w.is_first());
            mb::clobber(g_out);
          },
          c.mr);
      mb::print_row(r);
    }
  }

  // float lanes
  {
    const usize nf = hc::k_n / 4;
    for ( bool tf : { false, true } ) {
      const hsc::hopf_opts ou{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 3, .transform = tf };
      const max_t zn = hsc::hopf_into(hsc::floats{ hc::g_unit, nf }, ou, g_z, sizeof(g_z), sc);
      mb::row ru = mb::bench_one(
          "unit16k", tf ? "un unit d8 rot" : "un unit d8", nf * 4, nf * 4,
          [&]() {
            auto w = hsc::unhopf(hsc::bytes{ g_z, static_cast<usize>(zn) }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
            mb::sink_bool(w.is_first());
            mb::clobber(g_out);
          },
          1 << 8);
      mb::print_row(ru);
    }

    const hsc::hopf_opts oq{ .m = hsc::mode::quotient, .level = 6 };
    const max_t zq = hsc::hopf_into(hsc::floats{ hc::g_unit, nf }, oq, g_z, sizeof(g_z), sc);
    mb::row rq = mb::bench_one(
        "unit16k", "un quot", nf * 4, nf * 4,
        [&]() {
          auto w = hsc::unhopf(hsc::bytes{ g_z, static_cast<usize>(zq) }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
          mb::sink_bool(w.is_first());
          mb::clobber(g_out);
        },
        1 << 8);
    mb::print_row(rq);

    const hsc::hopf_opts oq4{ .m = hsc::mode::quat, .level = 6 };
    const max_t zq4 = hsc::hopf_into(hsc::floats{ hc::g_fiber8, nf }, oq4, g_z, sizeof(g_z), sc);
    mb::row rq4 = mb::bench_one(
        "fiber16k", "un quat", nf * 4, nf * 4,
        [&]() {
          auto w = hsc::unhopf(hsc::bytes{ g_z, static_cast<usize>(zq4) }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
          mb::sink_bool(w.is_first());
          mb::clobber(g_out);
        },
        1 << 8);
    mb::print_row(rq4);

    const hsc::hopf_opts oq8{ .m = hsc::mode::oct, .level = 6 };
    const max_t zq8 = hsc::hopf_into(hsc::floats{ hc::g_fiber16, nf }, oq8, g_z, sizeof(g_z), sc);
    mb::row rq8 = mb::bench_one(
        "fiber16k", "un oct", nf * 4, nf * 4,
        [&]() {
          auto w = hsc::unhopf(hsc::bytes{ g_z, static_cast<usize>(zq8) }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
          mb::sink_bool(w.is_first());
          mb::clobber(g_out);
        },
        1 << 8);
    mb::print_row(rq8);
  }

  mb::print_epilogue();
  return 0;
}
