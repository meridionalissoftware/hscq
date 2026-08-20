// Encode-side throughput per (mode, dim, level) cell over the synthetic corpora.
//
// NOTE the include order: hsc (via _corpus.hpp) before _bench_common.hpp. bbench pulls the linux
// perf headers, which #define __bitwise; that macro must not be live while micron's arbint parses,
// so every bench includes hsc first. The other benches cite this file for the explanation.
#include "_corpus.hpp"

#include "_bench_common.hpp"

static u8 g_out[1 << 20];

static void
print_aux(u64 in, usize comp, f64 cyc_per_op) noexcept
{
  mb::fmt2 cycb = mb::to_fmt2(in ? cyc_per_op / (f64)in : 0.0);
  micron::io::println("            comp=", comp, "B  cyc/byte(in)=", cycb.whole, ".", (u32)(cycb.frac_x100 / 10),
                      (u32)(cycb.frac_x100 % 10));
}

int
main()
{
  mb::pin_cpu0();
  mb::print_preamble("hsc hopf — encode throughput (cyc/op, cyc/byte of input)");
  hc::generate();
  mb::print_header();

  hsc::hopf_scratch sc;

  for ( usize i = 0; i < hc::corpora_count; ++i ) {
    const hc::corpus &c = hc::corpora[i];

    struct cfg {
      const char *name;
      u32 dl;
      i32 lvl;
      bool tf;
    };

    const cfg cfgs[] = { { "bin d8 L3", 3, 3 }, { "bin d8 L6", 3, 6 },           { "bin d16 L6", 4, 6 },
                         { "bin d4 L6", 2, 6 }, { "bin d8 L6 rot", 3, 6, true }, { "bin d16 L6 rot", 4, 6, true } };
    for ( const cfg &f : cfgs ) {
      const hsc::hopf_opts o{ .level = f.lvl, .dim_log2 = f.dl, .transform = f.tf };
      (void)hsc::hopf_into(hsc::bytes{ c.d, c.n }, o, g_out, sizeof(g_out), sc);
      usize comp = 0;
      mb::row r = mb::bench_one(
          c.name, f.name, c.n, c.n,
          [&]() {
            comp = hsc::hopf_into(hsc::bytes{ c.d, c.n }, o, g_out, sizeof(g_out), sc);
            mb::sink_size(comp);
            mb::clobber(g_out);
          },
          c.mr);
      mb::print_row(r);
      print_aux(c.n, comp, r.cyc_per_op);
    }
  }

  {
    const usize nf = hc::k_n / 4;
    for ( bool tf : { false, true } ) {
      const hsc::hopf_opts ov{ .m = hsc::mode::vec, .level = 6, .dim_log2 = 3, .transform = tf };
      (void)hsc::hopf_into(hsc::floats{ hc::g_gauss, nf }, ov, g_out, sizeof(g_out), sc);
      mb::row rv = mb::bench_one(
          "gauss16k", tf ? "vec d8 L6 rot" : "vec d8 L6", nf * 4, nf * 4,
          [&]() {
            const max_t w = hsc::hopf_into(hsc::floats{ hc::g_gauss, nf }, ov, g_out, sizeof(g_out), sc);
            mb::sink_size(static_cast<usize>(w));
            mb::clobber(g_out);
          },
          1 << 8);
      mb::print_row(rv);
    }

    for ( bool tf : { false, true } ) {
      const hsc::hopf_opts ou{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 3, .transform = tf };
      (void)hsc::hopf_into(hsc::floats{ hc::g_unit, nf }, ou, g_out, sizeof(g_out), sc);
      mb::row ru = mb::bench_one(
          "unit16k", tf ? "unit d8 L6 rot" : "unit d8 L6", nf * 4, nf * 4,
          [&]() {
            const max_t w = hsc::hopf_into(hsc::floats{ hc::g_unit, nf }, ou, g_out, sizeof(g_out), sc);
            mb::sink_size(static_cast<usize>(w));
            mb::clobber(g_out);
          },
          1 << 8);
      mb::print_row(ru);
    }

    const hsc::hopf_opts oq{ .m = hsc::mode::quotient, .level = 6 };
    (void)hsc::hopf_into(hsc::floats{ hc::g_unit, nf }, oq, g_out, sizeof(g_out), sc);
    mb::row rq = mb::bench_one(
        "unit16k", "quot L6", nf * 4, nf * 4,
        [&]() {
          const max_t w = hsc::hopf_into(hsc::floats{ hc::g_unit, nf }, oq, g_out, sizeof(g_out), sc);
          mb::sink_size(static_cast<usize>(w));
          mb::clobber(g_out);
        },
        1 << 8);
    mb::print_row(rq);

    const hsc::hopf_opts oq4{ .m = hsc::mode::quat, .level = 6 };
    (void)hsc::hopf_into(hsc::floats{ hc::g_fiber8, nf }, oq4, g_out, sizeof(g_out), sc);
    mb::row rq4 = mb::bench_one(
        "fiber16k", "quat L6", nf * 4, nf * 4,
        [&]() {
          const max_t w = hsc::hopf_into(hsc::floats{ hc::g_fiber8, nf }, oq4, g_out, sizeof(g_out), sc);
          mb::sink_size(static_cast<usize>(w));
          mb::clobber(g_out);
        },
        1 << 8);
    mb::print_row(rq4);

    const hsc::hopf_opts oq8{ .m = hsc::mode::oct, .level = 6 };
    (void)hsc::hopf_into(hsc::floats{ hc::g_fiber16, nf }, oq8, g_out, sizeof(g_out), sc);
    mb::row rq8 = mb::bench_one(
        "fiber16k", "oct L6", nf * 4, nf * 4,
        [&]() {
          const max_t w = hsc::hopf_into(hsc::floats{ hc::g_fiber16, nf }, oq8, g_out, sizeof(g_out), sc);
          mb::sink_size(static_cast<usize>(w));
          mb::clobber(g_out);
        },
        1 << 8);
    mb::print_row(rq8);
  }

  mb::print_epilogue();
  return 0;
}
