// Rate-distortion table: bits/byte and RMSE per (mode, dim, level) cell on the synthetic corpora,
// including the quat/oct fiber-saving columns. hsc before _bench_common.hpp (see hopf_bench.cpp).
#include "_corpus.hpp"

#include "_bench_common.hpp"

static u8 g_z[1 << 21];
static u8 g_out[1 << 20];
static f32 g_fout[1 << 18];

static void
print_f3(const char *tag, f64 v)
{
  const bool neg = v < 0;
  if ( neg ) v = -v;
  const u64 w = static_cast<u64>(v);
  const u64 f = static_cast<u64>((v - static_cast<f64>(w)) * 1000.0 + 0.5);
  micron::io::print(tag, neg ? "-" : "", w, ".", static_cast<u32>(f / 100), static_cast<u32>((f / 10) % 10), static_cast<u32>(f % 10));
}

int
main()
{
  mb::pin_cpu0();
  mb::print_preamble("hsc rate-distortion (bin on smooth bytes; unit/quotient on unit blocks)");
  hc::generate();
  hsc::hopf_scratch sc;

  micron::io::println("");
  micron::io::println("mode bin, smooth64k: bits/byte vs per-byte RMSE");
  for ( u32 dl : { 2u, 3u, 4u } ) {
    for ( i32 lvl : { 2, 4, 6, 8 } ) {
      const hsc::hopf_opts o{ .level = lvl, .dim_log2 = dl };
      const usize zn = hsc::hopf_into(hsc::bytes{ hc::g_smooth, hc::k_n }, o, g_z, sizeof(g_z), sc);
      auto r = hsc::unhopf(hsc::bytes{ g_z, zn }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
      f64 se = 0;
      for ( usize i = 0; i < hc::k_n; ++i ) {
        const f64 e = static_cast<f64>(g_out[i]) - static_cast<f64>(hc::g_smooth[i]);
        se += e * e;
      }
      micron::io::print("  dim=", 1u << dl, " L", static_cast<u32>(lvl));
      print_f3("  bits/byte=", static_cast<f64>(zn - 48) * 8.0 / static_cast<f64>(hc::k_n));
      print_f3("  rmse=", __builtin_sqrt(se / static_cast<f64>(hc::k_n)));
      micron::io::println("  ok=", r.is_first() ? 1 : 0);
    }
  }

  micron::io::println("");
  micron::io::println("mode unit vs quotient on unit 4-blocks: bits/block at the same d");
  for ( i32 lvl : { 3, 5, 6, 7 } ) {
    const usize nf = hc::k_n / 4;
    const hsc::hopf_opts ou{ .m = hsc::mode::unit, .level = lvl, .dim_log2 = 2 };
    const hsc::hopf_opts oq{ .m = hsc::mode::quotient, .level = lvl };
    const max_t zu = hsc::hopf_into(hsc::floats{ hc::g_unit, nf }, ou, g_z, sizeof(g_z), sc);
    const max_t zq = hsc::hopf_into(hsc::floats{ hc::g_unit, nf }, oq, g_z, sizeof(g_z), sc);
    const f64 blocks = static_cast<f64>(nf) / 4.0;
    micron::io::print("  L", static_cast<u32>(lvl));
    print_f3("  unit bits/blk=", static_cast<f64>(zu - 48) * 8.0 / blocks);
    print_f3("  quot bits/blk=", static_cast<f64>(zq - 48) * 8.0 / blocks);
    print_f3("  fiber saving=", (static_cast<f64>(zu) - static_cast<f64>(zq)) * 8.0 / blocks);
    micron::io::println("");
  }

  micron::io::println("");
  micron::io::println("mode unit vs quat/oct on fiber-symmetric pairs: bits/block at the same d");
  for ( i32 lvl : { 3, 5, 6, 7 } ) {
    const usize nf = hc::k_n / 4;
    const hsc::hopf_opts ou8{ .m = hsc::mode::unit, .level = lvl, .dim_log2 = 3 };
    const hsc::hopf_opts oq{ .m = hsc::mode::quat, .level = lvl };
    const max_t zu8 = hsc::hopf_into(hsc::floats{ hc::g_fiber8, nf }, ou8, g_z, sizeof(g_z), sc);
    const max_t zq = hsc::hopf_into(hsc::floats{ hc::g_fiber8, nf }, oq, g_z, sizeof(g_z), sc);
    const f64 blk8 = static_cast<f64>(nf) / 8.0;
    const hsc::hopf_opts ou16{ .m = hsc::mode::unit, .level = lvl, .dim_log2 = 4 };
    const hsc::hopf_opts oo{ .m = hsc::mode::oct, .level = lvl };
    const max_t zu16 = hsc::hopf_into(hsc::floats{ hc::g_fiber16, nf }, ou16, g_z, sizeof(g_z), sc);
    const max_t zo = hsc::hopf_into(hsc::floats{ hc::g_fiber16, nf }, oo, g_z, sizeof(g_z), sc);
    const f64 blk16 = static_cast<f64>(nf) / 16.0;
    micron::io::print("  L", static_cast<u32>(lvl));
    print_f3("  unit8 bits/blk=", static_cast<f64>(zu8 - 48) * 8.0 / blk8);
    print_f3("  quat=", static_cast<f64>(zq - 48) * 8.0 / blk8);
    print_f3("  saved=", (static_cast<f64>(zu8) - static_cast<f64>(zq)) * 8.0 / blk8);
    print_f3("  | unit16 bits/blk=", static_cast<f64>(zu16 - 48) * 8.0 / blk16);
    print_f3("  oct=", static_cast<f64>(zo - 48) * 8.0 / blk16);
    print_f3("  saved=", (static_cast<f64>(zu16) - static_cast<f64>(zo)) * 8.0 / blk16);
    micron::io::println("");
  }

  micron::io::println("");
  micron::io::println("mode vec, gauss16k dim scaling at L6: bits/sample vs RMSE");
  for ( u32 dl : { 2u, 3u, 4u, 5u } ) {
    const usize nf = hc::k_n / 4;
    const hsc::hopf_opts o{ .m = hsc::mode::vec, .level = 6, .dim_log2 = dl, .gain_bits = 8 };
    const max_t zn = hsc::hopf_into(hsc::floats{ hc::g_gauss, nf }, o, g_z, sizeof(g_z), sc);
    auto r = hsc::unhopf(hsc::bytes{ g_z, static_cast<usize>(zn) }, hsc::wfloats{ g_fout, sizeof(g_fout) / 4 });
    f64 se = 0, sg = 0;
    for ( usize i = 0; i < nf; ++i ) {
      const f64 e = static_cast<f64>(g_fout[i]) - static_cast<f64>(hc::g_gauss[i]);
      se += e * e;
      sg += static_cast<f64>(hc::g_gauss[i]) * static_cast<f64>(hc::g_gauss[i]);
    }
    micron::io::print("  dim=", 1u << dl);
    print_f3("  bits/sample=", static_cast<f64>(zn - 48) * 8.0 / static_cast<f64>(nf));
    print_f3("  rmse/rms=", __builtin_sqrt(se / sg));
    micron::io::println("  ok=", r.is_first() ? 1 : 0);
  }

  micron::io::println("");
  micron::io::println("transform on/off pairs per corpus (bin): per-byte RMSE at equal rate");
  {
    struct bcorp {
      const char *name;
      const u8 *d;
    };

    const bcorp bcs[] = { { "noise64k ", hc::g_noise }, { "smooth64k", hc::g_smooth }, { "spiky64k ", hc::g_spiky } };
    for ( const bcorp &c : bcs ) {
      for ( u32 dl : { 3u, 4u } ) {
        for ( i32 lvl : { 5, 6, 7 } ) {
          f64 rm[2];
          usize zs = 0;
          for ( u32 tf = 0; tf < 2; ++tf ) {
            const hsc::hopf_opts o{ .level = lvl, .dim_log2 = dl, .transform = tf == 1 };
            zs = hsc::hopf_into(hsc::bytes{ c.d, hc::k_n }, o, g_z, sizeof(g_z), sc);
            auto r = hsc::unhopf(hsc::bytes{ g_z, zs }, hsc::wbytes{ g_out, sizeof(g_out) }, sc);
            f64 se = 0;
            for ( usize i = 0; i < hc::k_n; ++i ) {
              const f64 e = static_cast<f64>(g_out[i]) - static_cast<f64>(c.d[i]);
              se += e * e;
            }
            rm[tf] = r.is_first() ? __builtin_sqrt(se / static_cast<f64>(hc::k_n)) : -1.0;
          }
          micron::io::print("  ", c.name, " dim=", 1u << dl, " L", static_cast<u32>(lvl));
          print_f3("  bits/byte=", static_cast<f64>(zs - 48) * 8.0 / static_cast<f64>(hc::k_n));
          print_f3("  rmse off=", rm[0]);
          print_f3("  on=", rm[1]);
          micron::io::println("");
        }
      }
    }
  }

  micron::io::println("");
  micron::io::println("transform on/off, onehot16k (near-one-hot unit 8-blocks): rmse/rms");
  for ( u32 mi : { 0u, 1u } ) {
    for ( i32 lvl : { 5, 6, 7 } ) {
      const usize nf = hc::k_n / 4;
      f64 rm[2];
      for ( u32 tf = 0; tf < 2; ++tf ) {
        const hsc::hopf_opts o{ .m = mi == 0 ? hsc::mode::vec : hsc::mode::unit, .level = lvl, .dim_log2 = 3, .transform = tf == 1 };
        const max_t zn = hsc::hopf_into(hsc::floats{ hc::g_onehot, nf }, o, g_z, sizeof(g_z), sc);
        auto r = hsc::unhopf(hsc::bytes{ g_z, static_cast<usize>(zn) }, hsc::wfloats{ g_fout, sizeof(g_fout) / 4 });
        f64 se = 0, sg = 0;
        for ( usize i = 0; i < nf; ++i ) {
          const f64 e = static_cast<f64>(g_fout[i]) - static_cast<f64>(hc::g_onehot[i]);
          se += e * e;
          sg += static_cast<f64>(hc::g_onehot[i]) * static_cast<f64>(hc::g_onehot[i]);
        }
        rm[tf] = r.is_first() ? __builtin_sqrt(se / sg) : -1.0;
      }
      micron::io::print("  ", mi == 0 ? "vec " : "unit", " d8 L", static_cast<u32>(lvl));
      print_f3("  rmse/rms off=", rm[0]);
      print_f3("  on=", rm[1]);
      micron::io::println("");
    }
  }

  micron::io::println("");
  return 0;
}
