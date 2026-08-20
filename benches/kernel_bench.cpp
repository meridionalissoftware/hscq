// hsc kernels — cycles per call for every hot leaf of the codec, isolated. This is the A/B
// harness: an optimization is accepted or rejected on a row of this table, and IPC + bmiss%
// are the microarchitectural scoreboard next to it.
//   duck build benches/kernel_bench.cpp --perf --fp -i ../micron -i ../micron/src && ./bin/kernel_bench
// NOTE the include order: hsc before bbench (__bitwise macro, see hopf_bench.cpp).
#include "_corpus.hpp"

#include "_bench_common.hpp"

static constexpr usize k_blocks = 1024;
static f64 g_blk[k_blocks][64];      // f64 unit-ish blocks, the analog input the sphere layer sees
static f32 g_f32[k_blocks][64];      // the same, f32, for the typed block layer
static f64 g_s2p[k_blocks][4];       // points on S^2 for the quotient lane
static u8 g_scr[1 << 16];

static const char *
dim_name(u32 dl) noexcept
{
  return dl == 2 ? "dim4" : (dl == 3 ? "dim8" : (dl == 4 ? "dim16" : (dl == 5 ? "dim32" : "dim64")));
}

int
main()
{
  mb::pin_cpu0();
  mb::print_preamble("hsc kernels — cycles per call, isolated (op = one call)");
  hc::generate();
  mb::spin_up();
  mb::print_header();

  hsc::hopf_scratch sc;

  for ( usize b = 0; b < k_blocks; ++b ) {
    f64 s = 0;
    for ( usize c = 0; c < 64; ++c ) {
      g_blk[b][c] = static_cast<f64>(hc::xs() >> 11) / 9007199254740992.0 - 0.5;
      s += g_blk[b][c] * g_blk[b][c];
    }
    const f64 nn = __builtin_sqrt(s);
    for ( usize c = 0; c < 64; ++c ) {
      g_blk[b][c] /= nn;
      g_f32[b][c] = static_cast<f32>(g_blk[b][c]);
    }
    f64 p[3] = { g_blk[b][0], g_blk[b][1], g_blk[b][2] };
    const f64 pn = __builtin_sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    g_s2p[b][0] = p[0] / pn;
    g_s2p[b][1] = p[1] / pn;
    g_s2p[b][2] = p[2] / pn;
    g_s2p[b][3] = 0;
  }

  const hsc::gain_quant gq{ .bits = 8, .scale = 1.0f };

  {
    (void)sc.build_tree(2, hsc::level_dq(6));
    const hsc::s3_skeleton s3 = hsc::__s3_of(sc.sk, sc.sk.nodes[sc.sk.root]);
    usize bi = 0;

    mb::print_row(
        mb::bench_one("s3", "s3_quantize", 1, 1, [&]() { mb::sink_size(hsc::s3_quantize(s3, g_blk[bi++ & (k_blocks - 1)])); }, 1 << 14));
    mb::print_row(mb::bench_one(
        "s3", "s3_quantize refine", 1, 1, [&]() { mb::sink_size(hsc::s3_quantize(s3, g_blk[bi++ & (k_blocks - 1)], 1)); }, 1 << 14));

    u64 codes[64];
    for ( usize i = 0; i < 64; ++i ) codes[i] = hsc::s3_quantize(s3, g_blk[i]);
    usize ci = 0;
    mb::print_row(mb::bench_one(
        "s3", "s3_decode", 1, 1,
        [&]() {
          f64 o[4];
          hsc::s3_decode(s3, codes[ci++ & 63], o);
          mb::clobber(o);
        },
        1 << 14));
  }

  {
    (void)sc.build_s2(hsc::level_dq(6));
    usize bi = 0;
    mb::print_row(
        mb::bench_one("s2", "s2_quantize", 1, 1, [&]() { mb::sink_size(hsc::s2_quantize(sc.s2, g_s2p[bi++ & (k_blocks - 1)])); }, 1 << 14));
    u64 codes[64];
    for ( usize i = 0; i < 64; ++i ) codes[i] = hsc::s2_quantize(sc.s2, g_s2p[i]);
    usize ci = 0;
    mb::print_row(mb::bench_one(
        "s2", "s2_decode", 1, 1,
        [&]() {
          f64 o[3];
          hsc::s2_decode(sc.s2, codes[ci++ & 63], o);
          mb::clobber(o);
        },
        1 << 14));
    usize qi = 0;
    mb::print_row(mb::bench_one(
        "quot", "hopf_project", 1, 1,
        [&]() {
          f64 p[3];
          mb::sink_size(static_cast<usize>(hsc::hopf_project(g_f32[qi++ & (k_blocks - 1)], p)));
          mb::clobber(p);
        },
        1 << 14));
    mb::print_row(mb::bench_one(
        "quot", "hopf_lift", 1, 1,
        [&]() {
          f32 z[4];
          hsc::hopf_lift(g_s2p[qi++ & (k_blocks - 1)], z);
          mb::clobber(z);
        },
        1 << 14));
  }

  // the S^4 suspension + quat lane (one scratch slot: run all S^4 rows before building S^8)
  {
    static f64 s4p[k_blocks][5];
    for ( usize b = 0; b < k_blocks; ++b ) (void)hsc::quat_project(g_f32[b], s4p[b]);
    (void)sc.build_susp(2, hsc::level_dq(6));
    usize bi = 0;
    hsc::tree_fields tf{};
    mb::print_row(mb::bench_one(
        "s4", "susp_quantize", 1, 1, [&]() { mb::sink_size(hsc::susp_quantize(sc.ss, sc.sk, s4p[bi++ & (k_blocks - 1)], tf)); }, 1 << 14));
    u32 bands[64];
    hsc::tree_fields fs[64];
    for ( usize i = 0; i < 64; ++i ) bands[i] = hsc::susp_quantize(sc.ss, sc.sk, s4p[i], fs[i]);
    usize ci = 0;
    mb::print_row(mb::bench_one(
        "s4", "susp_decode", 1, 1,
        [&]() {
          f64 o[5];
          hsc::susp_decode(sc.ss, sc.sk, bands[ci & 63], fs[ci & 63], o);
          ++ci;
          mb::clobber(o);
        },
        1 << 14));
    usize qi = 0;
    mb::print_row(mb::bench_one(
        "quat", "quat_project", 1, 1,
        [&]() {
          f64 p[5];
          mb::sink_size(static_cast<usize>(hsc::quat_project(g_f32[qi++ & (k_blocks - 1)], p)));
          mb::clobber(p);
        },
        1 << 14));
    mb::print_row(mb::bench_one(
        "quat", "quat_lift", 1, 1,
        [&]() {
          f32 z[8];
          hsc::quat_lift(s4p[qi++ & (k_blocks - 1)], z);
          mb::clobber(z);
        },
        1 << 14));
  }

  // the S^8 suspension + oct lane
  {
    static f64 s8p[k_blocks][9];
    for ( usize b = 0; b < k_blocks; ++b ) (void)hsc::oct_project(g_f32[b], s8p[b]);
    (void)sc.build_susp(3, hsc::level_dq(6));
    usize bi = 0;
    hsc::tree_fields tf{};
    mb::print_row(mb::bench_one(
        "s8", "susp_quantize", 1, 1, [&]() { mb::sink_size(hsc::susp_quantize(sc.ss, sc.sk, s8p[bi++ & (k_blocks - 1)], tf)); }, 1 << 14));
    u32 bands[64];
    hsc::tree_fields fs[64];
    for ( usize i = 0; i < 64; ++i ) bands[i] = hsc::susp_quantize(sc.ss, sc.sk, s8p[i], fs[i]);
    usize ci = 0;
    mb::print_row(mb::bench_one(
        "s8", "susp_decode", 1, 1,
        [&]() {
          f64 o[9];
          hsc::susp_decode(sc.ss, sc.sk, bands[ci & 63], fs[ci & 63], o);
          ++ci;
          mb::clobber(o);
        },
        1 << 14));
    usize qi = 0;
    mb::print_row(mb::bench_one(
        "oct", "oct_project", 1, 1,
        [&]() {
          f64 p[9];
          mb::sink_size(static_cast<usize>(hsc::oct_project(g_f32[qi++ & (k_blocks - 1)], p)));
          mb::clobber(p);
        },
        1 << 14));
    mb::print_row(mb::bench_one(
        "oct", "oct_lift", 1, 1,
        [&]() {
          f32 z[16];
          hsc::oct_lift(s8p[qi++ & (k_blocks - 1)], z);
          mb::clobber(z);
        },
        1 << 14));
  }

  for ( u32 dl = 2; dl <= 6; ++dl ) {
    const u32 n = 1u << dl;
    usize bi = 0;
    mb::print_row(mb::bench_one(
        dim_name(dl), "rot_fwd", 1, 1,
        [&]() {
          f64 v[64];
          const f64 *src = g_blk[bi++ & (k_blocks - 1)];
          for ( u32 c = 0; c < n; ++c ) v[c] = src[c];
          hsc::rot_fwd(v, dl);
          mb::clobber(v);
        },
        1 << 14));
    mb::print_row(mb::bench_one(
        dim_name(dl), "rot_inv", 1, 1,
        [&]() {
          f64 v[64];
          const f64 *src = g_blk[bi++ & (k_blocks - 1)];
          for ( u32 c = 0; c < n; ++c ) v[c] = src[c];
          hsc::rot_inv(v, dl);
          mb::clobber(v);
        },
        1 << 14));
  }

  for ( u32 dl = 2; dl <= 6; ++dl ) {
    (void)sc.build_tree(dl, hsc::level_dq(6));
    const char *nm = dim_name(dl);
    usize bi = 0;

    mb::print_row(mb::bench_one(
        nm, "quantize_block f32", 1, 1,
        [&]() {
          hsc::block_code bc{};
          mb::sink_size(static_cast<usize>(hsc::quantize_block(sc.sk, gq, g_f32[bi++ & (k_blocks - 1)], bc)));
        },
        1 << 14));

    hsc::block_code bc0{};
    (void)hsc::quantize_block(sc.sk, gq, g_f32[0], bc0);
    mb::print_row(mb::bench_one(
        nm, "reconstruct_block", 1, 1,
        [&]() {
          f32 x[64];
          hsc::reconstruct_block(sc.sk, gq, bc0, x);
          mb::clobber(x);
        },
        1 << 14));

    mb::print_row(mb::bench_one(
        nm, "quantize_unit", 1, 1,
        [&]() {
          hsc::tree_fields f{};
          mb::sink_size(static_cast<usize>(hsc::quantize_unit(sc.sk, g_f32[bi++ & (k_blocks - 1)], f)));
        },
        1 << 14));

    hsc::vq_index a(0u);
    hsc::pack_rank(sc.sk, sc.pt, bc0.shape, a);
    const u32 nbits = hsc::shape_bits(sc.sk, sc.pt);
    mb::print_row(mb::bench_one(
        nm, "put_wide", 1, 1,
        [&]() {
          hsc::bits::bitwriter w{ .acc = 0, .cnt = 0, .out = g_scr, .fast_end = g_scr + sizeof(g_scr) - 8 };
          hsc::put_wide(w, a, nbits);
          w.finish();
          mb::clobber(g_scr);
        },
        1 << 14));
    {
      hsc::bits::bitwriter w{ .acc = 0, .cnt = 0, .out = g_scr, .fast_end = g_scr + sizeof(g_scr) - 8 };
      hsc::put_wide(w, a, nbits);
      w.finish();
    }
    mb::print_row(mb::bench_one(
        nm, "get_wide", 1, 1,
        [&]() {
          hsc::bits::bitreader r{ .p = g_scr, .end = g_scr + sizeof(g_scr) };
          hsc::vq_index v(0u);
          mb::sink_bool(hsc::get_wide(r, nbits, v));
        },
        1 << 14));
  }

  {
    usize bi = 0;
    mb::print_row(mb::bench_one(
        "gain", "gq_encode", 1, 1, [&]() { mb::sink_size(hsc::gq_encode(gq, 0.25 + 0.5 * (f64)(bi++ & 255) / 255.0)); }, 1 << 16));
    mb::print_row(
        mb::bench_one("gain", "gq_decode", 1, 1, [&]() { mb::sink_size((usize)(hsc::gq_decode(gq, (u32)(bi++ & 255)) * 8.0)); }, 1 << 16));
  }

  mb::print_epilogue();
  return 0;
}
