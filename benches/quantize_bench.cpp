//  hsc quantizer kernels — cycles per block for the raw typed layer: s3 against tree depth,
//  refine on/off, and the arbint rank/unrank isolated from the geometry.
//    duck build benches/quantize_bench.cpp --perf --fp -i ../micron -i ../micron/src && ./bin/quantize_bench
//  NOTE the include order: hsc before bbench (__bitwise macro, see hopf_bench.cpp).
#include "_corpus.hpp"

#include "_bench_common.hpp"

int
main()
{
  mb::pin_cpu0();
  mb::print_preamble("hsc quantize — cycles per block (op = one block)");
  hc::generate();
  mb::print_header();

  hsc::hopf_scratch sc;
  static f64 blocks[1024][64];
  for ( usize b = 0; b < 1024; ++b )
    for ( usize c = 0; c < 64; ++c ) blocks[b][c] = static_cast<f64>(hc::xs() >> 11) / 9007199254740992.0 - 0.5;

  for ( u32 dl = 2; dl <= 6; ++dl ) {
    (void)sc.build_tree(dl, hsc::level_dq(6));
    const char *nm = dl == 2 ? "dim4" : (dl == 3 ? "dim8" : (dl == 4 ? "dim16" : (dl == 5 ? "dim32" : "dim64")));
    usize bi = 0;

    mb::row rq = mb::bench_one(
       nm, "quantize L6", 1, 1,
       [&]() {
         hsc::tree_fields f{};
         hsc::tree_quantize(sc.sk, blocks[bi++ & 1023], f);
         mb::sink_size(f.leaf[0] + static_cast<usize>(f.base[0]));
       },
       1 << 14);
    mb::print_row(rq);

    mb::row rr = mb::bench_one(
       nm, "quantize refine", 1, 1,
       [&]() {
         hsc::tree_fields f{};
         hsc::tree_quantize(sc.sk, blocks[bi++ & 1023], f, 1);
         mb::sink_size(f.leaf[0] + static_cast<usize>(f.base[0]));
       },
       1 << 14);
    mb::print_row(rr);

    hsc::tree_fields f0{};
    hsc::tree_quantize(sc.sk, blocks[0], f0);
    mb::row rk = mb::bench_one(
       nm, "rank+unrank", 1, 1,
       [&]() {
         hsc::vq_index a(0u);
         hsc::pack_rank(sc.sk, sc.pt, f0, a);
         hsc::tree_fields f{};
         (void)hsc::pack_unrank(sc.sk, sc.pt, a, f);
         mb::sink_size(static_cast<usize>(f.base[0]));
       },
       1 << 14);
    mb::print_row(rk);

    mb::row rd = mb::bench_one(
       nm, "decode", 1, 1,
       [&]() {
         f64 p[64];
         hsc::tree_decode(sc.sk, f0, p);
         mb::sink_size(static_cast<usize>(p[0] * 0 + 1));
         mb::clobber(p);
       },
       1 << 14);
    mb::print_row(rd);
  }

  mb::print_epilogue();
  return 0;
}
