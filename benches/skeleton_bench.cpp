//  hsc skeleton construction — the cost the scratch cache amortizes: tree_build + pack tables
//  per (dim, level), forced cold by alternating two keys.
//    duck build benches/skeleton_bench.cpp --perf --fp -i ../micron -i ../micron/src && ./bin/skeleton_bench
#include "_corpus.hpp"

#include "_bench_common.hpp"

int
main()
{
  mb::pin_cpu0();
  mb::print_preamble("hsc skeleton build — cyc/op (op = one cold build_tree incl. pack tables)");
  mb::print_header();

  hsc::hopf_scratch sc;

  struct cfg {
    const char *name;
    u32 dl;
    i32 lvl;
  };

  const cfg cfgs[]
     = { { "dim4 L6", 2, 6 }, { "dim4 L9", 2, 9 }, { "dim8 L6", 3, 6 }, { "dim16 L6", 4, 6 }, { "dim32 L7", 5, 7 }, { "dim64 L7", 6, 7 } };
  for ( const cfg &f : cfgs ) {
    bool flip = false;
    mb::row r = mb::bench_one(
       f.name, "build cold", 1, 1,
       [&]() {
         //  alternate the level with an adjacent one so every call rebuilds
         const i32 lvl = flip ? (f.lvl > 1 ? f.lvl - 1 : f.lvl + 1) : f.lvl;
         flip = !flip;
         (void)sc.build_tree(f.dl, hsc::level_dq(lvl));
         mb::sink_size(sc.nodes.size());
       },
       1 << 6);
    mb::print_row(r);
  }

  {
    bool flip = false;
    mb::row r = mb::bench_one(
       "s2 L7", "build cold", 1, 1,
       [&]() {
         (void)sc.build_s2(hsc::level_dq(flip ? 6 : 7));
         flip = !flip;
         mb::sink_size(sc.bands.size());
       },
       1 << 8);
    mb::print_row(r);
  }

  //  the suspensions share the ONE tree slot, so these rows also measure the documented
  //  slot-thrash a mixed vec/quat/oct workload pays, plus the per-band memo linear scan
  {
    struct scfg {
      const char *name;
      u32 child;
      i32 lvl;
    };

    const scfg scfgs[] = { { "s4 L6 (quat)", 2, 6 }, { "s4 L7 (quat)", 2, 7 }, { "s8 L6 (oct)", 3, 6 } };
    for ( const scfg &f : scfgs ) {
      bool flip = false;
      mb::row r = mb::bench_one(
         f.name, "build cold", 1, 1,
         [&]() {
           (void)sc.build_susp(f.child, hsc::level_dq(flip ? f.lvl - 1 : f.lvl));
           flip = !flip;
           mb::sink_size(sc.sbands.size());
         },
         1 << 6);
      mb::print_row(r);
    }
  }

  mb::print_epilogue();
  return 0;
}
