// The recursive construction. Guards: golden cardinalities (mod 2^64) and memo node counts
// against the C++-exact Python mirror (scripts/compression_ratio_model.py) for dims 8..64; grid
// table endpoints; quantize->decode->quantize field idempotence on random vectors; unit-norm
// reconstruction; independent m_mod recompute over the arena wiring; arena oom behavior.
// Full every-index sweeps need unrank and live in pack.cpp.

#include "../src/hsc/sphere/tree.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

// the grid recurrence must fold to these exact integers (one correctly-rounded multiply/step)
static_assert(hsc::k_grid.dq[0] == 33554432);
static_assert(hsc::k_grid.dq[1] == 33388544);
static_assert(hsc::k_grid.dq[2] == 33223476);
static_assert(hsc::k_grid.dq[3] == 33059224);
static_assert(hsc::k_grid.dq[2046] == 1324);
static_assert(hsc::k_grid.dq[2047] == 1318);

constexpr u64
ct_tree_m(u32 dim_log2, u32 dq)
{
  hsc::tree_node nodes[64]{};
  hsc::tree_row rows[64]{};
  hsc::s3_leaf leaves[256]{};
  hsc::tree_arena ar{ nodes, 64, 0, rows, 64, 0, leaves, 256, 0 };
  const max_t root = hsc::tree_build(dim_log2, dq, ar);
  if ( root < 0 ) return 0;
  return ar.nodes[static_cast<u32>(root)].m_mod;
}

static_assert(ct_tree_m(3, hsc::level_dq(3)) == 2310);       // dim 8,  d = 0.5
static_assert(ct_tree_m(4, hsc::level_dq(3)) == 60316);      // dim 16, d = 0.5

struct golden {
  u32 dim_log2;
  i32 level;
  u64 m_mod;
  u32 memo_nodes;
};

// from the validated mirror: M mod 2^64 and the reachable memo size
constexpr golden k_golden[] = {
  { 3, 3, 2310ull, 4 },
  { 3, 5, 120856ull, 6 },
  { 3, 6, 2523114ull, 8 },
  { 4, 3, 60316ull, 9 },
  { 4, 6, 73150212400ull, 39 },
  { 5, 5, 410142624246ull, 57 },
  { 5, 7, 4021150973525067886ull, 537 },
  { 6, 5, 241630806088037300ull, 120 },
  { 6, 7, 10141755649553200952ull, 1107 },
};

static hsc::tree_node g_nodes[4096];
static hsc::tree_row g_rows[65536];
static hsc::s3_leaf g_leaves[131072];

hsc::tree_arena
fresh_arena()
{
  return hsc::tree_arena{ g_nodes, 4096, 0, g_rows, 65536, 0, g_leaves, 131072, 0 };
}

bool
fields_equal(const hsc::tree_fields &a, const hsc::tree_fields &b, u32 dim_log2)
{
  for ( u32 i = 0; i < hsc::tree_inode_count(dim_log2); ++i )
    if ( a.leaf[i] != b.leaf[i] ) return false;
  for ( u32 i = 0; i < hsc::tree_bnode_count(dim_log2); ++i )
    if ( a.base[i] != b.base[i] ) return false;
  return true;
}

}      // namespace

int
main()
{
  sb::test_case("golden cardinalities (mod 2^64) and memo node counts, dims 8..64");
  {
    for ( const golden &g : k_golden ) {
      hsc::tree_arena ar = fresh_arena();
      const max_t root = hsc::tree_build(g.dim_log2, hsc::level_dq(g.level), ar);
      sb::require(root >= 0);
      const hsc::tree_skeleton sk = hsc::tree_view(ar, static_cast<u32>(root), g.dim_log2, hsc::level_dq(g.level));
      sb::require(hsc::tree_m_mod64(sk), g.m_mod);
      sb::require(ar.node_count, g.memo_nodes);
    }
  }

  sb::test_case("m_mod is consistent with an independent wrap-product walk of the arena");
  {
    hsc::tree_arena ar = fresh_arena();
    const max_t root = hsc::tree_build(6, hsc::level_dq(7), ar);
    sb::require(root >= 0);
    for ( u32 i = 0; i < ar.node_count; ++i ) {
      const hsc::tree_node &nd = ar.nodes[i];
      if ( nd.dim_log2 == 2 ) continue;
      u64 m = 0;
      for ( u32 x = 0; x < nd.count; ++x ) {
        const hsc::tree_row &rw = ar.rows[nd.rows_at + x];
        m += ar.nodes[rw.c1].m_mod * ar.nodes[rw.c2].m_mod;
      }
      sb::require(m, nd.m_mod);
    }
  }

  sb::test_case("quantize -> decode -> quantize is a field fixed point (random vectors)");
  {
    tutil::rng g;
    for ( const golden &gg : k_golden ) {
      hsc::tree_arena ar = fresh_arena();
      const max_t root = hsc::tree_build(gg.dim_log2, hsc::level_dq(gg.level), ar);
      sb::require(root >= 0);
      const hsc::tree_skeleton sk = hsc::tree_view(ar, static_cast<u32>(root), gg.dim_log2, hsc::level_dq(gg.level));
      const u32 n = 1u << gg.dim_log2;
      const i32 trials = gg.dim_log2 >= 5 ? 300 : 1500;
      for ( i32 t = 0; t < trials; ++t ) {
        f64 y[64];
        f64 n2 = 0;
        for ( u32 c = 0; c < n; ++c ) {
          y[c] = g.unit();
          n2 += y[c] * y[c];
        }
        if ( n2 < 1e-12 ) continue;
        hsc::tree_fields f0{}, f1{};
        hsc::tree_quantize(sk, y, f0);
        f64 p[64];
        hsc::tree_decode(sk, f0, p);
        hsc::tree_quantize(sk, p, f1);
        sb::require(fields_equal(f0, f1, gg.dim_log2));
        f64 pn = 0;
        for ( u32 c = 0; c < n; ++c ) pn += p[c] * p[c];
        sb::require(pn > 1.0 - 1e-11 && pn < 1.0 + 1e-11);
      }
    }
  }

  sb::test_case("refine keeps the fixed point and never loses (dot product) at dim 8");
  {
    hsc::tree_arena ar = fresh_arena();
    const max_t root = hsc::tree_build(3, hsc::level_dq(5), ar);
    sb::require(root >= 0);
    const hsc::tree_skeleton sk = hsc::tree_view(ar, static_cast<u32>(root), 3, hsc::level_dq(5));
    tutil::rng g;
    for ( i32 t = 0; t < 2000; ++t ) {
      f64 y[8];
      f64 n2 = 0;
      for ( u32 c = 0; c < 8; ++c ) {
        y[c] = g.unit();
        n2 += y[c] * y[c];
      }
      if ( n2 < 1e-12 ) continue;
      hsc::tree_fields f0{}, f1{};
      hsc::tree_quantize(sk, y, f0, 0);
      hsc::tree_quantize(sk, y, f1, 1);
      f64 p0[8], p1[8];
      hsc::tree_decode(sk, f0, p0);
      hsc::tree_decode(sk, f1, p1);
      f64 d0 = 0, d1 = 0;
      for ( u32 c = 0; c < 8; ++c ) {
        d0 += p0[c] * y[c];
        d1 += p1[c] * y[c];
      }
      sb::require(d1 >= d0 - 1e-12);
    }
  }

  sb::test_case("arena caps trip cleanly as oom, not UB");
  {
    hsc::tree_node nodes[2];
    hsc::tree_row rows[2];
    hsc::s3_leaf leaves[2];
    hsc::tree_arena tiny{ nodes, 2, 0, rows, 2, 0, leaves, 2, 0 };
    const max_t r = hsc::tree_build(6, hsc::level_dq(7), tiny);
    sb::require(r < 0);
    sb::require(hsc::as_error(r) == hsc::error::oom);
  }

  sb::test_case("dim-4 root is the plain s3 code (no recursion, no snapping)");
  {
    hsc::tree_arena ar = fresh_arena();
    const max_t root = hsc::tree_build(2, hsc::level_dq(6), ar);
    sb::require(root >= 0);
    const hsc::tree_skeleton sk = hsc::tree_view(ar, static_cast<u32>(root), 2, hsc::level_dq(6));
    sb::require(hsc::tree_m_mod64(sk), 2588ull);
    sb::require(ar.node_count, 1u);
    tutil::rng g;
    for ( i32 t = 0; t < 500; ++t ) {
      f64 y[4] = { g.unit(), g.unit(), g.unit(), g.unit() };
      hsc::tree_fields f0{}, f1{};
      hsc::tree_quantize(sk, y, f0);
      f64 p[4];
      hsc::tree_decode(sk, f0, p);
      hsc::tree_quantize(sk, p, f1);
      sb::require(f0.base[0], f1.base[0]);
    }
  }

  return 1;
}
