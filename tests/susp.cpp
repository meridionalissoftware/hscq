// The cap-anchored suspension quantizer behind modes quat (S^4 over S^3 children) and oct
// (S^8 over S^7 children). Guards: golden cardinalities against the Python mirror
// (scripts/compression_ratio_model.py), full index bijection sweeps on the u64 lane,
// brute-force minimum pairwise distance on whole codebooks (both bases), exact pole
// codewords, off-sphere clamp robustness, poles-only collapse at coarse d, and the
// refine-never-worse guarantee. The Hopf projections/lifts feeding this quantizer are
// tested in quat.cpp / oct.cpp, not here.

#include "../src/hsc/sphere/susp.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

constexpr u64
ct_m(u32 child_dl, u32 dq)
{
  hsc::tree_node nodes[64]{};
  hsc::tree_row rows[64]{};
  hsc::s3_leaf leaves[512]{};
  hsc::susp_band bands[64]{};
  hsc::tree_arena ar{ nodes, 64, 0, rows, 64, 0, leaves, 512, 0 };
  hsc::susp_skeleton ss{};
  if ( hsc::susp_build(dq, child_dl, ar, bands, ss) < 0 ) return 0;
  return ss.m_mod;
}

// golden cardinalities from the validated Python mirror, at the exact wire d_q values
// (equator-anchored layout: interior band count forced odd)
static_assert(ct_m(2, hsc::level_dq(1)) == 18);            // S^4, d = 0.9 (1 equator band + 2 poles)
static_assert(ct_m(2, hsc::level_dq(2)) == 86);            // S^4, d = 0.7
static_assert(ct_m(2, hsc::level_dq(3)) == 332);           // S^4, d = 0.5
static_assert(ct_m(2, hsc::level_dq(5)) == 2978);          // S^4, d = 0.3
static_assert(ct_m(2, hsc::level_dq(6)) == 16720);         // S^4, d = 0.2
static_assert(ct_m(3, hsc::level_dq(3)) == 4552);          // S^8, d = 0.5
static_assert(ct_m(3, hsc::level_dq(5)) == 356950);        // S^8, d = 0.3
static_assert(ct_m(3, hsc::level_dq(6)) == 10923842);      // S^8, d = 0.2
// the d = 2 floor and everything past d = sqrt(2) collapse to the two poles -- M = 2, never 1:
// the quotient family cannot go degenerate (record_bits >= 1 at every valid d_q)
static_assert(ct_m(2, hsc::dq_max) == 2);
static_assert(ct_m(3, hsc::dq_max) == 2);
static_assert(ct_m(2, hsc::dq_of(1.5)) == 2);
static_assert(ct_m(3, hsc::dq_of(1.5)) == 2);
// band-count cap: worst case at the validity floor dq_min
static_assert(hsc::susp_band_count(hsc::dq_min) == 31411);
static_assert(hsc::susp_band_count(hsc::dq_min) <= hsc::susp_max_bands);

static hsc::tree_node g_nodes[512];
static hsc::tree_row g_rows[1024];
static hsc::s3_leaf g_leaves[8192];
static hsc::susp_band g_bands[64];

struct built {
  hsc::tree_arena ar;
  hsc::tree_skeleton tv;      // arena view; root is meaningless for a suspension
  hsc::susp_skeleton ss;
};

built
build(u32 child_dl, i32 lvl)
{
  built b{ hsc::tree_arena{ g_nodes, 512, 0, g_rows, 1024, 0, g_leaves, 8192, 0 }, {}, {} };
  const max_t r = hsc::susp_build(hsc::level_dq(lvl), child_dl, b.ar, g_bands, b.ss);
  sb::require(r >= 0);
  b.tv = hsc::tree_view(b.ar, 0, child_dl, hsc::level_dq(lvl));
  return b;
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

// a random point on S^n as (vector, height): sum of uniforms per coordinate, normalized
void
random_point(tutil::rng &g, u32 n, f64 *p)
{
  f64 s = 0;
  for ( u32 c = 0; c <= n; ++c ) {
    const f64 u = (static_cast<f64>(g.next() & 0xFFFFFFu) / 8388608.0) - 1.0;
    p[c] = u;
    s += u * u;
  }
  const f64 inv = 1.0 / __builtin_sqrt(s > 0.0 ? s : 1.0);
  for ( u32 c = 0; c <= n; ++c ) p[c] *= inv;
}

}      // namespace

int
main()
{
  sb::test_case("S^4: full bijection sweep on the u64 lane, quantize(decode(a)) == a for every index");
  {
    for ( i32 lvl : { 1, 2, 3, 4, 5, 6 } ) {
      const built b = build(2, lvl);
      for ( u64 a = 0; a < b.ss.m_mod; ++a ) {
        const u32 band = hsc::__susp_band_of(b.ss, a);
        hsc::tree_fields f{};
        f.base[0] = a - b.ss.bd[band].off_mod;
        f64 p[5]{};
        hsc::susp_decode(b.ss, b.tv, band, f, p);
        hsc::tree_fields f2{};
        const u32 band2 = hsc::susp_quantize(b.ss, b.tv, p, f2);
        sb::require(band2, band);
        const bool pole = band == 0 || band == b.ss.count - 1;
        sb::require(pole || f2.base[0] == f.base[0]);
      }
    }
  }

  sb::test_case("S^8: fixed point over the whole L3 codebook, per-band field enumeration");
  {
    const built b = build(3, 3);
    u64 seen = 0;
    for ( u32 band = 0; band < b.ss.count; ++band ) {
      const bool pole = band == 0 || band == b.ss.count - 1;
      if ( pole ) {
        f64 p[9]{};
        hsc::tree_fields f{};
        hsc::susp_decode(b.ss, b.tv, band, f, p);
        hsc::tree_fields f2{};
        sb::require(hsc::susp_quantize(b.ss, b.tv, p, f2), band);
        ++seen;
        continue;
      }
      // a dim-8 child is one internal node over two dim-4 children per row
      const hsc::tree_skeleton cv = hsc::susp_child_view(b.ss, b.tv, band);
      const hsc::tree_node &nd = cv.nodes[cv.root];
      for ( u32 x = 0; x < nd.count; ++x ) {
        const hsc::tree_row &rw = cv.rows[nd.rows_at + x];
        const u64 m1 = cv.nodes[rw.c1].m_mod;
        const u64 m2 = cv.nodes[rw.c2].m_mod;
        for ( u64 a1 = 0; a1 < m1; ++a1 )
          for ( u64 a2 = 0; a2 < m2; ++a2 ) {
            hsc::tree_fields f{};
            f.leaf[0] = x;
            f.base[0] = a1;
            f.base[1] = a2;
            f64 p[9]{};
            hsc::susp_decode(b.ss, b.tv, band, f, p);
            hsc::tree_fields f2{};
            const u32 band2 = hsc::susp_quantize(b.ss, b.tv, p, f2);
            sb::require(band2, band);
            sb::require(fields_equal(f, f2, 3));
            ++seen;
          }
      }
    }
    sb::require(seen, 4552ull);      // every codeword reached exactly once
  }

  sb::test_case("brute-force minimum pairwise distance >= d on whole codebooks, both bases");
  {
    // S^4 at L2/L3/L5 (M = 86 / 332 / 2978)
    static f64 pts4[2978][5];
    for ( i32 lvl : { 2, 3, 5 } ) {
      const built b = build(2, lvl);
      for ( u64 a = 0; a < b.ss.m_mod; ++a ) {
        const u32 band = hsc::__susp_band_of(b.ss, a);
        hsc::tree_fields f{};
        f.base[0] = a - b.ss.bd[band].off_mod;
        hsc::susp_decode(b.ss, b.tv, band, f, pts4[a]);
      }
      f64 dmin2 = 16.0;
      for ( u64 a = 0; a < b.ss.m_mod; ++a )
        for ( u64 c = a + 1; c < b.ss.m_mod; ++c ) {
          f64 s = 0;
          for ( i32 k = 0; k < 5; ++k ) {
            const f64 t = pts4[a][k] - pts4[c][k];
            s += t * t;
          }
          if ( s < dmin2 ) dmin2 = s;
        }
      sb::require(__builtin_sqrt(dmin2) >= b.ss.d - 1e-9);
    }
    // S^8 at L3 (M = 4552), enumerated per band
    static f64 pts8[4552][9];
    {
      const built b = build(3, 3);
      u64 at = 0;
      for ( u32 band = 0; band < b.ss.count; ++band ) {
        const bool pole = band == 0 || band == b.ss.count - 1;
        if ( pole ) {
          hsc::tree_fields f{};
          hsc::susp_decode(b.ss, b.tv, band, f, pts8[at++]);
          continue;
        }
        const hsc::tree_skeleton cv = hsc::susp_child_view(b.ss, b.tv, band);
        const hsc::tree_node &nd = cv.nodes[cv.root];
        for ( u32 x = 0; x < nd.count; ++x ) {
          const hsc::tree_row &rw = cv.rows[nd.rows_at + x];
          for ( u64 a1 = 0; a1 < cv.nodes[rw.c1].m_mod; ++a1 )
            for ( u64 a2 = 0; a2 < cv.nodes[rw.c2].m_mod; ++a2 ) {
              hsc::tree_fields f{};
              f.leaf[0] = x;
              f.base[0] = a1;
              f.base[1] = a2;
              hsc::susp_decode(b.ss, b.tv, band, f, pts8[at++]);
            }
        }
      }
      sb::require(at, 4552ull);
      f64 dmin2 = 16.0;
      for ( u64 a = 0; a < at; ++a )
        for ( u64 c = a + 1; c < at; ++c ) {
          f64 s = 0;
          for ( i32 k = 0; k < 9; ++k ) {
            const f64 t = pts8[a][k] - pts8[c][k];
            s += t * t;
          }
          if ( s < dmin2 ) dmin2 = s;
        }
      sb::require(__builtin_sqrt(dmin2) >= b.ss.d - 1e-9);
    }
  }

  sb::test_case("decoded codewords are unit; poles are EXACT (0,...,0,+-1) codewords");
  {
    const built b = build(2, 6);
    tutil::rng g;
    for ( i32 t = 0; t < 2000; ++t ) {
      const u64 a = g.next() % b.ss.m_mod;
      const u32 band = hsc::__susp_band_of(b.ss, a);
      hsc::tree_fields f{};
      f.base[0] = a - b.ss.bd[band].off_mod;
      f64 p[5]{};
      hsc::susp_decode(b.ss, b.tv, band, f, p);
      f64 n2 = 0;
      for ( i32 k = 0; k < 5; ++k ) n2 += p[k] * p[k];
      sb::require(n2 > 1.0 - 1e-12 && n2 < 1.0 + 1e-12);
    }
    // the poles decode bit-exactly -- the [x:0]/[0:y] classes are why the layout is cap-anchored
    f64 pn[5]{}, ps[5]{};
    hsc::tree_fields f{};
    hsc::susp_decode(b.ss, b.tv, 0, f, pn);
    hsc::susp_decode(b.ss, b.tv, b.ss.count - 1, f, ps);
    for ( i32 k = 0; k < 4; ++k ) {
      sb::require(pn[k] == 0.0);
      sb::require(ps[k] == 0.0);
    }
    sb::require(pn[4] == 1.0);
    sb::require(ps[4] == -1.0);
    // and quantize deterministically from any whisker direction
    for ( i32 t = 0; t < 16; ++t ) {
      f64 p[5] = { 0, 0, 0, 0, 1.0 };
      p[t % 4] = 1e-9;
      hsc::tree_fields tf{};
      sb::require(hsc::susp_quantize(b.ss, b.tv, p, tf), 0u);
      p[4] = -1.0;
      sb::require(hsc::susp_quantize(b.ss, b.tv, p, tf), b.ss.count - 1);
    }
  }

  sb::test_case("off-sphere inputs clamp instead of NaN; band-boundary midpoints stay in range");
  {
    const built b = build(2, 3);
    hsc::tree_fields f{};
    const f64 over[5] = { 0.0, 0.0, 0.0, 0.0, 1.5 };
    const f64 under[5] = { 0.0, 0.0, 0.0, 0.0, -1.5 };
    sb::require(hsc::susp_quantize(b.ss, b.tv, over, f), 0u);
    sb::require(hsc::susp_quantize(b.ss, b.tv, under, f), b.ss.count - 1);
    // exactly at the pole/interior midpoint and the interior grid edges: any valid band, stable
    for ( const f64 th : { 0.5 * b.ss.th_first, b.ss.th_first, b.ss.th_last, 0.5 * (b.ss.th_last + hsc::k_pi) } ) {
      f64 st = 0, ct = 0;
      micron::sincos(th, st, ct);
      const f64 p[5] = { st, 0.0, 0.0, 0.0, ct };
      const u32 b1 = hsc::susp_quantize(b.ss, b.tv, p, f);
      const u32 b2 = hsc::susp_quantize(b.ss, b.tv, p, f);
      sb::require(b1 < b.ss.count);
      sb::require(b1, b2);
    }
  }

  sb::test_case("refine never worsens the codeword dot, S^4 and S^8");
  {
    tutil::rng g;
    for ( u32 child_dl : { 2u, 3u } ) {
      const built b = build(child_dl, 5);
      const u32 n = 1u << child_dl;
      for ( i32 t = 0; t < 2000; ++t ) {
        f64 p[9]{};
        random_point(g, n, p);
        hsc::tree_fields f0{}, f1{};
        const u32 b0 = hsc::susp_quantize(b.ss, b.tv, p, f0, 0);
        const u32 b1 = hsc::susp_quantize(b.ss, b.tv, p, f1, 1);
        f64 v0[9]{}, v1[9]{};
        hsc::susp_decode(b.ss, b.tv, b0, f0, v0);
        hsc::susp_decode(b.ss, b.tv, b1, f1, v1);
        f64 d0 = 0, d1 = 0;
        for ( u32 k = 0; k <= n; ++k ) {
          d0 += v0[k] * p[k];
          d1 += v1[k] * p[k];
        }
        sb::require(d1 >= d0 - 1e-15);
      }
    }
  }

  return 1;
}
