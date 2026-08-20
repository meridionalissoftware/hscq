//  Oct mode geometry, algebra first. Guards: the Cayley-Dickson multiplication table of THIS
//  convention (storage (e1..e7, e0), split A = (o0,o1,o2,o7), B = (o4,o5,o6,o3), product
//  (A1,B1)(A2,B2) = (A1A2 - conj(B2)B1, B2A1 + B1conj(A2))), the conjugation anti-automorphism,
//  norm multiplicativity (the composition-algebra property the Hopf map depends on), the
//  projection/section pair and fixed point over the WHOLE L3 codebook, TRUE fiber invariance
//  via the exact graph-sphere parametrization (fibers are {(a, q a)} -- NOT a group orbit),
//  scale invariance, the base-chordal error bound, refine-never-worse, and bad-input rejection.

#include "../src/hsc/codec/oct.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

void
bas(u32 label, f64 sign, f64 *o)
{
  for ( u32 k = 0; k < 8; ++k ) o[k] = 0.0;
  o[label == 0 ? 7 : label - 1] = sign;
}

bool
oct_eq(const f64 *a, const f64 *b)
{
  for ( u32 k = 0; k < 8; ++k )
    if ( a[k] != b[k] ) return false;
  return true;
}

void
random_oct(tutil::rng &g, f64 *o)
{
  for ( u32 k = 0; k < 8; ++k ) o[k] = g.unit();
}

void
random_unit_oct(tutil::rng &g, f64 *o)
{
  random_oct(g, o);
  const f64 n2 = hsc::__fma_norm2(o, 8);
  const f64 inv = 1.0 / __builtin_sqrt(n2 > 1e-12 ? n2 : 1.0);
  for ( u32 k = 0; k < 8; ++k ) o[k] *= inv;
}

bool
fields_equal(const hsc::tree_fields &a, const hsc::tree_fields &b)
{
  for ( u32 i = 0; i < hsc::tree_inode_count(3); ++i )
    if ( a.leaf[i] != b.leaf[i] ) return false;
  for ( u32 i = 0; i < hsc::tree_bnode_count(3); ++i )
    if ( a.base[i] != b.base[i] ) return false;
  return true;
}

void
fiber_mate(const f64 *p, const f64 *u, f32 *z)
{
  const f64 h = p[8];
  const f64 s = __builtin_sqrt((1.0 - h) * 0.5);
  const f64 c = __builtin_sqrt((1.0 + h) * 0.5);
  f64 x[8], y[8];
  if ( s < 1e-6 ) {
    for ( u32 k = 0; k < 8; ++k ) {
      x[k] = c * u[k];
      y[k] = 0.0;
    }
  } else {
    f64 xv[8];
    hsc::oct_mul(p, u, xv);
    for ( u32 k = 0; k < 8; ++k ) {
      x[k] = xv[k] / (2.0 * s);
      y[k] = s * u[k];
    }
  }
  for ( u32 k = 0; k < 8; ++k ) {
    z[k] = static_cast<f32>(x[k]);
    z[8 + k] = static_cast<f32>(y[k]);
  }
}

static hsc::tree_node g_nodes[512];
static hsc::tree_row g_rows[1024];
static hsc::s3_leaf g_leaves[8192];
static hsc::susp_band g_bands[64];

}      //  namespace

int
main()
{
  sb::test_case("the multiplication table of this Cayley-Dickson convention");
  {
    struct rel {
      u32 a, b, c;
      f64 sign;
    };

    constexpr rel rels[] = {
      { 1, 2, 3, 1.0 }, { 2, 3, 1, 1.0 }, { 3, 1, 2, 1.0 },  { 1, 4, 5, 1.0 },  { 2, 4, 6, 1.0 },  { 3, 4, 7, 1.0 },
      { 4, 5, 1, 1.0 }, { 2, 5, 7, 1.0 }, { 5, 6, 3, -1.0 }, { 6, 7, 1, -1.0 }, { 4, 1, 5, -1.0 }, { 5, 5, 0, -1.0 },
    };
    for ( const rel &r : rels ) {
      f64 a[8], b[8], want[8], got[8];
      bas(r.a, 1.0, a);
      bas(r.b, 1.0, b);
      bas(r.c, r.sign, want);
      hsc::oct_mul(a, b, got);
      sb::require(oct_eq(got, want));
    }

    tutil::rng g;
    for ( i32 t = 0; t < 32; ++t ) {
      f64 e0[8], o[8], l[8], r[8];
      bas(0, 1.0, e0);
      random_oct(g, o);
      hsc::oct_mul(e0, o, l);
      hsc::oct_mul(o, e0, r);
      sb::require(oct_eq(l, o));
      sb::require(oct_eq(r, o));
    }
  }

  sb::test_case("conjugation is an anti-automorphism and the norm is multiplicative");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 1000; ++t ) {
      f64 a[8], b[8], ab[8], cab[8], ca[8], cb[8], cbca[8];
      random_oct(g, a);
      random_oct(g, b);
      hsc::oct_mul(a, b, ab);
      hsc::oct_conj(ab, cab);
      hsc::oct_conj(a, ca);
      hsc::oct_conj(b, cb);
      hsc::oct_mul(cb, ca, cbca);
      f64 e2 = 0;
      for ( u32 k = 0; k < 8; ++k ) {
        const f64 e = cab[k] - cbca[k];
        e2 += e * e;
      }
      sb::require(__builtin_sqrt(e2) < 1e-12 * (1.0 + __builtin_sqrt(hsc::__fma_norm2(ab, 8))));

      const f64 nab = __builtin_sqrt(hsc::__fma_norm2(ab, 8));
      const f64 na = __builtin_sqrt(hsc::__fma_norm2(a, 8));
      const f64 nb = __builtin_sqrt(hsc::__fma_norm2(b, 8));
      const f64 rel = nab / (na * nb > 1e-300 ? na * nb : 1.0) - 1.0;
      sb::require(rel < 1e-12 && rel > -1e-12);
    }
  }

  hsc::tree_arena ar{ g_nodes, 512, 0, g_rows, 1024, 0, g_leaves, 8192, 0 };
  hsc::susp_skeleton ss{};
  sb::require(hsc::susp_build(hsc::level_dq(3), 3, ar, g_bands, ss) >= 0);
  const hsc::tree_skeleton tv = hsc::tree_view(ar, 0, 3, hsc::level_dq(3));

  sb::test_case("section identity and fixed point over the whole L3 codebook");
  {
    u64 seen = 0;
    for ( u32 band = 0; band < ss.count; ++band ) {
      const bool pole = band == 0 || band == ss.count - 1;
      if ( pole ) {
        f32 z[16];
        hsc::tree_fields f{};
        hsc::oct_reconstruct(ss, tv, band, f, z);
        hsc::tree_fields f2{};
        u32 band2 = 0;
        sb::require(hsc::oct_quantize(ss, tv, z, band2, f2) >= 0);
        sb::require(band2, band);
        ++seen;
        continue;
      }
      const hsc::tree_skeleton cv = hsc::susp_child_view(ss, tv, band);
      const hsc::tree_node &nd = cv.nodes[cv.root];
      for ( u32 x = 0; x < nd.count; ++x ) {
        const hsc::tree_row &rw = cv.rows[nd.rows_at + x];
        for ( u64 a1 = 0; a1 < cv.nodes[rw.c1].m_mod; ++a1 )
          for ( u64 a2 = 0; a2 < cv.nodes[rw.c2].m_mod; ++a2 ) {
            hsc::tree_fields f{};
            f.leaf[0] = x;
            f.base[0] = a1;
            f.base[1] = a2;
            f32 z[16];
            hsc::oct_reconstruct(ss, tv, band, f, z);

            f64 n2 = 0;
            for ( i32 c = 0; c < 16; ++c ) n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
            sb::require(n2 > 1.0 - 1e-6 && n2 < 1.0 + 1e-6);
            for ( i32 c = 0; c < 7; ++c ) sb::require(hsc::__f2u(z[c]) == 0u);
            sb::require(static_cast<f64>(z[7]) >= 0.0);
            hsc::tree_fields f2{};
            u32 band2 = 0;
            sb::require(hsc::oct_quantize(ss, tv, z, band2, f2) >= 0);
            sb::require(band2, band);
            sb::require(fields_equal(f, f2));
            ++seen;
          }
      }
    }
    sb::require(seen, 4552ull);

    f32 z[16];
    hsc::tree_fields f{};
    hsc::oct_reconstruct(ss, tv, ss.count - 1, f, z);
    for ( i32 c = 0; c < 15; ++c ) sb::require(z[c] == 0.0f);
    sb::require(z[15] == 1.0f);
  }

  sb::test_case("fiber invariance: the whole S^7 fiber (graph sphere) quantizes to one class");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 800; ++t ) {
      f32 z[16];
      f64 n2 = 0;
      for ( i32 c = 0; c < 16; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      f64 p[9]{};
      sb::require(hsc::oct_project(z, p) >= 0);
      hsc::tree_fields f0{};
      u32 b0 = 0;
      sb::require(hsc::oct_quantize(ss, tv, z, b0, f0) >= 0);
      for ( i32 k = 0; k < 12; ++k ) {
        f64 u[8];
        random_unit_oct(g, u);
        f32 zm[16];
        fiber_mate(p, u, zm);
        hsc::tree_fields fm{};
        u32 bm = 0;
        sb::require(hsc::oct_quantize(ss, tv, zm, bm, fm) >= 0);
        sb::require(bm, b0);
        const bool pole = b0 == 0 || b0 == ss.count - 1;
        sb::require(pole || fields_equal(fm, f0));
      }
    }
  }

  sb::test_case("scale invariance: lambda * (o0, o1) quantizes identically");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 500; ++t ) {
      f32 z[16];
      f64 n2 = 0;
      for ( i32 c = 0; c < 16; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      hsc::tree_fields f0{}, f1{}, f2{};
      u32 b0 = 0, b1 = 0, b2 = 0;
      sb::require(hsc::oct_quantize(ss, tv, z, b0, f0) >= 0);
      f32 zs[16];
      for ( i32 c = 0; c < 16; ++c ) zs[c] = z[c] * 37.5f;
      sb::require(hsc::oct_quantize(ss, tv, zs, b1, f1) >= 0);
      for ( i32 c = 0; c < 16; ++c ) zs[c] = z[c] * 0.001f;
      sb::require(hsc::oct_quantize(ss, tv, zs, b2, f2) >= 0);
      sb::require(b1, b0);
      sb::require(b2, b0);
      const bool pole = b0 == 0 || b0 == ss.count - 1;
      sb::require(pole || (fields_equal(f1, f0) && fields_equal(f2, f0)));
    }
  }

  sb::test_case("base-chordal error bound for unit inputs");
  {
    tutil::rng g;
    const f64 d = hsc::d_of(hsc::level_dq(3));
    f64 se = 0, worst = 0;
    i32 cnt = 0;
    for ( i32 t = 0; t < 4000; ++t ) {
      f32 z[16];
      f64 n2 = 0;
      for ( i32 c = 0; c < 16; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      f64 p[9]{}, ph[9]{};
      sb::require(hsc::oct_project(z, p) >= 0);
      hsc::tree_fields f{};
      u32 band = 0;
      sb::require(hsc::oct_quantize(ss, tv, z, band, f) >= 0);
      f32 zh[16];
      hsc::oct_reconstruct(ss, tv, band, f, zh);
      sb::require(hsc::oct_project(zh, ph) >= 0);
      f64 e2 = 0;
      for ( i32 c = 0; c < 9; ++c ) {
        const f64 e = ph[c] - p[c];
        e2 += e * e;
      }
      const f64 dc = __builtin_sqrt(e2);
      if ( dc > worst ) worst = dc;
      se += e2;
      ++cnt;
    }

    sb::require(worst <= d * 2.5 + 1e-6);
    sb::require(__builtin_sqrt(se / cnt) < d * 1.25);
  }

  sb::test_case("refine never worsens the base-chordal error");
  {
    tutil::rng g;
    for ( i32 t = 0; t < 2000; ++t ) {
      f32 z[16];
      f64 n2 = 0;
      for ( i32 c = 0; c < 16; ++c ) {
        z[c] = static_cast<f32>(g.unit());
        n2 += static_cast<f64>(z[c]) * static_cast<f64>(z[c]);
      }
      if ( n2 < 1e-6 ) continue;
      f64 p[9]{}, p0[9]{}, p1[9]{};
      sb::require(hsc::oct_project(z, p) >= 0);
      hsc::tree_fields f0{}, f1{};
      u32 b0 = 0, b1 = 0;
      sb::require(hsc::oct_quantize(ss, tv, z, b0, f0, 0) >= 0);
      sb::require(hsc::oct_quantize(ss, tv, z, b1, f1, 1) >= 0);
      hsc::susp_decode(ss, tv, b0, f0, p0);
      hsc::susp_decode(ss, tv, b1, f1, p1);
      f64 e0 = 0, e1 = 0;
      for ( i32 c = 0; c < 9; ++c ) {
        const f64 a = p0[c] - p[c], b = p1[c] - p[c];
        e0 += a * a;
        e1 += b * b;
      }
      sb::require(e1 <= e0 + 1e-12);
    }
  }

  sb::test_case("zero and non-finite blocks are rejected as bad_value");
  {
    f32 z[16]{};
    hsc::tree_fields f{};
    u32 band = 0;
    sb::require(hsc::as_error(hsc::oct_quantize(ss, tv, z, band, f)) == hsc::error::bad_value);
    f32 zn[16]{};
    zn[0] = 1.0f;
    zn[9] = hsc::__u2f(0x7FC00000u);
    sb::require(hsc::as_error(hsc::oct_quantize(ss, tv, zn, band, f)) == hsc::error::bad_value);
  }

  return 1;
}
