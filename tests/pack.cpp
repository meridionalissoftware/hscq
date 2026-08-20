// Exact big-integer packing. Guards: the FULL index sweep through the whole pipeline
// (unrank -> fields -> decode -> vector -> quantize -> fields -> rank == identity) for entire
// dim-8/16 codebooks -- this is the test the tree layer deferred; sampled rank/unrank identity
// at dims 32/64 where M ~ 2^90..2^148; shape_bits against golden ceil(log2 M); rejection of
// indices >= M; the wide-field bit I/O round-trip including >64-bit values; and the
// suspension band tables behind mode oct (prefix identity, full/sampled rank-unrank sweeps,
// agreement with the sphere layer's u64 off_mod on the quat lane).

#include "../src/hsc/codec/pack.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

static hsc::tree_node g_nodes[2048];
static hsc::tree_row g_rows[32768];
static hsc::s3_leaf g_leaves[65536];
static hsc::vq_index g_node_m[2048];
static hsc::vq_index g_row_off[32768];
static hsc::susp_band g_bands[512];
static hsc::vq_index g_band_off[513];

struct sbuilt {
  hsc::tree_skeleton tv;      // arena view; root meaningless for a suspension
  hsc::pack_tables pt;
  hsc::susp_skeleton ss;
  hsc::susp_pack sp;
};

sbuilt
sbuild(u32 child_dl, i32 level)
{
  hsc::tree_arena ar{ g_nodes, 2048, 0, g_rows, 32768, 0, g_leaves, 65536, 0 };
  hsc::susp_skeleton ss{};
  sb::require(hsc::susp_build(hsc::level_dq(level), child_dl, ar, g_bands, ss) >= 0);
  hsc::pack_tables pt{ g_node_m, g_row_off };
  sb::require(hsc::pack_build(ar, pt) >= 0);
  hsc::susp_pack sp{ g_band_off, 0 };
  sb::require(hsc::susp_pack_build(ss, pt, sp) >= 0);
  return { hsc::tree_view(ar, 0, child_dl, hsc::level_dq(level)), pt, ss, sp };
}

struct built {
  hsc::tree_skeleton sk;
  hsc::pack_tables pt;
};

built
build(u32 dim_log2, i32 level)
{
  hsc::tree_arena ar{ g_nodes, 2048, 0, g_rows, 32768, 0, g_leaves, 65536, 0 };
  const max_t root = hsc::tree_build(dim_log2, hsc::level_dq(level), ar);
  sb::require(root >= 0);
  hsc::pack_tables pt{ g_node_m, g_row_off };
  sb::require(hsc::pack_build(ar, pt) >= 0);
  return { hsc::tree_view(ar, static_cast<u32>(root), dim_log2, hsc::level_dq(level)), pt };
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
  sb::test_case("shape_bits == ceil(log2 M) goldens across dims and levels");
  {
    struct row {
      u32 dl;
      i32 lvl;
      u32 bits;
    };

    // ceil of the mirror's log2M: 11.174 16.883 21.267 | 15.880 36.090 | 38.577 89.595 | 57.746 147.680
    constexpr row k[]
        = { { 3, 3, 12 }, { 3, 5, 17 }, { 3, 6, 22 }, { 4, 3, 16 }, { 4, 6, 37 }, { 5, 5, 39 }, { 5, 7, 90 }, { 6, 5, 58 }, { 6, 7, 148 } };
    for ( const row &r : k ) {
      built b = build(r.dl, r.lvl);
      sb::require(hsc::shape_bits(b.sk, b.pt), r.bits);
    }
  }

  sb::test_case("FULL sweep dim 8 d=0.5: unrank -> decode -> quantize -> rank is the identity");
  {
    built b = build(3, 3);      // M = 2310
    const u64 M = static_cast<u64>(hsc::pack_m(b.sk, b.pt));
    sb::require(M, 2310ull);
    for ( u64 a = 0; a < M; ++a ) {
      hsc::tree_fields f{}, f2{};
      sb::require(hsc::pack_unrank(b.sk, b.pt, hsc::vq_index(a), f) >= 0);
      f64 p[8];
      hsc::tree_decode(b.sk, f, p);
      hsc::tree_quantize(b.sk, p, f2);
      sb::require(fields_equal(f, f2, 3));
      hsc::vq_index back(0u);
      hsc::pack_rank(b.sk, b.pt, f2, back);
      sb::require(back == hsc::vq_index(a));
    }
  }

  sb::test_case("FULL sweep dim 16 d=0.5: rank(unrank(a)) == a for all 60316 indices");
  {
    built b = build(4, 3);
    const u64 M = static_cast<u64>(hsc::pack_m(b.sk, b.pt));
    sb::require(M, 60316ull);
    tutil::rng g;
    for ( u64 a = 0; a < M; ++a ) {
      hsc::tree_fields f{};
      sb::require(hsc::pack_unrank(b.sk, b.pt, hsc::vq_index(a), f) >= 0);
      hsc::vq_index back(0u);
      hsc::pack_rank(b.sk, b.pt, f, back);
      sb::require(back == hsc::vq_index(a));
      if ( g.below(64) == 0 ) {      // geometry spot checks along the sweep
        hsc::tree_fields f2{};
        f64 p[16];
        hsc::tree_decode(b.sk, f, p);
        hsc::tree_quantize(b.sk, p, f2);
        sb::require(fields_equal(f, f2, 4));
      }
    }
  }

  sb::test_case("sampled rank/unrank identity at dims 32/64 (M ~ 2^90 .. 2^148)");
  {
    tutil::rng g;

    struct scase {
      u32 dl;
      i32 lvl;
      i32 trials;
    };

    for ( scase sc : { scase{ 5, 7, 400 }, scase{ 6, 7, 200 } } ) {
      const u32 dl = sc.dl;
      const i32 lvl = sc.lvl;
      const i32 trials = sc.trials;
      built b = build(dl, lvl);
      const u32 bits = hsc::shape_bits(b.sk, b.pt);
      for ( i32 t = 0; t < trials; ++t ) {
        hsc::vq_index r(0u);      // random value of shape_bits width, reduced mod M
        for ( u32 pos = 0; pos < bits; pos += 32 ) {
          hsc::vq_index c(static_cast<u32>(g.next()));
          c <<= pos;
          r |= c;
        }
        const auto qr = micron::math::divmod(r, hsc::pack_m(b.sk, b.pt));
        hsc::tree_fields f{};
        sb::require(hsc::pack_unrank(b.sk, b.pt, qr.rem, f) >= 0);
        hsc::vq_index back(0u);
        hsc::pack_rank(b.sk, b.pt, f, back);
        sb::require(back == qr.rem);
        // and the geometric fixed point holds for the decoded vector
        hsc::tree_fields f2{};
        f64 p[64];
        hsc::tree_decode(b.sk, f, p);
        hsc::tree_quantize(b.sk, p, f2);
        sb::require(fields_equal(f, f2, dl));
      }
    }
  }

  sb::test_case("index >= M is rejected as bad_stream before any tree walk");
  {
    built b = build(3, 3);
    hsc::tree_fields f{};
    sb::require(hsc::as_error(hsc::pack_unrank(b.sk, b.pt, hsc::pack_m(b.sk, b.pt), f)) == hsc::error::bad_stream);
    hsc::vq_index big = hsc::pack_m(b.sk, b.pt);
    big += 12345u;
    sb::require(hsc::as_error(hsc::pack_unrank(b.sk, b.pt, big, f)) == hsc::error::bad_stream);
  }

  sb::test_case("suspension tables: prefix identity, bit goldens, and u64 off_mod agreement");
  {
    // S^8 suspension (mode oct): band_off is the arbint prefix sum of the child cardinalities
    for ( i32 lvl : { 3, 6, 10 } ) {
      sbuilt b = sbuild(3, lvl);
      for ( u32 band = 0; band < b.ss.count; ++band ) {
        hsc::vq_index w = b.sp.band_off[band + 1];
        w -= b.sp.band_off[band];
        const bool pole = band == 0 || band == b.ss.count - 1;
        sb::require(w == (pole ? hsc::vq_index(1u) : b.pt.node_m[b.ss.bd[band].child]));
      }
      // total M mod 2^64 must equal the sphere layer's wrapping accumulation (the wire guard)
      hsc::vq_index m = hsc::susp_m(b.sp);
      sb::require(static_cast<u64>(m), b.ss.m_mod);
      if ( lvl == 6 ) sb::require(hsc::susp_bits(b.sp), 24u);
      if ( lvl == 10 ) sb::require(hsc::susp_bits(b.sp), 59u);      // L12 = 77 lives in the Python mirror (rows exceed these arenas)
    }
    // S^4 suspension (mode quat) never enters the arbint lane, but its u64 off_mod must be
    // exactly what susp_pack_build derives -- the two lanes may not drift
    sbuilt q = sbuild(2, 6);
    for ( u32 band = 0; band < q.ss.count; ++band ) sb::require(q.sp.band_off[band] == hsc::vq_index(q.ss.bd[band].off_mod));
    sb::require(hsc::susp_m(q.sp) == hsc::vq_index(q.ss.m_mod));
    sb::require(hsc::susp_bits(q.sp), 15u);
  }

  sb::test_case("suspension FULL sweep S^8 d=0.5: unrank -> decode -> quantize -> rank identity");
  {
    sbuilt b = sbuild(3, 3);      // M = 4552
    sb::require(hsc::susp_m(b.sp) == hsc::vq_index(4552u));
    tutil::rng g;
    for ( u64 a = 0; a < 4552; ++a ) {
      hsc::vq_index ia(a);
      hsc::tree_fields f{};
      u32 band = 0;
      hsc::pack_scratch ps;
      sb::require(hsc::susp_unrank(b.ss, b.tv, b.pt, b.sp, ia, band, f, ps) >= 0);
      hsc::vq_index back(0u);
      hsc::susp_rank(b.ss, b.tv, b.pt, b.sp, band, f, back, ps);
      sb::require(back == hsc::vq_index(a));
      if ( g.below(8) == 0 ) {      // geometry spot checks along the sweep
        f64 p[9];
        hsc::susp_decode(b.ss, b.tv, band, f, p);
        hsc::tree_fields f2{};
        sb::require(hsc::susp_quantize(b.ss, b.tv, p, f2), band);
        const bool pole = band == 0 || band == b.ss.count - 1;
        sb::require(pole || fields_equal(f, f2, 3));
      }
    }
  }

  sb::test_case("suspension sampled rank/unrank identity at S^8 L6 (M = 10714086)");
  {
    sbuilt b = sbuild(3, 6);
    tutil::rng g;
    const u32 bits = hsc::susp_bits(b.sp);
    for ( i32 t = 0; t < 500; ++t ) {
      hsc::vq_index r(0u);
      for ( u32 pos = 0; pos < bits; pos += 32 ) {
        hsc::vq_index c(static_cast<u32>(g.next()));
        c <<= pos;
        r |= c;
      }
      const auto qr = micron::math::divmod(r, hsc::susp_m(b.sp));
      hsc::vq_index a = qr.rem;
      hsc::tree_fields f{};
      u32 band = 0;
      hsc::pack_scratch ps;
      sb::require(hsc::susp_unrank(b.ss, b.tv, b.pt, b.sp, a, band, f, ps) >= 0);
      hsc::vq_index back(0u);
      hsc::susp_rank(b.ss, b.tv, b.pt, b.sp, band, f, back, ps);
      sb::require(back == qr.rem);
    }
  }

  sb::test_case("suspension index >= M is rejected as bad_stream");
  {
    sbuilt b = sbuild(3, 3);
    hsc::tree_fields f{};
    u32 band = 0;
    hsc::pack_scratch ps;
    hsc::vq_index m = hsc::susp_m(b.sp);
    sb::require(hsc::as_error(hsc::susp_unrank(b.ss, b.tv, b.pt, b.sp, m, band, f, ps)) == hsc::error::bad_stream);
    hsc::vq_index big = hsc::susp_m(b.sp);
    big += 12345u;
    sb::require(hsc::as_error(hsc::susp_unrank(b.ss, b.tv, b.pt, b.sp, big, band, f, ps)) == hsc::error::bad_stream);
  }

  sb::test_case("put_wide/get_wide round-trip, 1..200 bit fields, mixed with narrow fields");
  {
    tutil::rng g;
    static u8 buf[1 << 16];
    for ( i32 iter = 0; iter < 200; ++iter ) {
      u32 widths[64];
      hsc::vq_index vals[64];
      const u32 nf = 8 + static_cast<u32>(g.below(56));
      hsc::bits::bitwriter w{ .acc = 0, .cnt = 0, .out = buf, .fast_end = buf + sizeof(buf) - 8 };
      for ( u32 i = 0; i < nf; ++i ) {
        widths[i] = 1 + static_cast<u32>(g.below(200));
        vals[i] = hsc::vq_index(0u);
        for ( u32 pos = 0; pos < widths[i]; pos += 32 ) {
          hsc::vq_index c(static_cast<u32>(g.next()));
          c <<= pos;
          vals[i] |= c;
        }
        // mask to width: keep only the low `widths[i]` bits
        hsc::vq_index mask(1u);
        mask <<= widths[i];
        mask -= 1u;
        vals[i] &= mask;
        hsc::put_wide(w, vals[i], widths[i]);
      }
      const u8 *end = w.finish();
      hsc::bits::bitreader r{ .p = buf, .end = end };
      for ( u32 i = 0; i < nf; ++i ) {
        hsc::vq_index v(0u);
        sb::require(hsc::get_wide(r, widths[i], v));
        sb::require(v == vals[i]);
      }
    }
  }

  sb::test_case("a zero-width shape field is legal (M == 1 degenerate code)");
  {
    hsc::tree_arena ar{ g_nodes, 2048, 0, g_rows, 32768, 0, g_leaves, 65536, 0 };
    const max_t root = hsc::tree_build(2, hsc::dq_max, ar);
    sb::require(root >= 0);
    hsc::pack_tables pt{ g_node_m, g_row_off };
    sb::require(hsc::pack_build(ar, pt) >= 0);
    const hsc::tree_skeleton sk = hsc::tree_view(ar, static_cast<u32>(root), 2, hsc::dq_max);
    sb::require(hsc::shape_bits(sk, pt), 0u);
    hsc::tree_fields f{};
    sb::require(hsc::pack_unrank(sk, pt, hsc::vq_index(0u), f) >= 0);
    hsc::vq_index back(7u);
    hsc::pack_rank(sk, pt, f, back);
    sb::require(back == hsc::vq_index(0u));
  }

  return 1;
}
