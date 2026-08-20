// The fixed-rate contract, asserted rather than assumed. hsc's compressed size is a closed-form
// function of (mode, dim_log2, d_q, gain_bits, n_elems) and of nothing about the data, so
// hsc::rate()'s per-block accounting must reproduce the byte count hopf_into actually emits --
// EXACTLY, in every cell. A mismatch here is a bug in the accounting or in the packer, never a
// tolerance. Also pins the frozen level 1..9 d_q constants, the appended 10..16 ladder and its
// M(4) cardinalities, the clamp at both ends, and degenerate() against the cells where the
// recursion collapses to M == 1 (the shape field vanishes and only the gain survives).

#include "../src/hsc/hsc.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>
#include <micron/vector/vector.hpp>

#include <snowball/snowball.hpp>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// the frozen ladder: levels 1..9 are load-bearing for every golden in tests/ and every number in
// CLAUDE.md's perf reference. If one of these moves, the whole published record is invalidated.

static_assert(hsc::level_dq(1) == 15099494);
static_assert(hsc::level_dq(2) == 11744051);
static_assert(hsc::level_dq(3) == 8388608);
static_assert(hsc::level_dq(4) == 6710886);
static_assert(hsc::level_dq(5) == 5033165);
static_assert(hsc::level_dq(6) == 3355443);
static_assert(hsc::level_dq(7) == 1677722);
static_assert(hsc::level_dq(8) == 838861);
static_assert(hsc::level_dq(9) == 335544);

// the appended fine end, and that it lands exactly on the API floor with no gap
static_assert(hsc::level_dq(10) == 167772);
static_assert(hsc::level_dq(11) == 83886);
static_assert(hsc::level_dq(12) == 33554);
static_assert(hsc::level_dq(13) == 16777);
static_assert(hsc::level_dq(14) == 8389);
static_assert(hsc::level_dq(15) == 3355);
static_assert(hsc::level_dq(16) == 1678);
static_assert(hsc::level_dq(16) == hsc::dq_min);
static_assert(hsc::dq_of(1e-4) == hsc::dq_min);

// clamp at both ends
static_assert(hsc::level_dq(0) == hsc::level_dq(1));
static_assert(hsc::level_dq(-7) == hsc::level_dq(1));
static_assert(hsc::level_dq(17) == hsc::level_dq(16));
static_assert(hsc::level_dq(9999) == hsc::level_dq(16));

// strictly decreasing d_q: the ladder is monotone in quality, which the quality search relies on
static_assert(hsc::level_dq(1) > hsc::level_dq(2) && hsc::level_dq(9) > hsc::level_dq(10));
static_assert(hsc::level_dq(15) > hsc::level_dq(16));

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// comptime cardinality goldens for the new levels (4D base case; mirrors tests/s3.cpp's style)

static_assert(hsc::ct::s3_m<hsc::level_dq(10)>() == 22704306);
static_assert(hsc::ct::s3_m<hsc::level_dq(11)>() == 181975364);
static_assert(hsc::ct::s3_m<hsc::level_dq(12)>() == 2847067414);
static_assert(hsc::ct::s3_m<hsc::level_dq(13)>() == 22785126711ull);

// the new consteval rate probes, against tests/pack.cpp's shape_bits goldens
static_assert(hsc::ct::shape_bits<3, hsc::level_dq(3)>() == 12);
static_assert(hsc::ct::shape_bits<3, hsc::level_dq(5)>() == 17);
static_assert(hsc::ct::shape_bits<3, hsc::level_dq(6)>() == 22);
static_assert(hsc::ct::shape_bits<4, hsc::level_dq(3)>() == 16);
static_assert(hsc::ct::shape_bits<4, hsc::level_dq(6)>() == 37);
static_assert(hsc::ct::rate_bits<3, hsc::level_dq(6), 8>() == 30);      // dim8 L6 = 3.75 bits/byte
static_assert(hsc::ct::s2_bits<hsc::level_dq(3)>() == 6);               // M_S2 = 46
static_assert(hsc::ct::s2_bits<hsc::level_dq(7)>() == 11);              // M_S2 = 1236
static_assert(hsc::ct::s4_bits<hsc::level_dq(3)>() == 9);               // M_S4 = 332
static_assert(hsc::ct::s4_bits<hsc::level_dq(6)>() == 15);              // M_S4 = 16720 (vs unit d8: 22 -> 7 bits/block saved)
static_assert(hsc::ct::s8_bits<hsc::level_dq(3)>() == 13);              // M_S8 = 4552
static_assert(hsc::ct::s8_bits<hsc::level_dq(6)>() == 24);              // M_S8 = 10923842 (vs unit d16: 37 -> 13 bits/block saved)

// the bit-exact bin cell (hsc::opts_exact_bytes): dim4 L13 = shape 35 + gain 8 = 43 bits per 4 bytes
// = 10.75 bits/byte, ratio 0.744x. These two are what makes that an API number rather than a claim.
static_assert(hsc::ct::shape_bits<2, hsc::level_dq(13)>() == 35);
static_assert(hsc::ct::rate_bits<2, hsc::level_dq(13), 8>() == 43);

// exact_bytes() is a cell predicate, so pin it in both directions -- what it must accept and, more
// importantly, everything it must refuse. 0 from bin_exact_level reads "never measured", not "no".
static_assert(hsc::bin_exact_level(2) == 13);
static_assert(hsc::bin_exact_level(3) == 0);
static_assert(hsc::exact_bytes(hsc::opts_exact_bytes));
static_assert(hsc::exact_bytes({ .level = 14, .dim_log2 = 2 }));                            // finer still
static_assert(hsc::exact_bytes({ .level = 16, .dim_log2 = 2 }));                            // the floor
static_assert(hsc::exact_bytes({ .d = 5e-4, .dim_log2 = 2 }));                              // d, not level
static_assert(!hsc::exact_bytes({ .level = 12, .dim_log2 = 2 }));                           // coarser: ~5 bytes / 512 KiB
static_assert(!hsc::exact_bytes({ .level = 13, .dim_log2 = 3 }));                           // dim8 unmeasured
static_assert(!hsc::exact_bytes({ .m = hsc::mode::vec, .level = 13, .dim_log2 = 2 }));      // no integers to snap to
static_assert(!hsc::exact_bytes({ .level = 13, .dim_log2 = 2, .gain_bits = 4 }));           // coarser gain field
static_assert(!hsc::exact_bytes({ .level = 13, .dim_log2 = 2, .transform = true }));        // measured with bit2 off

// the degenerate corner, proven at compile time: at dim >= 16 level 1 has no shape field at all
static_assert(hsc::ct::shape_bits<2, hsc::level_dq(1)>() == 4);
static_assert(hsc::ct::shape_bits<3, hsc::level_dq(1)>() == 6);
static_assert(hsc::ct::shape_bits<4, hsc::level_dq(1)>() == 0);

namespace
{

// the closed form from rate.hpp's header comment, written out independently of rate_info so the
// two derivations have to agree rather than share a bug
usize
predicted_stream(const hsc::rate_info &ri, usize n_elems) noexcept
{
  const u64 nb = hsc::__nblocks(ri.m, ri.dim_log2, n_elems);
  const u64 bits = nb * static_cast<u64>(ri.record_bits);
  return hsc::k_header_size + static_cast<usize>((bits + 7) / 8) + hsc::k_trailer_size;
}

}      // namespace

int
main()
{
  hsc::hopf_scratch sc;
  tutil::rng g;

  sb::test_case("rate: bin lane stream size is exactly predicted, every dim x level x gain_bits");
  {
    micron::vector<u8> src;
    src.reserve(4099);
    for ( usize i = 0; i < 4093; ++i ) src.push_back(static_cast<u8>(g.next()));      // ragged tail on purpose
    micron::vector<u8> z;
    z.reserve(1 << 16);
    z.resize(1 << 16);
    for ( u32 dl : { 2u, 3u, 4u } ) {
      for ( i32 lvl = 1; lvl <= 16; ++lvl ) {
        for ( u32 gb : { 1u, 8u, 24u } ) {
          const hsc::hopf_opts o{ .level = lvl, .dim_log2 = dl, .gain_bits = gb };
          auto r = hsc::rate(o, sc);
          sb::require(r.is_first());
          const hsc::rate_info ri = r.cast<hsc::rate_info>();
          sb::require(ri.record_bits, ri.gain_bits + ri.shape_bits);
          sb::require(ri.gain_bits, gb);
          const usize got = hsc::hopf_into(hsc::bytes{ src.begin(), src.size() }, o, z.begin(), z.size(), sc);
          sb::require_greater(got, 0ull);
          sb::require(got, predicted_stream(ri, src.size()));
          sb::require(got, hsc::bound(src.size(), o, sc));              // the scratch bound is the same claim
          sb::require_greater(hsc::bound(src.size(), o) + 1, got);      // scratch-free envelope >= actual
        }
      }
    }
  }

  sb::test_case("rate: dim32/dim64 agree too (skeleton cost is why the grid above stops at dim16)");
  {
    micron::vector<u8> src;
    src.reserve(4099);
    for ( usize i = 0; i < 4096; ++i ) src.push_back(static_cast<u8>(g.next()));
    micron::vector<u8> z;
    z.reserve(1 << 16);
    z.resize(1 << 16);
    for ( u32 dl : { 5u, 6u } ) {
      for ( i32 lvl : { 1, 2, 3, 5, 7, 9 } ) {
        const hsc::hopf_opts o{ .level = lvl, .dim_log2 = dl };
        auto r = hsc::rate(o, sc);
        sb::require(r.is_first());
        const usize got = hsc::hopf_into(hsc::bytes{ src.begin(), src.size() }, o, z.begin(), z.size(), sc);
        sb::require_greater(got, 0ull);
        sb::require(got, predicted_stream(r.cast<hsc::rate_info>(), src.size()));
      }
    }
  }

  sb::test_case("rate: float lanes -- vec, unit and quotient sizes are exactly predicted");
  {
    const usize nf = 2048;
    micron::vector<f32> src;
    src.reserve(nf + 1);
    for ( usize i = 0; i < nf; ++i ) src.push_back(static_cast<f32>(g.unit()) + 1.5f);
    micron::vector<u8> z;
    z.reserve(1 << 16);
    z.resize(1 << 16);
    for ( i32 lvl : { 1, 3, 6, 9, 12, 14 } ) {
      for ( u32 dl : { 2u, 3u, 4u } ) {
        for ( hsc::mode m : { hsc::mode::vec, hsc::mode::unit } ) {
          const hsc::hopf_opts o{ .m = m, .level = lvl, .dim_log2 = dl };
          auto r = hsc::rate(o, sc);
          sb::require(r.is_first());
          const hsc::rate_info ri = r.cast<hsc::rate_info>();
          sb::require(ri.gain_bits, m == hsc::mode::vec ? 8u : 0u);      // unit drops the gain field
          const max_t got = hsc::hopf_into(hsc::floats{ src.begin(), nf }, o, z.begin(), z.size(), sc);
          sb::require_greater(got, static_cast<max_t>(0));
          sb::require(static_cast<usize>(got), predicted_stream(ri, nf));
        }
      }
      const hsc::hopf_opts oq{ .m = hsc::mode::quotient, .level = lvl };
      auto rq = hsc::rate(oq, sc);
      sb::require(rq.is_first());
      const hsc::rate_info riq = rq.cast<hsc::rate_info>();
      sb::require(riq.block_elems, 4u);
      sb::require(riq.gain_bits, 0u);
      const max_t gq = hsc::hopf_into(hsc::floats{ src.begin(), nf }, oq, z.begin(), z.size(), sc);
      sb::require_greater(gq, static_cast<max_t>(0));
      sb::require(static_cast<usize>(gq), predicted_stream(riq, nf));
      for ( hsc::mode fm : { hsc::mode::quat, hsc::mode::oct } ) {
        // L14 oct wants a multi-million-row scratch attempt; the identity is pinned through L12
        if ( fm == hsc::mode::oct && lvl > 12 ) continue;
        const hsc::hopf_opts of{ .m = fm, .level = lvl };
        auto rf = hsc::rate(of, sc);
        sb::require(rf.is_first());
        const hsc::rate_info rif = rf.cast<hsc::rate_info>();
        sb::require(rif.block_elems, fm == hsc::mode::quat ? 8u : 16u);
        sb::require(rif.gain_bits, 0u);
        const max_t gf = hsc::hopf_into(hsc::floats{ src.begin(), nf }, of, z.begin(), z.size(), sc);
        sb::require_greater(gf, static_cast<max_t>(0));
        sb::require(static_cast<usize>(gf), predicted_stream(rif, nf));
        sb::require(static_cast<usize>(gf), hsc::bound(nf, of, sc));              // the scratch bound is the same claim
        sb::require_greater(hsc::bound(nf, of) + 1, static_cast<usize>(gf));      // scratch-free envelope >= actual
      }
    }
  }

  sb::test_case("rate: quotient really is cheaper than unit at the same d (the fiber saving)");
  {
    for ( i32 lvl : { 3, 5, 6, 7 } ) {
      const hsc::hopf_opts ou{ .m = hsc::mode::unit, .level = lvl, .dim_log2 = 2 };
      const hsc::hopf_opts oq{ .m = hsc::mode::quotient, .level = lvl };
      auto ru = hsc::rate(ou, sc);
      auto rq = hsc::rate(oq, sc);
      sb::require(ru.is_first() && rq.is_first());
      sb::require_greater(ru.cast<hsc::rate_info>().record_bits, rq.cast<hsc::rate_info>().record_bits);
    }
  }

  sb::test_case("rate: quat/oct are cheaper than unit d8/d16 at the same d (3 and 7 fiber dims)");
  {
    for ( i32 lvl : { 3, 5, 6, 7 } ) {
      const hsc::hopf_opts ou8{ .m = hsc::mode::unit, .level = lvl, .dim_log2 = 3 };
      const hsc::hopf_opts oq{ .m = hsc::mode::quat, .level = lvl };
      auto ru8 = hsc::rate(ou8, sc);
      auto rq = hsc::rate(oq, sc);
      sb::require(ru8.is_first() && rq.is_first());
      sb::require_greater(ru8.cast<hsc::rate_info>().record_bits, rq.cast<hsc::rate_info>().record_bits);
      const hsc::hopf_opts ou16{ .m = hsc::mode::unit, .level = lvl, .dim_log2 = 4 };
      const hsc::hopf_opts oo{ .m = hsc::mode::oct, .level = lvl };
      auto ru16 = hsc::rate(ou16, sc);
      auto ro = hsc::rate(oo, sc);
      sb::require(ru16.is_first() && ro.is_first());
      sb::require_greater(ru16.cast<hsc::rate_info>().record_bits, ro.cast<hsc::rate_info>().record_bits);
    }
    // the measured savings at L6: quat 22 - 15 = 7 bits/block, oct 37 - 24 = 13 bits/block
    auto q6 = hsc::rate(hsc::hopf_opts{ .m = hsc::mode::quat, .level = 6 }, sc);
    auto o6 = hsc::rate(hsc::hopf_opts{ .m = hsc::mode::oct, .level = 6 }, sc);
    sb::require(q6.cast<hsc::rate_info>().shape_bits, 15u);
    sb::require(o6.cast<hsc::rate_info>().shape_bits, 24u);
    // and the family can never go degenerate: even the d = 2 floor keeps the two pole codewords
    auto qd = hsc::rate(hsc::hopf_opts{ .m = hsc::mode::quat, .d = 2.0 }, sc);
    auto od = hsc::rate(hsc::hopf_opts{ .m = hsc::mode::oct, .d = 2.0 }, sc);
    sb::require(qd.cast<hsc::rate_info>().shape_bits, 1u);
    sb::require(od.cast<hsc::rate_info>().shape_bits, 1u);
    sb::require(!hsc::degenerate(hsc::hopf_opts{ .m = hsc::mode::quat, .d = 2.0 }, sc));
    sb::require(!hsc::degenerate(hsc::hopf_opts{ .m = hsc::mode::oct, .d = 2.0 }, sc));
  }

  sb::test_case("rate: ratio_of matches the stream a caller actually gets, frame included");
  {
    micron::vector<u8> src;
    src.reserve(8192);
    for ( usize i = 0; i < 8192; ++i ) src.push_back(static_cast<u8>(g.next()));
    micron::vector<u8> z;
    z.reserve(1 << 16);
    z.resize(1 << 16);
    for ( u32 dl : { 2u, 3u, 4u, 5u } ) {
      for ( i32 lvl : { 2, 6, 10 } ) {
        const hsc::hopf_opts o{ .level = lvl, .dim_log2 = dl };
        const usize got = hsc::hopf_into(hsc::bytes{ src.begin(), src.size() }, o, z.begin(), z.size(), sc);
        const f64 want = static_cast<f64>(src.size()) / static_cast<f64>(got);
        const f64 have = hsc::ratio_of(src.size(), o, sc);
        sb::require(have > want - 1e-12 && have < want + 1e-12);
        // the container-free asymptote is the ceiling the 48-byte frame keeps you under
        sb::require_greater(hsc::rate(o, sc).cast<hsc::rate_info>().ratio + 1e-12, have);
      }
    }
  }

  sb::test_case("rate: degenerate() flags exactly the cells whose shape field vanished");
  {
    // level 1 (d = 0.9) already collapses at dim >= 16; the whole coarse region below it does too
    sb::require_false(hsc::degenerate(hsc::hopf_opts{ .level = 1, .dim_log2 = 2 }, sc));
    sb::require_false(hsc::degenerate(hsc::hopf_opts{ .level = 1, .dim_log2 = 3 }, sc));
    sb::require_true(hsc::degenerate(hsc::hopf_opts{ .level = 1, .dim_log2 = 4 }, sc));
    sb::require_true(hsc::degenerate(hsc::hopf_opts{ .level = 1, .dim_log2 = 5 }, sc));
    sb::require_true(hsc::degenerate(hsc::hopf_opts{ .level = 1, .dim_log2 = 6 }, sc));
    // an explicit d coarser than any preset: gain-only at every dim above the 4D base
    sb::require_true(hsc::degenerate(hsc::hopf_opts{ .d = 1.5, .dim_log2 = 3 }, sc));
    sb::require_true(hsc::degenerate(hsc::hopf_opts{ .d = 1.5, .dim_log2 = 6 }, sc));
    // and nothing in the useful region is flagged
    for ( u32 dl : { 2u, 3u, 4u, 5u, 6u } )
      for ( i32 lvl : { 2, 3, 6, 9 } ) sb::require_false(hsc::degenerate(hsc::hopf_opts{ .level = lvl, .dim_log2 = dl }, sc));
  }

  sb::test_case("rate: a degenerate stream still round-trips -- it is data loss, not corruption");
  {
    micron::vector<u8> src;
    src.reserve(1024);
    for ( usize i = 0; i < 1024; ++i ) src.push_back(static_cast<u8>(g.next()));
    micron::vector<u8> z;
    z.reserve(4096);
    z.resize(4096);
    const hsc::hopf_opts o{ .level = 1, .dim_log2 = 4 };
    auto ri = hsc::rate(o, sc);
    sb::require(ri.is_first());
    sb::require(ri.cast<hsc::rate_info>().shape_bits, 0u);
    sb::require(ri.cast<hsc::rate_info>().record_bits, 8u);      // gain only
    const usize got = hsc::hopf_into(hsc::bytes{ src.begin(), src.size() }, o, z.begin(), z.size(), sc);
    sb::require(got, predicted_stream(ri.cast<hsc::rate_info>(), src.size()));
    auto back = hsc::unhopf(hsc::bytes{ z.begin(), got }, sc);
    sb::require(back.is_first());
    sb::require(back.cast<hsc::fhsc>().size(), src.size());
  }

  sb::test_case("rate: the new fine levels are usable end to end, not just tabulated");
  {
    micron::vector<u8> src;
    src.reserve(2048);
    for ( usize i = 0; i < 2048; ++i ) src.push_back(static_cast<u8>(128 + (g.next() & 7u)));
    micron::vector<u8> z;
    z.reserve(1 << 15);
    z.resize(1 << 15);
    micron::vector<u8> out;
    out.reserve(4096);
    out.resize(4096);
    f64 prev = 1e300;
    for ( i32 lvl = 9; lvl <= 16; ++lvl ) {
      const hsc::hopf_opts o{ .level = lvl, .dim_log2 = 2 };
      const usize got = hsc::hopf_into(hsc::bytes{ src.begin(), src.size() }, o, z.begin(), z.size(), sc);
      sb::require_greater(got, 0ull);
      auto r = hsc::unhopf(hsc::bytes{ z.begin(), got }, hsc::wbytes{ out.begin(), out.size() }, sc);
      sb::require(r.is_first());
      f64 se = 0;
      for ( usize i = 0; i < src.size(); ++i ) {
        const f64 e = static_cast<f64>(out[i]) - static_cast<f64>(src[i]);
        se += e * e;
      }
      const f64 rmse = __builtin_sqrt(se / static_cast<f64>(src.size()));
      sb::require(rmse <= prev + 1e-9);      // a finer level never gets worse
      prev = rmse;
    }
  }

  sb::test_case("rate: hsc::opts_exact_bytes round-trips FULL-RANGE bytes bit-exactly");
  {
    // the case above walks the ladder on a 3-bit dynamic range, which is the easy direction. This is
    // the claim BENCHMARKS.md's psnr 99.99 column actually makes: at dim 4 / L13 mode::bin is exact
    // on arbitrary bytes, because the per-element error lands under half a count and the integers
    // round back to themselves. Not a tolerance -- every byte, or the test fails.
    micron::vector<u8> src;
    src.reserve(4096);
    for ( usize i = 0; i < 4093; ++i ) src.push_back(static_cast<u8>(g.next()));      // ragged on purpose
    micron::vector<u8> z;
    z.reserve(1 << 15);
    z.resize(1 << 15);
    micron::vector<u8> out;
    out.reserve(4096);
    out.resize(4096);

    const hsc::hopf_opts o = hsc::opts_exact_bytes;
    sb::require_true(hsc::exact_bytes(o));
    auto ri = hsc::rate(o, sc);
    sb::require(ri.is_first());
    sb::require(ri.cast<hsc::rate_info>().shape_bits, 35u);
    sb::require(ri.cast<hsc::rate_info>().record_bits, 43u);
    sb::require(ri.cast<hsc::rate_info>().bits_per_input_byte, 10.75);      // EXPANSION, 0.744x

    const usize got = hsc::hopf_into(hsc::bytes{ src.begin(), src.size() }, o, z.begin(), z.size(), sc);
    sb::require(got, predicted_stream(ri.cast<hsc::rate_info>(), src.size()));
    sb::require_greater(got, src.size());      // exactness costs rate: the stream is BIGGER
    auto r = hsc::unhopf(hsc::bytes{ z.begin(), got }, hsc::wbytes{ out.begin(), out.size() }, sc);
    sb::require(r.is_first());
    sb::require(r.cast<usize>(), src.size());
    usize bad = 0;
    for ( usize i = 0; i < src.size(); ++i )
      if ( out[i] != src[i] ) ++bad;
    sb::require(bad, 0ull);
  }

  sb::end_test_case();
  return 1;
}
