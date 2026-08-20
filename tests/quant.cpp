// The porcelain, held to the same standard as the plumbing. Three things are load-bearing here
// and none of them is a tolerance:
//
//   1. qplan::bytes is EXACT. The plan is a promise about the artifact's size made before the data
//      is touched; every cell of the shape x dim x level matrix must land on it to the byte, in
//      both the bare-stream and container lanes.
//   2. Row random access is BIT-IDENTICAL to the full decode. unhopf_range seeks to
//      block * record_bits and skips the payload crc; if the seek were off by a bit the result
//      would still look like plausible floats, so the test compares bit patterns, not values.
//   3. The preset ladder's bits_per_weight equals hsc::rate()'s accounting. Those constants are
//      documentation that ships as API, and this is what keeps them from becoming a comment that
//      lies -- the same discipline as the golden cardinalities in tests/rate.cpp.
//
// Also pinned: the honesty guards (degenerate cells, the unit lane on wide rows, quotient on a
// tensor), container framing rejection, and the padding path where cols is not a multiple of dim.

#include "../src/hsc/hsc.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>
#include <micron/vector/vector.hpp>

#include <snowball/snowball.hpp>

namespace
{

// a weight-matrix-ish tensor: centered, correlated down a row, with a couple of outlier rows so
// the per-chunk gain full-scale actually has something to localize
micron::vector<f32>
make_weights(usize rows, usize cols, tutil::rng &g)
{
  micron::vector<f32> w;
  w.reserve(rows * cols + 1);
  w.resize(rows * cols);
  for ( usize r = 0; r < rows; ++r ) {
    const f64 scale = (r % 97 == 0) ? 40.0 : 1.0;      // outlier rows
    f64 acc = 0;
    for ( usize c = 0; c < cols; ++c ) {
      acc = acc * 0.75 + g.unit() * 0.25;
      w[r * cols + c] = static_cast<f32>(acc * scale);
    }
  }
  return w;
}

hsc::tensor
view_of(const micron::vector<f32> &w, usize cols)
{
  return hsc::tensor::of(hsc::floats{ w.begin(), w.size() }, cols);
}

// bit-pattern equality: a decode that is one ulp off is a format bug, not a rounding difference
bool
same_bits(const f32 *a, const f32 *b, usize n)
{
  for ( usize i = 0; i < n; ++i )
    if ( hsc::__f2u(a[i]) != hsc::__f2u(b[i]) ) return false;
  return true;
}

};      // namespace

int
main()
{
  hsc::hopf_scratch sc;
  tutil::rng g;

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: the preset ladder's bits_per_weight is rate()'s accounting, exactly");
  {
    for ( usize i = 0; i < hsc::k_qpresets; ++i ) {
      const hsc::qpreset &q = hsc::q_presets[i];
      const hsc::hopf_opts o{ .m = hsc::mode::vec, .level = q.level, .dim_log2 = q.dim_log2, .gain_bits = q.gain_bits };
      const auto r = hsc::rate(o, sc);
      sb::require(r.is_first());
      const hsc::rate_info &ri = r.cast<hsc::rate_info>();
      sb::require(q.bits_per_weight == ri.bits_per_elem);
      sb::require(q.est_rel_rmse == hsc::est_rel_rmse(q.dim_log2, hsc::d_of(ri.dq)));
      sb::require_greater(ri.shape_bits, 0u);      // no preset may sit on a degenerate cell
    }
    // pick() hands back the best quality that fits, and nothing when even the cheapest does not
    sb::require(hsc::pick(hsc::q_min.bits_per_weight - 0.001) == nullptr);
    sb::require(hsc::pick(1e9)->name == hsc::q_finest.name);
    for ( usize i = 0; i < hsc::k_qpresets; ++i ) sb::require(hsc::pick(hsc::q_presets[i].bits_per_weight)->name == hsc::q_presets[i].name);
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: plan bytes == bytes actually emitted, bare stream, every shape x cell");
  {
    const usize shapes[][2] = { { 40, 64 }, { 40, 384 }, { 17, 100 }, { 33, 7 }, { 1, 128 }, { 128, 1 }, { 1, 1 }, { 5, 3 } };
    for ( const auto &s : shapes ) {
      micron::vector<f32> w = make_weights(s[0], s[1], g);
      const hsc::tensor t = view_of(w, s[1]);
      sb::require(t.rows, s[0]);
      sb::require(t.cols, s[1]);
      for ( u32 dl : { 2u, 3u, 4u } ) {
        for ( i32 lvl : { 3, 6, 8 } ) {
          hsc::target tg{};
          tg.level = lvl;
          tg.dim_log2 = dl;
          const auto pr = hsc::plan_for(t, tg, sc);
          if ( !pr.is_first() ) {
            sb::require(pr.cast<hsc::error>() == hsc::error::bad_opts);      // only degeneracy may refuse
            continue;
          }
          const hsc::qplan &p = pr.cast<hsc::qplan>();
          const auto zr = hsc::quantize(t, p, sc);
          sb::require(zr.is_first());
          const hsc::qstream &q = zr.cast<hsc::qstream>();
          const hsc::fhsc &z = q.z;
          sb::require(z.size(), p.bytes);
          sb::require(p.bytes, hsc::qbound(p));

          micron::vector<f32> back;
          back.reserve(t.elems() + 1);
          back.resize(t.elems());
          const auto dr = hsc::dequantize(hsc::bytes{ z.begin(), z.size() }, p, hsc::wfloats{ back.begin(), back.size() }, sc);
          sb::require(dr.is_first());
          sb::require(dr.cast<usize>(), t.elems());
          // a lossy codec still has to stay on the right side of the no-information line
          const hsc::qerror e = hsc::measure(t, hsc::floats{ back.begin(), back.size() });
          sb::require(e.elems, t.elems());
          sb::require(e.rel_rmse < 1.4142135623730951 * 1.05);
        }
      }
    }
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: row random access is BIT-IDENTICAL to the full decode (bare stream)");
  {
    for ( usize cols : { 64u, 384u, 100u, 7u } ) {
      const usize rows = 53;
      micron::vector<f32> w = make_weights(rows, cols, g);
      const hsc::tensor t = view_of(w, cols);
      for ( u32 dl : { 2u, 3u, 5u } ) {
        hsc::target tg{};
        tg.level = 6;
        tg.dim_log2 = dl;
        const auto pr = hsc::plan_for(t, tg, sc);
        sb::require(pr.is_first());
        const hsc::qplan &p = pr.cast<hsc::qplan>();
        const auto zr = hsc::quantize(t, p, sc);
        sb::require(zr.is_first());
        const hsc::bytes z = zr.cast<hsc::qstream>().view();

        micron::vector<f32> full;
        full.reserve(t.elems() + 1);
        full.resize(t.elems());
        sb::require(hsc::dequantize(z, p, hsc::wfloats{ full.begin(), full.size() }, sc).is_first());

        micron::vector<f32> part;
        part.reserve(rows * cols + 1);
        part.resize(rows * cols);
        for ( usize i = 0; i < rows; ++i ) {
          const auto rr = hsc::dequantize_row(z, p, i, hsc::wfloats{ part.begin(), cols }, sc);
          sb::require(rr.is_first());
          sb::require(rr.cast<usize>(), cols);
          sb::require(same_bits(part.begin(), full.begin() + i * cols, cols));
        }
        // multi-row spans, including the last row and a span of one
        for ( usize first : { usize(0), usize(1), usize(17), rows - 1 } ) {
          const usize count = first + 7 <= rows ? 7 : rows - first;
          const auto rr = hsc::dequantize_rows(z, p, first, count, hsc::wfloats{ part.begin(), count * cols }, sc);
          sb::require(rr.is_first());
          sb::require(same_bits(part.begin(), full.begin() + first * cols, count * cols));
        }
        // out of range is refused, not clamped -- and the check does not wrap on a huge first
        sb::require(!hsc::dequantize_rows(z, p, rows, 1, hsc::wfloats{ part.begin(), cols }, sc).is_first());
        sb::require(!hsc::dequantize_rows(z, p, 0, rows + 1, hsc::wfloats{ part.begin(), cols }, sc).is_first());
        sb::require(!hsc::dequantize_rows(z, p, ~usize(0), 1, hsc::wfloats{ part.begin(), cols }, sc).is_first());
        sb::require(!hsc::dequantize_rows(z, p, 1, ~usize(0), hsc::wfloats{ part.begin(), cols }, sc).is_first());
      }
    }
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: unhopf_range == unhopf over arbitrary block windows, every lane");
  {
    const usize n = 4096;
    micron::vector<f32> src;
    src.reserve(n + 1);
    src.resize(n);
    for ( usize i = 0; i < n; ++i ) src[i] = static_cast<f32>(g.unit());
    micron::vector<u8> bsrc;
    bsrc.reserve(n + 1);
    for ( usize i = 0; i < n; ++i ) bsrc.push_back(static_cast<u8>(g.next()));

    for ( u32 dl : { 2u, 3u, 4u } ) {
      const u32 dim = 1u << dl;
      // unit needs whole blocks and a nonzero norm per block; normalize a copy
      micron::vector<f32> usrc = src;
      for ( usize b = 0; b + dim <= n; b += dim ) {
        f64 s2 = 0;
        for ( u32 c = 0; c < dim; ++c ) s2 += static_cast<f64>(usrc[b + c]) * static_cast<f64>(usrc[b + c]);
        if ( !(s2 > 0) ) {
          usrc[b] = 1.0f;
          continue;
        }
        const f64 inv = 1.0 / __builtin_sqrt(s2);
        for ( u32 c = 0; c < dim; ++c ) usrc[b + c] = static_cast<f32>(static_cast<f64>(usrc[b + c]) * inv);
      }

      for ( hsc::mode m : { hsc::mode::bin, hsc::mode::vec, hsc::mode::unit, hsc::mode::quotient, hsc::mode::quat, hsc::mode::oct } ) {
        if ( m == hsc::mode::quotient && dl != 2 ) continue;      // the fibrations pin the width
        if ( m == hsc::mode::quat && dl != 3 ) continue;
        if ( m == hsc::mode::oct && dl != 4 ) continue;
        const hsc::hopf_opts o{ .m = m, .level = 6, .dim_log2 = dl };
        micron::vector<u8> z;
        z.reserve(1 << 20);
        z.resize(1 << 20);
        max_t zn = 0;
        if ( m == hsc::mode::bin )
          zn = static_cast<max_t>(hsc::hopf_into(hsc::bytes{ bsrc.begin(), bsrc.size() }, o, z.begin(), z.size(), sc));
        else
          zn = hsc::hopf_into(hsc::floats{ m == hsc::mode::vec ? src.begin() : usrc.begin(), n }, o, z.begin(), z.size(), sc);
        sb::require_greater(zn, static_cast<max_t>(0));
        const hsc::bytes zs{ z.begin(), static_cast<usize>(zn) };

        const auto vr = hsc::verify(zs);
        sb::require(vr.is_first());
        const hsc::hopf_info fi = vr.cast<hsc::hopf_info>();
        const u64 be = hsc::__block_elems(fi.m, fi.dim_log2);

        micron::vector<u8> full;
        full.reserve(n * 4 + 1);
        full.resize(n * 4);
        sb::require(hsc::unhopf(zs, hsc::wbytes{ full.begin(), full.size() }, sc).is_first());

        micron::vector<u8> part;
        part.reserve(n * 4 + 1);
        part.resize(n * 4);
        const u64 firsts[] = { 0ull, 1ull, 3ull, fi.nblocks / 2, fi.nblocks - 1 };
        for ( u64 first : firsts ) {
          const u64 count = first + 5 <= fi.nblocks ? 5 : fi.nblocks - first;
          const auto rr = hsc::unhopf_range(zs, first, count, hsc::wbytes{ part.begin(), part.size() }, sc);
          sb::require(rr.is_first());
          const usize elems = rr.cast<usize>();
          const usize esz = fi.m == hsc::mode::bin ? 1 : 4;
          sb::require(elems, static_cast<usize>(hsc::__hopf::__range_elems(fi, first, count)));
          for ( usize k = 0; k < elems * esz; ++k ) sb::require(part[k], full[static_cast<usize>(first * be) * esz + k]);
        }
        // a window past the end is a length error
        sb::require(!hsc::unhopf_range(zs, fi.nblocks, 1, hsc::wbytes{ part.begin(), part.size() }, sc).is_first());
        // the f32 lane refuses a byte-mode stream, as unhopf does
        if ( m == hsc::mode::bin )
          sb::require(!hsc::unhopf_range(zs, 0, 1, hsc::wfloats{ reinterpret_cast<f32 *>(part.begin()), n }, sc).is_first());
      }
    }
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: the HSCQ container round-trips, and its rows match the whole decode");
  {
    const usize rows = 300, cols = 96;
    micron::vector<f32> w = make_weights(rows, cols, g);
    const hsc::tensor t = view_of(w, cols);
    for ( usize cr : { usize(0), usize(1), usize(7), usize(300) } ) {
      hsc::target tg{};
      tg.level = 6;
      tg.dim_log2 = 3;
      tg.chunked = true;
      tg.chunk_rows = cr;
      const auto pr = hsc::plan_for(t, tg, sc);
      sb::require(pr.is_first());
      const hsc::qplan &p = pr.cast<hsc::qplan>();
      sb::require(p.chunks, (rows + p.chunk_rows - 1) / p.chunk_rows);

      const auto pk = hsc::pack(t, p, sc);
      sb::require(pk.is_first());
      const hsc::bytes blob{ pk.cast<hsc::fhsc>().begin(), pk.cast<hsc::fhsc>().size() };
      sb::require(blob.size(), p.bytes);

      const auto qi = hsc::qprobe(blob);
      sb::require(qi.is_first());
      const hsc::qinfo &info = qi.cast<hsc::qinfo>();
      sb::require(info.rows, rows);
      sb::require(info.cols, cols);
      sb::require(info.chunks, p.chunks);
      sb::require(info.chunk_rows, p.chunk_rows);
      sb::require(info.p.record_bits, p.record_bits);
      sb::require(info.p.bytes, p.bytes);
      sb::require(info.p.dq, p.dq);

      micron::vector<f32> full;
      full.reserve(t.elems() + 1);
      full.resize(t.elems());
      sb::require(hsc::unpack(blob, hsc::wfloats{ full.begin(), full.size() }, sc).is_first());

      // every chunk is an ordinary hsc stream: the plumbing reads it with no help from us
      for ( usize c = 0; c < p.chunks; ++c ) {
        const hsc::bytes cs = hsc::__quant::chunk_at(blob, p, c);
        sb::require(hsc::verify(cs).is_first());
        sb::require(hsc::hopf_probe(cs).is_first());
      }

      micron::vector<f32> one;
      one.reserve(cols * 9 + 1);
      one.resize(cols * 9);
      hsc::qreader rd{ blob, sc };
      sb::require(rd.status() == hsc::error::ok);
      for ( usize i = 0; i < rows; i += 11 ) {
        sb::require(rd.row(i, hsc::wfloats{ one.begin(), cols }).is_first());
        sb::require(same_bits(one.begin(), full.begin() + i * cols, cols));
      }
      // a span that crosses chunk boundaries stitches correctly
      for ( usize first : { usize(0), p.chunk_rows - 1, rows - 9 } ) {
        const usize count = first + 9 <= rows ? 9 : rows - first;
        sb::require(rd.rows_into(first, count, hsc::wfloats{ one.begin(), count * cols }).is_first());
        sb::require(same_bits(one.begin(), full.begin() + first * cols, count * cols));
        sb::require(hsc::unpack_rows(blob, first, count, hsc::wfloats{ one.begin(), count * cols }, sc).is_first());
        sb::require(same_bits(one.begin(), full.begin() + first * cols, count * cols));
      }
    }
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: chunking localizes the gain full-scale, and gain_range() predicts when");
  {
    // ONE full-scale covers a whole stream and the gain is quantized uniformly against it, so a
    // block below full_scale / 2^gain_bits reconstructs to exactly zero. Past that dynamic range
    // a single stream does not merely lose accuracy on the quiet rows, it deletes them.
    const usize rows = 400, cols = 64, loud = 40;
    tutil::rng gl;      // a fresh stream: this case's data must not depend on what ran before it
    micron::vector<f32> w = make_weights(rows, cols, gl);
    for ( usize r = 0; r < loud; ++r )
      for ( usize c = 0; c < cols; ++c ) w[r * cols + c] *= 500.0f;
    const hsc::tensor t = view_of(w, cols);
    hsc::target tg{};
    tg.level = 6;
    tg.dim_log2 = 3;

    const hsc::grange gr = hsc::gain_range(t, 3, 8);
    sb::require_greater(gr.ratio, 256.0);      // past what 8 gain bits can resolve
    sb::require_greater(gr.zeroed, 0ull);      // and it says so, before anything is encoded
    sb::require(gr.blocks, (rows * cols) / 8);
    sb::require(gr.empty, 0ull);
    sb::require_greater(gr.max_norm, gr.min_norm);

    const auto p1 = hsc::plan_for(t, tg, sc);
    sb::require(p1.is_first());
    const auto z1 = hsc::quantize(t, p1.cast<hsc::qplan>(), sc);
    sb::require(z1.is_first());
    micron::vector<f32> b1;
    b1.reserve(t.elems() + 1);
    b1.resize(t.elems());
    sb::require(hsc::dequantize(z1.cast<hsc::qstream>(), hsc::wfloats{ b1.begin(), b1.size() }, sc).is_first());

    hsc::target tg2 = tg;
    tg2.chunked = true;
    tg2.chunk_rows = 16;      // small enough that most chunks hold no loud row
    const auto p2 = hsc::plan_for(t, tg2, sc);
    sb::require(p2.is_first());
    const auto z2 = hsc::pack(t, p2.cast<hsc::qplan>(), sc);
    sb::require(z2.is_first());
    micron::vector<f32> b2;
    b2.reserve(t.elems() + 1);
    b2.resize(t.elems());
    sb::require(
        hsc::unpack(hsc::bytes{ z2.cast<hsc::fhsc>().begin(), z2.cast<hsc::fhsc>().size() }, hsc::wfloats{ b2.begin(), b2.size() }, sc)
            .is_first());

    // measured on the QUIET rows: the whole-tensor figure is dominated by the loud ones, which
    // quantize fine either way, and would hide the entire effect
    const usize q0 = loud * cols;
    const hsc::qerror e1 = hsc::measure(hsc::floats{ w.begin() + q0, t.elems() - q0 }, hsc::floats{ b1.begin() + q0, t.elems() - q0 });
    const hsc::qerror e2 = hsc::measure(hsc::floats{ w.begin() + q0, t.elems() - q0 }, hsc::floats{ b2.begin() + q0, t.elems() - q0 });
    sb::require_greater(e1.rel_rmse, 0.9);      // the single stream zeroed them
    sb::require(e2.rel_rmse < 0.5);             // the container kept them
    sb::require_greater(e1.rel_rmse, e2.rel_rmse * 2.0);

    // below the cliff the two lanes agree to about a percent, and the bare stream is smaller.
    // built with tightly clustered block norms on purpose -- make_weights() has loud rows in it,
    // and a random walk throws the odd near-zero block, either of which puts the ratio over 256
    micron::vector<f32> q;
    q.reserve(rows * cols + 1);
    q.resize(rows * cols);
    for ( usize i = 0; i < rows * cols; ++i ) q[i] = static_cast<f32>(1.0 + 0.5 * gl.unit());
    const hsc::tensor tq = view_of(q, cols);
    const hsc::grange qr = hsc::gain_range(tq, 3, 8);
    sb::require(qr.ratio < 256.0);
    sb::require(qr.zeroed, 0ull);
    const auto q1 = hsc::quantize(tq, hsc::plan_for(tq, tg, sc).cast<hsc::qplan>(), sc);
    const auto q2 = hsc::pack(tq, hsc::plan_for(tq, tg2, sc).cast<hsc::qplan>(), sc);
    sb::require(q1.is_first());
    sb::require(q2.is_first());
    sb::require_greater(q2.cast<hsc::fhsc>().size(), q1.cast<hsc::qstream>().size());
    micron::vector<f32> c1, c2;
    c1.reserve(tq.elems() + 1);
    c1.resize(tq.elems());
    c2.reserve(tq.elems() + 1);
    c2.resize(tq.elems());
    sb::require(hsc::dequantize(q1.cast<hsc::qstream>(), hsc::wfloats{ c1.begin(), c1.size() }, sc).is_first());
    sb::require(
        hsc::unpack(hsc::bytes{ q2.cast<hsc::fhsc>().begin(), q2.cast<hsc::fhsc>().size() }, hsc::wfloats{ c2.begin(), c2.size() }, sc)
            .is_first());
    const f64 r1 = hsc::measure(tq, hsc::floats{ c1.begin(), c1.size() }).rel_rmse;
    const f64 r2 = hsc::measure(tq, hsc::floats{ c2.begin(), c2.size() }).rel_rmse;
    sb::require(r1 < r2 * 1.1);
    sb::require(r2 < r1 * 1.1);

    // an all-zero tensor has no range to speak of and must not divide by anything
    micron::vector<f32> z;
    z.reserve(64 * 8 + 1);
    z.resize(64 * 8);
    const hsc::grange zr = hsc::gain_range(view_of(z, 8), 3, 8);
    sb::require(zr.ratio == 0.0);
    sb::require(zr.empty, zr.blocks);
    sb::require(zr.zeroed, 0ull);
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: honesty guards -- degeneracy, the unit lane, the quotient lane");
  {
    micron::vector<f32> w = make_weights(8, 64, g);
    const hsc::tensor t = view_of(w, 64);

    // level 1 at dim 16+ collapses to M == 1: the shape field vanishes and only the gain is left
    hsc::target deg{};
    deg.level = 1;
    deg.dim_log2 = 4;
    sb::require(!hsc::plan_for(t, deg, sc).is_first());
    deg.allow_degenerate = true;
    const auto dr = hsc::plan_for(t, deg, sc);
    sb::require(dr.is_first());
    sb::require(dr.cast<hsc::qplan>().degenerate);
    sb::require(dr.cast<hsc::qplan>().shape_bits, 0u);
    // ... and a degenerate plan still cannot be used to encode: the ratio is not a rate
    micron::vector<u8> buf;
    buf.reserve(1 << 16);
    buf.resize(1 << 16);
    sb::require(!hsc::quantize_into(t, dr.cast<hsc::qplan>(), hsc::wbytes{ buf.begin(), buf.size() }, sc).is_first());

    // unit keeps no per-block magnitude, so it is refused unless a row IS one block
    hsc::target u{};
    u.m = hsc::mode::unit;
    sb::require(!hsc::plan_for(t, u, sc).is_first());
    const hsc::tensor t8 = view_of(w, 8);
    u.min_dim_log2 = 2;
    u.max_dim_log2 = 6;
    const auto ur = hsc::plan_for(t8, u, sc);
    sb::require(ur.is_first());
    sb::require(ur.cast<hsc::qplan>().dim_log2, 3u);                                       // cols == 8 == dim
    sb::require(ur.cast<hsc::qplan>().record_bits, ur.cast<hsc::qplan>().shape_bits);      // no gain field

    // the quotient family has no fiber symmetry to quotient in a weight matrix (a quat/oct
    // plan would delete 3/7 real dimensions per pair), and bin is not this lane
    for ( hsc::mode fm : { hsc::mode::quotient, hsc::mode::quat, hsc::mode::oct, hsc::mode::bin } ) {
      hsc::target q{};
      q.m = fm;
      sb::require(!hsc::plan_for(t, q, sc).is_first());
    }

    // the two lanes do not accept each other's plans: quantize() means a bare stream, pack() a
    // container, and the plan says which
    hsc::target ch{};
    ch.level = 6;
    ch.dim_log2 = 3;
    ch.chunked = true;
    const auto cp = hsc::plan_for(t, ch, sc);
    sb::require(cp.is_first());
    sb::require(!hsc::quantize(t, cp.cast<hsc::qplan>(), sc).is_first());
    sb::require(!hsc::quantize_into(t, cp.cast<hsc::qplan>(), hsc::wbytes{ buf.begin(), buf.size() }, sc).is_first());
    hsc::target bare = ch;
    bare.chunked = false;
    const auto bp = hsc::plan_for(t, bare, sc);
    sb::require(bp.is_first());
    sb::require(!hsc::pack(t, bp.cast<hsc::qplan>(), sc).is_first());
    sb::require(!hsc::pack_into(t, bp.cast<hsc::qplan>(), hsc::wbytes{ buf.begin(), buf.size() }, sc).is_first());
    // and the qstream lane round-trips through the plan it carries
    const auto qs = hsc::quantize(t, bare, sc);
    sb::require(qs.is_first());
    micron::vector<f32> qb;
    qb.reserve(t.elems() + 1);
    qb.resize(t.elems());
    sb::require(hsc::dequantize(qs.cast<hsc::qstream>(), hsc::wfloats{ qb.begin(), qb.size() }, sc).is_first());
    sb::require(hsc::dequantize_row(qs.cast<hsc::qstream>(), 3, hsc::wfloats{ qb.begin(), 64 }, sc).is_first());

    // a chunk boundary must fall on a block boundary
    hsc::target fc{};
    fc.chunked = true;
    fc.row_aligned = false;
    sb::require(!hsc::plan_for(t, fc, sc).is_first());

    // empty shapes and a nonsense gain width
    sb::require(!hsc::plan_for(0, 64, hsc::target{}, sc).is_first());
    sb::require(!hsc::plan_for(64, 0, hsc::target{}, sc).is_first());
    hsc::target gb{};
    gb.gain_bits = 25;
    sb::require(!hsc::plan_for(t, gb, sc).is_first());
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: targets -- a bits budget is never exceeded, a quality goal is met");
  {
    micron::vector<f32> w = make_weights(64, 256, g);
    const hsc::tensor t = view_of(w, 256);
    for ( f64 budget : { 1.5, 2.0, 3.0, 4.0, 6.0 } ) {
      hsc::target tg{};
      tg.bits_per_weight = budget;
      const auto pr = hsc::plan_for(t, tg, sc);
      sb::require(pr.is_first());
      const hsc::qplan &p = pr.cast<hsc::qplan>();
      sb::require(p.bits_per_weight <= budget);
      sb::require(!p.degenerate);
      const auto zr = hsc::quantize(t, p, sc);
      sb::require(zr.is_first());
      sb::require(zr.cast<hsc::qstream>().size(), p.bytes);
      // the artifact really is inside the budget, measured off the bytes on the wire
      sb::require(static_cast<f64>(zr.cast<hsc::qstream>().size()) * 8.0 / static_cast<f64>(t.elems()) <= budget);
    }
    for ( f64 goal : { 0.5, 0.25, 0.12 } ) {
      hsc::target tg{};
      tg.rel_rmse = goal;
      const auto pr = hsc::plan_for(t, tg, sc);
      sb::require(pr.is_first());
      sb::require(pr.cast<hsc::qplan>().est_rel_rmse <= goal);
    }
    // a budget nothing can meet is refused rather than silently overspent
    hsc::target imp{};
    imp.bits_per_weight = 0.01;
    sb::require(!hsc::plan_for(t, imp, sc).is_first());
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: the default plan follows the shape, and padding is paid for in the rate");
  {
    // 384 is a multiple of 32, so the default is q_balanced untouched
    micron::vector<f32> w = make_weights(8, 384, g);
    const auto pr = hsc::plan_for(view_of(w, 384), hsc::target{}, sc);
    sb::require(pr.is_first());
    sb::require(pr.cast<hsc::qplan>().dim_log2, hsc::q_balanced.dim_log2);
    sb::require(pr.cast<hsc::qplan>().level, hsc::q_balanced.level);
    sb::require(pr.cast<hsc::qplan>().padded_cols, 384ull);

    // 300 is a multiple of 4 alone, so the ladder walks down to a dim that divides it
    micron::vector<f32> w2 = make_weights(8, 300, g);
    const auto p2 = hsc::plan_for(view_of(w2, 300), hsc::target{}, sc);
    sb::require(p2.is_first());
    sb::require(300u % (1u << p2.cast<hsc::qplan>().dim_log2), 0u);
    sb::require(p2.cast<hsc::qplan>().padded_cols, 300ull);

    // 100 with dim 8 pads each row to 104: the waste shows up in the rate, it is not hidden
    hsc::target tg{};
    tg.level = 6;
    tg.dim_log2 = 3;
    micron::vector<f32> w3 = make_weights(9, 100, g);
    const hsc::tensor t3 = view_of(w3, 100);
    const auto p3 = hsc::plan_for(t3, tg, sc);
    sb::require(p3.is_first());
    const hsc::qplan &pp = p3.cast<hsc::qplan>();
    sb::require(pp.blocks_per_row, 13ull);
    sb::require(pp.padded_cols, 104ull);
    sb::require(pp.blocks, 9ull * 13ull);
    sb::require(pp.payload_bits_per_weight > static_cast<f64>(pp.record_bits) / 8.0);      // strictly above the unpadded rate
    const auto z3 = hsc::quantize(t3, pp, sc);
    sb::require(z3.is_first());
    sb::require(z3.cast<hsc::qstream>().size(), pp.bytes);
    micron::vector<f32> b3;
    b3.reserve(t3.elems() + 1);
    b3.resize(t3.elems());
    sb::require(hsc::dequantize(z3.cast<hsc::qstream>().view(), pp, hsc::wfloats{ b3.begin(), b3.size() }, sc).is_first());
    // the pad slots must not leak into the decoded rows
    const hsc::qerror e = hsc::measure(t3, hsc::floats{ b3.begin(), b3.size() });
    sb::require(e.rel_rmse < 1.0);

    // flat blocking wastes nothing and refuses row access
    hsc::target flat = tg;
    flat.row_aligned = false;
    const auto p4 = hsc::plan_for(t3, flat, sc);
    sb::require(p4.is_first());
    sb::require(p4.cast<hsc::qplan>().padded_cols, 100ull);
    sb::require(p4.cast<hsc::qplan>().blocks, (9ull * 100ull + 7ull) / 8ull);
    sb::require_greater(pp.bytes + 1, p4.cast<hsc::qplan>().bytes);      // flat is never larger
    const auto z4 = hsc::quantize(t3, p4.cast<hsc::qplan>(), sc);
    sb::require(z4.is_first());
    sb::require(
        !hsc::dequantize_row(z4.cast<hsc::qstream>().view(), p4.cast<hsc::qplan>(), 0, hsc::wfloats{ b3.begin(), 100 }, sc).is_first());
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: container framing is validated, never trusted");
  {
    micron::vector<f32> w = make_weights(40, 64, g);
    const hsc::tensor t = view_of(w, 64);
    hsc::target tg{};
    tg.level = 6;
    tg.dim_log2 = 3;
    tg.chunked = true;
    tg.chunk_rows = 8;
    const auto pr = hsc::plan_for(t, tg, sc);
    sb::require(pr.is_first());
    const auto pk = hsc::pack(t, pr.cast<hsc::qplan>(), sc);
    sb::require(pk.is_first());
    micron::vector<u8> blob;
    blob.reserve(pk.cast<hsc::fhsc>().size() + 1);
    for ( usize i = 0; i < pk.cast<hsc::fhsc>().size(); ++i ) blob.push_back(pk.cast<hsc::fhsc>().begin()[i]);
    const hsc::bytes good{ blob.begin(), blob.size() };
    sb::require(hsc::qprobe(good).is_first());

    const auto corrupt = [&](usize at, u8 x) {
      micron::vector<u8> c = blob;
      c[at] ^= x;
      return hsc::qprobe(hsc::bytes{ c.begin(), c.size() });
    };
    sb::require(corrupt(0, 0xFF).cast<hsc::error>() == hsc::error::bad_container);       // magic
    sb::require(corrupt(4, 0xFF).cast<hsc::error>() == hsc::error::unsupported);         // version
    sb::require(corrupt(8, 0x01).cast<hsc::error>() == hsc::error::bad_container);       // rows, hc catches it
    sb::require(corrupt(24, 0x01).cast<hsc::error>() == hsc::error::bad_container);      // chunk_rows, hc catches it
    sb::require(corrupt(31, 0xFF).cast<hsc::error>() == hsc::error::bad_container);      // the hc byte itself
    sb::require(corrupt(28, 0x01).cast<hsc::error>() == hsc::error::bad_container);      // reserved must stay zero
    // truncation, at both ends of the head
    for ( usize n : { usize(0), usize(8), usize(31), usize(32), hsc::k_qhead_size + 40 } )
      sb::require(!hsc::qprobe(hsc::bytes{ blob.begin(), n }).is_first());
    sb::require(!hsc::qprobe(hsc::bytes{ blob.begin(), blob.size() - 1 }).is_first());

    // a flipped payload byte survives qprobe (which never reads it) and dies in verify/unpack
    micron::vector<u8> c = blob;
    c[hsc::k_qhead_size + hsc::k_header_size + 3] ^= 0x40;
    const hsc::bytes bad{ c.begin(), c.size() };
    sb::require(hsc::qprobe(bad).is_first());
    sb::require(hsc::verify(hsc::__quant::chunk_at(bad, pr.cast<hsc::qplan>(), 0)).cast<hsc::error>() == hsc::error::bad_checksum);
    micron::vector<f32> out;
    out.reserve(t.elems() + 1);
    out.resize(t.elems());
    sb::require(!hsc::unpack(bad, hsc::wfloats{ out.begin(), out.size() }, sc).is_first());
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  sb::test_case("quant: measure() agrees with itself, and the fitted law is in the right ballpark");
  {
    micron::vector<f32> w = make_weights(64, 256, g);
    const hsc::tensor t = view_of(w, 256);
    const hsc::qerror id = hsc::measure(t, t.v);
    sb::require(id.rmse == 0.0);
    sb::require(id.rel_rmse == 0.0);
    sb::require(id.cos > 0.999999);
    sb::require(id.elems, t.elems());

    // gaussian-ish centered data is exactly the domain the law was fitted on: within a factor of
    // two either way, which is the honest claim (per-corpus medians are 5-14% on the non-flat
    // corpora, but this tensor is one draw, not a corpus)
    for ( u32 dl : { 3u, 5u } ) {
      hsc::target tg{};
      tg.level = 6;
      tg.dim_log2 = dl;
      const auto pr = hsc::plan_for(t, tg, sc);
      sb::require(pr.is_first());
      const hsc::qplan &p = pr.cast<hsc::qplan>();
      const auto zr = hsc::quantize(t, p, sc);
      sb::require(zr.is_first());
      micron::vector<f32> back;
      back.reserve(t.elems() + 1);
      back.resize(t.elems());
      sb::require(hsc::dequantize(zr.cast<hsc::qstream>().view(), p, hsc::wfloats{ back.begin(), back.size() }, sc).is_first());
      const hsc::qerror e = hsc::measure(t, hsc::floats{ back.begin(), back.size() });
      sb::require(e.rel_rmse < p.est_rel_rmse * 2.0);
      sb::require(e.rel_rmse > p.est_rel_rmse * 0.5);
      sb::require(e.psnr_db > 0.0);
    }
  }

  sb::end_test_case();
  return 1;
}
