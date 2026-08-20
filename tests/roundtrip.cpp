// End-to-end container round-trips for every mode across dims and levels: exact deterministic
// stream sizes, tail handling, distortion envelopes (statistical -- see block.hpp on packing
// vs covering), bound() >= actual (and scratch-bound == actual), the short-output/short-cap
// contracts, typed-lane mode policing, scratch reuse across parameter changes, and the
// transform lane (bit2 round trips at unchanged rate, the spiky-bytes win, quotient forced off).

#include "../src/hsc/hsc.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>
#include <micron/vector/vector.hpp>

#include <snowball/snowball.hpp>

namespace
{

micron::vector<u8>
noise_bytes(usize n, tutil::rng &g)
{
  micron::vector<u8> v;
  v.reserve(n + 1);
  for ( usize i = 0; i < n; ++i ) v.push_back(static_cast<u8>(g.next()));
  return v;
}

micron::vector<u8>
smooth_bytes(usize n, tutil::rng &g)
{
  micron::vector<u8> v;
  v.reserve(n + 1);
  f64 x = 128.0;
  for ( usize i = 0; i < n; ++i ) {
    x += g.unit() * 9.0;
    x = x < 0 ? 0 : (x > 255 ? 255 : x);
    v.push_back(static_cast<u8>(x));
  }
  return v;
}

// the transform's target workload: one full-scale excursion per 8 bytes on a 0x80 floor --
// centered, (almost) axis-aligned spikes, the packing-not-covering worst case
micron::vector<u8>
spiky_bytes(usize n, tutil::rng &g)
{
  micron::vector<u8> v;
  v.reserve(n + 1);
  for ( usize i = 0; i < n; ++i ) v.push_back(u8(0x80));
  for ( usize b = 0; b + 8 <= n; b += 8 ) {
    const u64 r = g.next();
    v[b + (r & 7u)] = (r >> 3) & 1u ? u8(0xFF) : u8(0x00);
  }
  return v;
}

}      // namespace

int
main()
{
  hsc::hopf_scratch sc;
  tutil::rng g;

  sb::test_case("bin: round trip at every dim x level, exact stream size, tail preserved");
  {
    for ( u32 dl = 2; dl <= 6; ++dl ) {
      for ( i32 lvl : { 3, 6 } ) {
        const usize n_in = 2048 + dl;      // deliberately not block-aligned
        const auto src = noise_bytes(n_in, g);
        const hsc::hopf_opts o{ .level = lvl, .dim_log2 = dl };
        hsc::fhsc z = hsc::hopf(tutil::view(src), o, sc);
        sb::require(z.size() > 0);
        sb::require(z.size() <= hsc::bound(n_in, o));
        sb::require(z.size(), hsc::bound(n_in, o, sc));      // scratch bound is exact

        auto back = hsc::unhopf(hsc::bytes{ z.first(), z.size() }, sc);
        sb::require(back.is_first());
        auto &b = back.cast<hsc::fhsc>();
        sb::require(b.size(), n_in);
      }
    }
  }

  sb::test_case("bin: distortion tracks the level (smooth data, per-byte RMSE envelope)");
  {
    const usize n_in = 8192;
    const auto src = smooth_bytes(n_in, g);
    f64 prev_rmse = 1e9;
    for ( i32 lvl : { 3, 5, 6, 7 } ) {
      const hsc::hopf_opts o{ .level = lvl, .dim_log2 = 3 };
      hsc::fhsc z = hsc::hopf(tutil::view(src), o, sc);
      auto back = hsc::unhopf(hsc::bytes{ z.first(), z.size() }, sc);
      sb::require(back.is_first());
      auto &b = back.cast<hsc::fhsc>();
      f64 se = 0;
      for ( usize i = 0; i < n_in; ++i ) {
        const f64 e = static_cast<f64>(b.first()[i]) - static_cast<f64>(src[i]);
        se += e * e;
      }
      const f64 rmse = __builtin_sqrt(se / static_cast<f64>(n_in));
      const f64 d = hsc::d_of(hsc::level_dq(lvl));
      sb::require(rmse <= 127.5 * 2.0 * d * 2.5);      // loose absolute envelope
      sb::require(rmse <= prev_rmse + 1e-9);           // finer level never gets worse
      prev_rmse = rmse;
    }
  }

  sb::test_case("transform: bin round trips at every dim x level, same exact size, bit2 on the wire");
  {
    for ( u32 dl = 2; dl <= 6; ++dl ) {
      for ( i32 lvl : { 3, 6 } ) {
        const usize n_in = 2048 + dl;
        const auto src = noise_bytes(n_in, g);
        const hsc::hopf_opts o{ .level = lvl, .dim_log2 = dl, .transform = true };
        hsc::fhsc z = hsc::hopf(tutil::view(src), o, sc);
        sb::require(z.size(), hsc::bound(n_in, o, sc));      // analog-side: rate is untouched
        auto pi = hsc::hopf_probe(hsc::bytes{ z.first(), z.size() });
        sb::require(pi.is_first());
        sb::require(pi.cast<hsc::hopf_info>().transform);
        auto back = hsc::unhopf(hsc::bytes{ z.first(), z.size() }, sc);
        sb::require(back.is_first());
        sb::require(back.cast<hsc::fhsc>().size(), n_in);
      }
    }
  }

  sb::test_case("transform: spiky bytes improve, on beats off (the case the feature exists for)");
  {
    const usize n_in = 8192;
    const auto src = spiky_bytes(n_in, g);
    for ( u32 dl : { 3u, 4u } ) {
      f64 rmse[2];
      for ( u32 tf = 0; tf < 2; ++tf ) {
        const hsc::hopf_opts o{ .level = 6, .dim_log2 = dl, .transform = tf == 1 };
        hsc::fhsc z = hsc::hopf(tutil::view(src), o, sc);
        auto back = hsc::unhopf(hsc::bytes{ z.first(), z.size() }, sc);
        sb::require(back.is_first());
        auto &b = back.cast<hsc::fhsc>();
        f64 se = 0;
        for ( usize i = 0; i < n_in; ++i ) {
          const f64 e = static_cast<f64>(b.first()[i]) - static_cast<f64>(src[i]);
          se += e * e;
        }
        rmse[tf] = __builtin_sqrt(se / static_cast<f64>(n_in));
      }
      sb::require(rmse[1] < rmse[0]);      // decisive in the benches; here just strictly better
    }
  }

  sb::test_case("bin: the empty input is exactly the 48-byte stream and round-trips");
  {
    hsc::fhsc z = hsc::hopf(hsc::bytes{ nullptr, 0 }, hsc::hopf_opts{}, sc);
    sb::require(z.size(), hsc::k_header_size + hsc::k_trailer_size);
    auto back = hsc::unhopf(hsc::bytes{ z.first(), z.size() }, sc);
    sb::require(back.is_first());
    sb::require(back.cast<hsc::fhsc>().size(), 0ull);
  }

  sb::test_case("vec: f32 blocks round-trip with the advertised gain scale");
  {
    for ( u32 dl : { 2u, 3u, 4u } ) {
      const u32 n = 1u << dl;
      const usize elems = 256 * n + n / 2;      // partial tail block
      micron::vector<f32> src;
      src.reserve(elems + 1);
      for ( usize i = 0; i < elems; ++i ) src.push_back(static_cast<f32>(g.unit() * 3.0));
      const hsc::hopf_opts o{ .m = hsc::mode::vec, .level = 6, .dim_log2 = dl, .gain_bits = 10 };
      auto zr = hsc::hopf(hsc::as_floats(src), o, sc);
      sb::require(zr.is_first());
      auto &z = zr.cast<hsc::fhsc>();

      auto pi = hsc::hopf_probe(hsc::bytes{ z.first(), z.size() });
      sb::require(pi.is_first());
      sb::require(pi.cast<hsc::hopf_info>().m == hsc::mode::vec);
      sb::require(pi.cast<hsc::hopf_info>().n_elems, static_cast<u64>(elems));

      auto back = hsc::unhopf_f32(hsc::bytes{ z.first(), z.size() });
      sb::require(back.is_first());
      auto &b = back.cast<hsc::fhsc32>();
      sb::require(b.size(), elems);
      const f64 d = hsc::d_of(hsc::level_dq(6));
      const f64 scale = static_cast<f64>(hsc::__u2f(pi.cast<hsc::hopf_info>().gscale_bits));
      f64 se = 0;
      for ( usize i = 0; i < elems; ++i ) {
        const f64 e = static_cast<f64>(b.first()[i]) - static_cast<f64>(src[i]);
        se += e * e;
      }
      const u64 nb = (elems + n - 1) / n;
      // per-block error envelope 2.5 g d + gain step, g <= scale
      const f64 env = 2.5 * scale * d + scale / 1023.0;
      sb::require(__builtin_sqrt(se / static_cast<f64>(nb)) <= env);
    }
  }

  sb::test_case("vec: NaN and Inf inputs fail bad_value; bin streams refuse the f32 lane");
  {
    micron::vector<f32> bad;
    bad.reserve(9);
    for ( i32 i = 0; i < 8; ++i ) bad.push_back(1.0f);
    bad[5] = hsc::__u2f(0x7FC00000u);
    auto r = hsc::hopf(hsc::as_floats(bad), hsc::hopf_opts{ .m = hsc::mode::vec }, sc);
    sb::require(r.is_second());
    sb::require(r.cast<hsc::error>() == hsc::error::bad_value);

    const auto src = noise_bytes(64, g);
    hsc::fhsc z = hsc::hopf(tutil::view(src), hsc::hopf_opts{}, sc);
    auto t = hsc::unhopf_f32(hsc::bytes{ z.first(), z.size() });
    sb::require(t.is_second());
    sb::require(t.cast<hsc::error>() == hsc::error::bad_opts);
  }

  sb::test_case("unit: no gain field, unit blocks round-trip, divisibility enforced");
  {
    const u32 n = 8;
    const usize elems = 128 * n;
    micron::vector<f32> src;
    src.reserve(elems + 1);
    for ( usize b = 0; b < elems / n; ++b ) {
      f64 v[8];
      f64 s = 0;
      for ( u32 c = 0; c < n; ++c ) {
        v[c] = g.unit() + 1e-3;
        s += v[c] * v[c];
      }
      const f64 nn = __builtin_sqrt(s);
      for ( u32 c = 0; c < n; ++c ) src.push_back(static_cast<f32>(v[c] / nn));
    }
    const hsc::hopf_opts o{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 3 };
    auto zr = hsc::hopf(hsc::as_floats(src), o, sc);
    sb::require(zr.is_first());
    auto &z = zr.cast<hsc::fhsc>();
    auto back = hsc::unhopf_f32(hsc::bytes{ z.first(), z.size() });
    sb::require(back.is_first());
    auto &b = back.cast<hsc::fhsc32>();
    const f64 d = hsc::d_of(hsc::level_dq(6));
    for ( usize blk = 0; blk < elems / n; ++blk ) {
      f64 e2 = 0;
      for ( u32 c = 0; c < n; ++c ) {
        const f64 e = static_cast<f64>(b.first()[blk * n + c]) - static_cast<f64>(src[blk * n + c]);
        e2 += e * e;
      }
      sb::require(__builtin_sqrt(e2) <= d * 2.5 + 1e-6);
    }
    // ragged length refuses
    micron::vector<f32> ragged;
    ragged.reserve(n + 4);
    for ( u32 i = 0; i < n + 3; ++i ) ragged.push_back(0.5f);
    auto rr = hsc::hopf(hsc::as_floats(ragged), o, sc);
    sb::require(rr.is_second());
    sb::require(rr.cast<hsc::error>() == hsc::error::bad_length);
  }

  sb::test_case("quotient: phase-invariant round trip; representative is canonical");
  {
    const usize elems = 4 * 512;
    micron::vector<f32> src;
    src.reserve(elems + 1);
    for ( usize i = 0; i < elems; ++i ) src.push_back(static_cast<f32>(g.unit() + 1e-4));
    const hsc::hopf_opts o{ .m = hsc::mode::quotient, .level = 6 };
    auto zr = hsc::hopf(hsc::as_floats(src), o, sc);
    sb::require(zr.is_first());
    auto &z = zr.cast<hsc::fhsc>();

    // rotating every pair by a global phase must give the identical stream
    micron::vector<f32> rot;
    rot.reserve(elems + 1);
    {
      const f64 c = micron::cos(1.234), s = micron::sin(1.234);
      for ( usize b = 0; b < elems / 4; ++b ) {
        const f32 *p = src.data() + b * 4;
        rot.push_back(static_cast<f32>(p[0] * c - p[1] * s));
        rot.push_back(static_cast<f32>(p[0] * s + p[1] * c));
        rot.push_back(static_cast<f32>(p[2] * c - p[3] * s));
        rot.push_back(static_cast<f32>(p[2] * s + p[3] * c));
      }
    }
    auto zr2 = hsc::hopf(hsc::as_floats(rot), o, sc);
    sb::require(zr2.is_first());
    auto &z2 = zr2.cast<hsc::fhsc>();
    sb::require(z.size() == z2.size());
    sb::require(tutil::bytes_equal({ z.first(), z.size() }, { z2.first(), z2.size() }));

    auto back = hsc::unhopf_f32(hsc::bytes{ z.first(), z.size() });
    sb::require(back.is_first());
    auto &b = back.cast<hsc::fhsc32>();
    sb::require(b.size(), elems);
    for ( usize blk = 0; blk < elems / 4; ++blk ) {
      sb::require(static_cast<f64>(b.first()[blk * 4]) >= 0.0);      // canonical z0 real >= 0
      sb::require(b.first()[blk * 4 + 1] == 0.0f);
    }
  }

  sb::test_case("quat: fiber-invariant round trip; representative is canonical");
  {
    const usize elems = 8 * 256;
    micron::vector<f32> src;
    src.reserve(elems + 1);
    for ( usize i = 0; i < elems; ++i ) src.push_back(static_cast<f32>(g.unit() + 1e-4));
    const hsc::hopf_opts o{ .m = hsc::mode::quat, .level = 6 };
    auto zr = hsc::hopf(hsc::as_floats(src), o, sc);
    sb::require(zr.is_first());
    auto &z = zr.cast<hsc::fhsc>();

    // right-multiplying every pair by ONE global unit quaternion must give the identical stream
    f64 gu[4] = { 0.3, -0.5, 0.4, 0.7 };
    {
      const f64 n = __builtin_sqrt(hsc::__fma_norm2(gu, 4));
      for ( u32 k = 0; k < 4; ++k ) gu[k] /= n;
    }
    micron::vector<f32> rot;
    rot.reserve(elems + 1);
    for ( usize blk = 0; blk < elems / 8; ++blk ) {
      f64 x[4], y[4], xr[4], yr[4];
      for ( u32 k = 0; k < 4; ++k ) {
        x[k] = static_cast<f64>(src[blk * 8 + k]);
        y[k] = static_cast<f64>(src[blk * 8 + 4 + k]);
      }
      hsc::quat_mul(x, gu, xr);
      hsc::quat_mul(y, gu, yr);
      for ( u32 k = 0; k < 4; ++k ) rot.push_back(static_cast<f32>(xr[k]));
      for ( u32 k = 0; k < 4; ++k ) rot.push_back(static_cast<f32>(yr[k]));
    }
    auto zr2 = hsc::hopf(hsc::as_floats(rot), o, sc);
    sb::require(zr2.is_first());
    auto &z2 = zr2.cast<hsc::fhsc>();
    sb::require(z.size() == z2.size());
    sb::require(tutil::bytes_equal({ z.first(), z.size() }, { z2.first(), z2.size() }));

    auto back = hsc::unhopf_f32(hsc::bytes{ z.first(), z.size() });
    sb::require(back.is_first());
    auto &b = back.cast<hsc::fhsc32>();
    sb::require(b.size(), elems);
    for ( usize blk = 0; blk < elems / 8; ++blk ) {
      for ( u32 k = 0; k < 3; ++k ) sb::require(b.first()[blk * 8 + k] == 0.0f);      // canonical q0 = (0,0,0,c)
      sb::require(static_cast<f64>(b.first()[blk * 8 + 3]) >= 0.0);
    }

    // ragged length refuses
    micron::vector<f32> ragged;
    ragged.reserve(21);
    for ( u32 i = 0; i < 20; ++i ) ragged.push_back(0.5f);
    auto rr = hsc::hopf(hsc::as_floats(ragged), o, sc);
    sb::require(rr.is_second());
    sb::require(rr.cast<hsc::error>() == hsc::error::bad_length);
  }

  sb::test_case("oct: fiber-invariant round trip; representative is canonical");
  {
    const usize elems = 16 * 128;
    micron::vector<f32> src;
    src.reserve(elems + 1);
    for ( usize i = 0; i < elems; ++i ) src.push_back(static_cast<f32>(g.unit() + 1e-4));
    const hsc::hopf_opts o{ .m = hsc::mode::oct, .level = 5 };
    auto zr = hsc::hopf(hsc::as_floats(src), o, sc);
    sb::require(zr.is_first());
    auto &z = zr.cast<hsc::fhsc>();

    // replace every pair by a fiber mate through one fixed unit octonion (the graph-sphere
    // parametrization: y = s u, x = (v u)/(2s)) -- must give the identical stream
    f64 u[8] = { 0.2, -0.4, 0.1, 0.6, -0.3, 0.2, 0.4, 0.35 };
    {
      const f64 n = __builtin_sqrt(hsc::__fma_norm2(u, 8));
      for ( u32 k = 0; k < 8; ++k ) u[k] /= n;
    }
    micron::vector<f32> rot;
    rot.reserve(elems + 1);
    for ( usize blk = 0; blk < elems / 16; ++blk ) {
      f64 p[9]{};
      sb::require(hsc::oct_project(src.data() + blk * 16, p) >= 0);
      const f64 s = __builtin_sqrt((1.0 - p[8]) * 0.5);
      const f64 c = __builtin_sqrt((1.0 + p[8]) * 0.5);
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
      for ( u32 k = 0; k < 8; ++k ) rot.push_back(static_cast<f32>(x[k]));
      for ( u32 k = 0; k < 8; ++k ) rot.push_back(static_cast<f32>(y[k]));
    }
    auto zr2 = hsc::hopf(hsc::as_floats(rot), o, sc);
    sb::require(zr2.is_first());
    auto &z2 = zr2.cast<hsc::fhsc>();
    sb::require(z.size() == z2.size());
    sb::require(tutil::bytes_equal({ z.first(), z.size() }, { z2.first(), z2.size() }));

    auto back = hsc::unhopf_f32(hsc::bytes{ z.first(), z.size() });
    sb::require(back.is_first());
    auto &b = back.cast<hsc::fhsc32>();
    sb::require(b.size(), elems);
    for ( usize blk = 0; blk < elems / 16; ++blk ) {
      for ( u32 k = 0; k < 7; ++k ) sb::require(b.first()[blk * 16 + k] == 0.0f);      // canonical o0 = (0,...,0,c)
      sb::require(static_cast<f64>(b.first()[blk * 16 + 7]) >= 0.0);
    }

    // ragged length refuses
    micron::vector<f32> ragged;
    ragged.reserve(25);
    for ( u32 i = 0; i < 24; ++i ) ragged.push_back(0.5f);
    auto rr = hsc::hopf(hsc::as_floats(ragged), o, sc);
    sb::require(rr.is_second());
    sb::require(rr.cast<hsc::error>() == hsc::error::bad_length);
  }

  sb::test_case("transform: vec and unit hold their envelopes; quotient forces bit2 off the wire");
  {
    const u32 n = 8;
    const f64 d = hsc::d_of(hsc::level_dq(6));
    const usize elems = 256 * n + n / 2;      // partial tail block
    micron::vector<f32> src;
    src.reserve(elems + 1);
    for ( usize i = 0; i < elems; ++i ) src.push_back(static_cast<f32>(g.unit() * 3.0));
    const hsc::hopf_opts ov{ .m = hsc::mode::vec, .level = 6, .dim_log2 = 3, .gain_bits = 10, .transform = true };
    auto zr = hsc::hopf(hsc::as_floats(src), ov, sc);
    sb::require(zr.is_first());
    auto &z = zr.cast<hsc::fhsc>();
    auto pi = hsc::hopf_probe(hsc::bytes{ z.first(), z.size() });
    sb::require(pi.is_first());
    sb::require(pi.cast<hsc::hopf_info>().transform);
    auto back = hsc::unhopf_f32(hsc::bytes{ z.first(), z.size() });
    sb::require(back.is_first());
    auto &b = back.cast<hsc::fhsc32>();
    const f64 scale = static_cast<f64>(hsc::__u2f(pi.cast<hsc::hopf_info>().gscale_bits));
    f64 se = 0;
    for ( usize i = 0; i < elems; ++i ) {
      const f64 e = static_cast<f64>(b.first()[i]) - static_cast<f64>(src[i]);
      se += e * e;
    }
    const u64 nb = (elems + n - 1) / n;
    sb::require(__builtin_sqrt(se / static_cast<f64>(nb)) <= 2.5 * scale * d + scale / 1023.0);

    micron::vector<f32> us;
    us.reserve(64 * n + 1);
    for ( usize blk = 0; blk < 64; ++blk ) {
      f64 v[8], s = 0;
      for ( u32 c = 0; c < n; ++c ) {
        v[c] = g.unit() + 1e-3;
        s += v[c] * v[c];
      }
      const f64 nn = __builtin_sqrt(s);
      for ( u32 c = 0; c < n; ++c ) us.push_back(static_cast<f32>(v[c] / nn));
    }
    const hsc::hopf_opts ou{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 3, .transform = true };
    auto zu = hsc::hopf(hsc::as_floats(us), ou, sc);
    sb::require(zu.is_first());
    auto &zb = zu.cast<hsc::fhsc>();
    auto bu = hsc::unhopf_f32(hsc::bytes{ zb.first(), zb.size() });
    sb::require(bu.is_first());
    auto &ub = bu.cast<hsc::fhsc32>();
    for ( usize blk = 0; blk < 64; ++blk ) {
      f64 e2 = 0;
      for ( u32 c = 0; c < n; ++c ) {
        const f64 e = static_cast<f64>(ub.first()[blk * n + c]) - static_cast<f64>(us[blk * n + c]);
        e2 += e * e;
      }
      sb::require(__builtin_sqrt(e2) <= d * 2.5 + 1e-6);
    }

    // the quotient family: .transform = true is accepted but resolve forces it off the wire
    for ( hsc::mode fm : { hsc::mode::quotient, hsc::mode::quat, hsc::mode::oct } ) {
      const usize be = fm == hsc::mode::quotient ? 4 : (fm == hsc::mode::quat ? 8 : 16);
      micron::vector<f32> qs;
      qs.reserve(be * 64 + 1);
      for ( usize i = 0; i < be * 64; ++i ) qs.push_back(static_cast<f32>(g.unit() + 1e-4));
      const hsc::hopf_opts oq{ .m = fm, .level = 5, .transform = true };
      auto zq = hsc::hopf(hsc::as_floats(qs), oq, sc);
      sb::require(zq.is_first());
      auto &zqb = zq.cast<hsc::fhsc>();
      sb::require((zqb.first()[5] & 0x04u) == 0);
      auto qi = hsc::hopf_probe(hsc::bytes{ zqb.first(), zqb.size() });
      sb::require(qi.is_first());
      sb::require(!qi.cast<hsc::hopf_info>().transform);
      auto qb = hsc::unhopf_f32(hsc::bytes{ zqb.first(), zqb.size() });
      sb::require(qb.is_first());
    }
  }

  sb::test_case("hopf_into cap contracts: bin returns 0, f32 lane short_output");
  {
    const auto src = noise_bytes(512, g);
    u8 tiny[16];
    sb::require(hsc::hopf_into(tutil::view(src), hsc::hopf_opts{}, tiny, sizeof(tiny), sc), 0ull);

    micron::vector<f32> fsrc;
    fsrc.reserve(65);
    for ( i32 i = 0; i < 64; ++i ) fsrc.push_back(0.5f);
    const max_t r = hsc::hopf_into(hsc::as_floats(fsrc), hsc::hopf_opts{ .m = hsc::mode::vec }, tiny, sizeof(tiny), sc);
    sb::require(hsc::as_error(r) == hsc::error::short_output);
  }

  sb::test_case("unhopf short output buffer fails short_output");
  {
    const auto src = noise_bytes(256, g);
    hsc::fhsc z = hsc::hopf(tutil::view(src), hsc::hopf_opts{}, sc);
    u8 small[64];
    auto r = hsc::unhopf(hsc::bytes{ z.first(), z.size() }, hsc::wbytes{ small, sizeof(small) });
    sb::require(r.is_second());
    sb::require(r.cast<hsc::error>() == hsc::error::short_output);
  }

  return 1;
}
