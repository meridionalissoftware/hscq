//  Decoder fuzz: deterministic corpus of valid streams under single-bit flips at every byte,
//  truncations, random garbage, and magic-prefixed garbage. The contract is NO crash and NO
//  UB (run this under duck debug --asan --ubsan as the sanitizer gate) -- every input returns
//  cleanly. A flip is allowed to decode successfully in the rare cases the 8-bit header check
//  collides AND every field still validates; what it may never do is walk out of bounds.

#include "../src/hsc/hsc.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>
#include <micron/vector/vector.hpp>

#include <snowball/snowball.hpp>

namespace
{

//  exercise every decode surface on one candidate buffer
void
poke(hsc::hopf_scratch &sc, const u8 *p, usize n)
{
  (void)hsc::hopf_probe(hsc::bytes{ p, n });
  (void)hsc::verify(hsc::bytes{ p, n });
  auto owned = hsc::unhopf(hsc::bytes{ p, n }, sc);
  (void)owned;
  static u8 out[1 << 16];
  (void)hsc::unhopf(hsc::bytes{ p, n }, hsc::wbytes{ out, sizeof(out) });
  static f32 fout[1 << 12];
  (void)hsc::unhopf(hsc::bytes{ p, n }, hsc::wfloats{ fout, sizeof(fout) / sizeof(f32) });

  //  the range surface parses the same attacker-controlled header and then SEEKS on the block
  //  count and record width it read out of it, so it needs the same treatment. Windows are taken
  //  from the header's own nblocks (in range and past the end) and from wild values.
  const u64 wild[] = { 0ull, 1ull, 7ull, 0xFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull };
  for ( u64 f : wild )
    for ( u64 c : wild ) {
      (void)hsc::unhopf_range(hsc::bytes{ p, n }, f, c, hsc::wbytes{ out, sizeof(out) }, sc);
      (void)hsc::unhopf_range(hsc::bytes{ p, n }, f, c, hsc::wfloats{ fout, sizeof(fout) / sizeof(f32) }, sc);
    }
  const auto probe = hsc::hopf_probe(hsc::bytes{ p, n });
  if ( probe.is_first() ) {
    const u64 nb = probe.cast<hsc::hopf_info>().nblocks;
    const u64 at[] = { nb, nb ? nb - 1 : 0, nb / 2, nb + 1 };
    for ( u64 f : at )
      for ( u64 c : at ) {
        (void)hsc::unhopf_range(hsc::bytes{ p, n }, f, c, hsc::wbytes{ out, sizeof(out) }, sc);
        (void)hsc::unhopf_range(hsc::bytes{ p, n }, f, c, hsc::wfloats{ fout, sizeof(fout) / sizeof(f32) }, sc);
      }
    //  a deliberately undersized sink: short_output, never a write past the end
    (void)hsc::unhopf_range(hsc::bytes{ p, n }, 0, nb, hsc::wbytes{ out, 1 }, sc);
  }
}

}      //  namespace

int
main()
{
  hsc::hopf_scratch sc;
  tutil::rng g;

  //  seed streams: one per mode, mixed dims/levels, including empty and tiny payloads, plus
  //  transformed (flags bit2) variants of every transformable mode
  micron::vector<micron::vector<u8>> seeds;
  seeds.reserve(8);
  {
    micron::vector<u8> raw;
    raw.reserve(600);
    for ( i32 i = 0; i < 555; ++i ) raw.push_back(static_cast<u8>(g.next()));
    micron::vector<f32> fl;
    fl.reserve(260);
    for ( i32 i = 0; i < 256; ++i ) fl.push_back(static_cast<f32>(g.unit() + 1e-4));

    auto push = [&](hsc::fhsc &&z) {
      micron::vector<u8> s;
      s.reserve(z.size() + 1);
      for ( usize i = 0; i < z.size(); ++i ) s.push_back(z.first()[i]);
      seeds.push_back(micron::move(s));
    };
    push(hsc::hopf(tutil::view(raw), hsc::hopf_opts{ .level = 6, .dim_log2 = 3 }, sc));
    push(hsc::hopf(tutil::view(raw), hsc::hopf_opts{ .level = 3, .dim_log2 = 2 }, sc));
    push(hsc::hopf(hsc::bytes{ nullptr, 0 }, hsc::hopf_opts{}, sc));
    auto v = hsc::hopf(hsc::as_floats(fl), hsc::hopf_opts{ .m = hsc::mode::vec, .level = 6, .dim_log2 = 4 }, sc);
    sb::require(v.is_first());
    push(micron::move(v.cast<hsc::fhsc>()));
    auto q = hsc::hopf(hsc::as_floats(fl), hsc::hopf_opts{ .m = hsc::mode::quotient, .level = 6 }, sc);
    sb::require(q.is_first());
    push(micron::move(q.cast<hsc::fhsc>()));
    auto q4 = hsc::hopf(hsc::as_floats(fl), hsc::hopf_opts{ .m = hsc::mode::quat, .level = 6 }, sc);
    sb::require(q4.is_first());
    push(micron::move(q4.cast<hsc::fhsc>()));
    auto q8 = hsc::hopf(hsc::as_floats(fl), hsc::hopf_opts{ .m = hsc::mode::oct, .level = 6 }, sc);
    sb::require(q8.is_first());
    push(micron::move(q8.cast<hsc::fhsc>()));
    //  transformed seeds, one per transformable mode: flips explore bit2-adjacent state
    push(hsc::hopf(tutil::view(raw), hsc::hopf_opts{ .level = 6, .dim_log2 = 3, .transform = true }, sc));
    auto vt = hsc::hopf(hsc::as_floats(fl), hsc::hopf_opts{ .m = hsc::mode::vec, .level = 6, .dim_log2 = 4, .transform = true }, sc);
    sb::require(vt.is_first());
    push(micron::move(vt.cast<hsc::fhsc>()));
    auto ut = hsc::hopf(hsc::as_floats(fl), hsc::hopf_opts{ .m = hsc::mode::unit, .level = 6, .dim_log2 = 3, .transform = true }, sc);
    sb::require(ut.is_first());
    push(micron::move(ut.cast<hsc::fhsc>()));
  }

  sb::test_case("single-bit flips at every byte of every seed return cleanly");
  {
    for ( usize si = 0; si < seeds.size(); ++si ) {
      micron::vector<u8> m;
      m.reserve(seeds[si].size() + 1);
      for ( usize i = 0; i < seeds[si].size(); ++i ) m.push_back(seeds[si][i]);
      for ( usize pos = 0; pos < m.size(); ++pos ) {
        const u8 keep = m[pos];
        m[pos] = keep ^ static_cast<u8>(1u << (pos % 8));
        poke(sc, m.data(), m.size());
        m[pos] = keep;
      }
    }
  }

  sb::test_case("every truncation length of every seed returns cleanly");
  {
    for ( usize si = 0; si < seeds.size(); ++si )
      for ( usize cut = 0; cut <= seeds[si].size(); ++cut ) poke(sc, seeds[si].data(), cut);
  }

  sb::test_case("random garbage and magic-prefixed garbage return cleanly");
  {
    static u8 buf[4096];
    for ( i32 t = 0; t < 400; ++t ) {
      const usize n = 1 + static_cast<usize>(g.below(sizeof(buf)));
      for ( usize i = 0; i < n; ++i ) buf[i] = static_cast<u8>(g.next());
      poke(sc, buf, n);
      //  now force the magic + a plausible version so parsing goes deeper
      if ( n >= 6 ) {
        hsc::__store32(buf, hsc::k_magic);
        buf[4] = 1;
        buf[5] = static_cast<u8>(g.below(8));      //  reach the transform bit too
        poke(sc, buf, n);
        //  and a correct hc byte so field validation itself gets fuzzed
        if ( n >= 48 ) {
          buf[39] = static_cast<u8>(hsc::xxh32(hsc::bytes{ buf, 39 }) >> 8);
          poke(sc, buf, n);
        }
      }
    }
  }

  sb::test_case("valid seeds still decode after the fuzz (scratch state is not poisoned)");
  {
    for ( usize si = 0; si < seeds.size(); ++si ) {
      auto r = hsc::unhopf(hsc::bytes{ seeds[si].data(), seeds[si].size() }, sc);
      sb::require(r.is_first());
    }
  }

  return 1;
}
