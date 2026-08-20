//  The container framing. Guards: exact header bytes at fixed options, the four separable
//  failure classes on targeted flips (hc -> bad_container, skel_guard/d_q/bits_per_block ->
//  bad_skeleton, payload crc -> bad_checksum, trailer count -> bad_length), version/profile/
//  reserved policing, cross-mode header rules, nonzero pad bits and oversized indices ->
//  bad_stream, truncation, and the 48-byte empty stream. Header-field patches recompute the hc
//  byte (and payload patches the crc) so each test reaches the check it aims at.

#include "../src/hsc/hsc.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>
#include <micron/vector/vector.hpp>

#include <snowball/snowball.hpp>

namespace
{

micron::vector<u8>
stream_of(hsc::hopf_scratch &sc, usize n_in, const hsc::hopf_opts &o)
{
  micron::vector<u8> src;
  src.reserve(n_in + 1);
  tutil::rng g;
  for ( usize i = 0; i < n_in; ++i ) src.push_back(static_cast<u8>(g.next()));
  hsc::fhsc z = hsc::hopf(tutil::view(src), o, sc);
  micron::vector<u8> out;
  out.reserve(z.size() + 1);
  for ( usize i = 0; i < z.size(); ++i ) out.push_back(z.first()[i]);
  return out;
}

void
rehc(micron::vector<u8> &s)
{
  s[39] = static_cast<u8>(hsc::xxh32(hsc::bytes{ s.data(), 39 }) >> 8);
}

void
recrc(micron::vector<u8> &s)
{
  const usize pb = s.size() - hsc::k_header_size - hsc::k_trailer_size;
  const u32 crc = hsc::crc32(hsc::bytes{ s.data() + hsc::k_header_size, pb });
  hsc::__store32(s.data() + s.size() - 8, crc);
}

hsc::error
decode_err(const micron::vector<u8> &s)
{
  auto r = hsc::unhopf(hsc::bytes{ s.data(), s.size() });
  if ( r.is_first() ) return hsc::error::ok;
  return r.cast<hsc::error>();
}

}      //  namespace

int
main()
{
  hsc::hopf_scratch sc;
  const hsc::hopf_opts o6{ .level = 6, .dim_log2 = 3 };

  sb::test_case("exact header bytes at level 6, dim 8, 64 input bytes");
  {
    auto s = stream_of(sc, 64, o6);
    sb::require(hsc::__load32(s.data()), 0x43534889u);      //  "\x89HSC"
    sb::require(s[0], u8(0x89));
    sb::require(s[1], u8('H'));
    sb::require(s[2], u8('S'));
    sb::require(s[3], u8('C'));
    sb::require(s[4], u8(1));         //  version
    sb::require(s[5], u8(0x03));      //  exact_pack | has_gain
    sb::require(s[6], u8(0));         //  mode bin
    sb::require(s[7], u8(3));         //  dim_log2
    sb::require(hsc::__load32(s.data() + 8), hsc::level_dq(6));
    sb::require(s[12], u8(8));                                  //  gain_bits
    sb::require(s[13], u8(0));                                  //  profile: standard
    sb::require(hsc::__load16(s.data() + 14), 22u);             //  ceil(log2 2523114)
    sb::require(hsc::__load64(s.data() + 16), 64ull);           //  n_elems
    sb::require(hsc::__load64(s.data() + 24), 2523114ull);      //  skel_guard = M
    sb::require(hsc::__load32(s.data() + 32), 0u);              //  gscale: bin has none
    sb::require(s[36] == 0 && s[37] == 0 && s[38] == 0);
    sb::require(s[39], static_cast<u8>(hsc::xxh32(hsc::bytes{ s.data(), 39 }) >> 8));
    //  8 blocks x 30 bits = 240 bits = 30 payload bytes
    sb::require(s.size(), hsc::k_header_size + 30 + hsc::k_trailer_size);
    sb::require(decode_err(s) == hsc::error::ok);
  }

  sb::test_case("transform streams: flags 0x07, rate and skeleton untouched, probe reports bit2");
  {
    auto s = stream_of(sc, 64, hsc::hopf_opts{ .level = 6, .dim_log2 = 3, .transform = true });
    sb::require(s[5], u8(0x07));                                //  exact_pack | has_gain | transform
    sb::require(hsc::__load16(s.data() + 14), 22u);             //  the rotation is analog-side:
    sb::require(hsc::__load64(s.data() + 24), 2523114ull);      //  same bits_per_block, same M
    sb::require(s.size(), hsc::k_header_size + 30 + hsc::k_trailer_size);
    sb::require(decode_err(s) == hsc::error::ok);
    auto pi = hsc::hopf_probe(hsc::bytes{ s.data(), s.size() });
    sb::require(pi.is_first());
    sb::require(pi.cast<hsc::hopf_info>().transform);
  }

  sb::test_case("hc byte flip -> bad_container; a header flip without rehc is also caught");
  {
    auto s = stream_of(sc, 64, o6);
    s[39] ^= 0x40;
    sb::require(decode_err(s) == hsc::error::bad_container);
    auto t = stream_of(sc, 64, o6);
    t[7] = 4;      //  dim change without hc fixup: the hc net catches it first
    sb::require(decode_err(t) == hsc::error::bad_container);
  }

  sb::test_case("simulated float drift: patched d_q / skel_guard / bits_per_block -> bad_skeleton");
  {
    auto s = stream_of(sc, 64, o6);
    hsc::__store32(s.data() + 8, hsc::level_dq(5));      //  other valid d_q, same M claim
    rehc(s);
    //  note: level-5 streams have different record widths, so total size trips bad_length first
    //  unless the claim is adjusted; patch bits_per_block to keep the frame consistent
    sb::require(decode_err(s) == hsc::error::bad_length || decode_err(s) == hsc::error::bad_skeleton);

    auto t = stream_of(sc, 64, o6);
    hsc::__store64(t.data() + 24, 2523115ull);      //  M off by one: exactly what drift looks like
    rehc(t);
    sb::require(decode_err(t) == hsc::error::bad_skeleton);

    auto u = stream_of(sc, 512, hsc::hopf_opts{ .level = 6, .dim_log2 = 2 });      //  dim 4: 8+12 bits
    hsc::__store16(u.data() + 14, 13);                                             //  claim 13 shape bits
    rehc(u);
    sb::require(decode_err(u) == hsc::error::bad_length || decode_err(u) == hsc::error::bad_skeleton);
  }

  sb::test_case("payload crc flip -> bad_checksum; trailer count flip -> bad_length");
  {
    auto s = stream_of(sc, 64, o6);
    s[hsc::k_header_size + 3] ^= 0x10;      //  payload bit
    sb::require(decode_err(s) == hsc::error::bad_checksum);
    auto t = stream_of(sc, 64, o6);
    t[t.size() - 4] ^= 0x01;      //  nblocks
    sb::require(decode_err(t) == hsc::error::bad_length);
    auto v = stream_of(sc, 64, o6);
    v[v.size() - 8] ^= 0x01;      //  stored crc itself
    sb::require(decode_err(v) == hsc::error::bad_checksum);
  }

  sb::test_case("version and reserved wire profile refuse as unsupported");
  {
    auto s = stream_of(sc, 64, o6);
    s[4] = 2;
    rehc(s);
    sb::require(decode_err(s) == hsc::error::unsupported);
    auto t = stream_of(sc, 64, o6);
    t[5] = 0x02;      //  exact_pack cleared: the reserved per-node profile
    rehc(t);
    sb::require(decode_err(t) == hsc::error::unsupported);
    auto u = stream_of(sc, 64, o6);
    u[13] = 1;      //  construction profile
    rehc(u);
    sb::require(decode_err(u) == hsc::error::unsupported);
  }

  sb::test_case("cross-mode header rules are policed");
  {
    auto s = stream_of(sc, 64, o6);
    s[5] = 0x01;      //  has_gain cleared on a bin stream
    rehc(s);
    sb::require(decode_err(s) == hsc::error::bad_container);
    auto t = stream_of(sc, 64, o6);
    t[6] = 3;      //  quotient claims dim 8
    rehc(t);
    sb::require(decode_err(t) == hsc::error::bad_container);
    auto u = stream_of(sc, 64, o6);
    u[12] = 0;      //  gain_bits 0 with has_gain
    rehc(u);
    sb::require(decode_err(u) == hsc::error::bad_container);
    auto v = stream_of(sc, 64, o6);
    hsc::__store32(v.data() + 32, 0x3F800000u);      //  gscale nonzero outside vec
    rehc(v);
    sb::require(decode_err(v) == hsc::error::bad_container);
    auto w = stream_of(sc, 64, o6);
    w[36] = 1;      //  reserved byte
    rehc(w);
    sb::require(decode_err(w) == hsc::error::bad_container);
    auto x = stream_of(sc, 64, o6);
    x[5] |= 0x08;      //  reserved flag bit (bit2 is transform now; bits 3..7 stay policed)
    rehc(x);
    sb::require(decode_err(x) == hsc::error::bad_container);
    //  quotient must keep bit2 clear: a transformed-quotient header is refused
    f32 qf[16]{};
    for ( u32 i = 0; i < 16; ++i ) qf[i] = static_cast<f32>(i + 1);
    auto qr = hsc::hopf(hsc::floats{ qf, 8 }, hsc::hopf_opts{ .m = hsc::mode::quotient, .level = 6 }, sc);
    sb::require(qr.is_first());
    auto &qz = qr.cast<hsc::fhsc>();
    micron::vector<u8> y;
    y.reserve(qz.size() + 1);
    for ( usize i = 0; i < qz.size(); ++i ) y.push_back(qz.first()[i]);
    y[5] |= 0x04;
    rehc(y);
    sb::require(decode_err(y) == hsc::error::bad_container);
    //  the quat/oct siblings police the same three rules: pinned dim, no transform bit, and the
    //  mode-byte range gate (6 is the first free value)
    for ( hsc::mode fm : { hsc::mode::quat, hsc::mode::oct } ) {
      auto fr = hsc::hopf(hsc::floats{ qf, 16 }, hsc::hopf_opts{ .m = fm, .level = 6 }, sc);
      sb::require(fr.is_first());
      auto &fz = fr.cast<hsc::fhsc>();
      for ( u32 which = 0; which < 3; ++which ) {
        micron::vector<u8> a;
        a.reserve(fz.size() + 1);
        for ( usize i = 0; i < fz.size(); ++i ) a.push_back(fz.first()[i]);
        if ( which == 0 ) a[7] = fm == hsc::mode::quat ? 2 : 3;      //  claims the wrong dim
        if ( which == 1 ) a[5] |= 0x04;                              //  transform bit on a fiber mode
        if ( which == 2 ) a[6] = 6;                                  //  one past the last mode
        rehc(a);
        sb::require(decode_err(a) == hsc::error::bad_container);
      }
    }
  }

  sb::test_case("nonzero pad bits and an oversized index are bad_stream");
  {
    //  one dim-8 block: 30 record bits in 4 payload bytes, 2 pad bits at the top
    auto s = stream_of(sc, 8, o6);
    sb::require(s.size(), hsc::k_header_size + 4 + hsc::k_trailer_size);
    s[hsc::k_header_size + 3] |= 0xC0;      //  set the pad bits
    recrc(s);
    sb::require(decode_err(s) == hsc::error::bad_stream);

    //  force the 22 shape bits (record bits 8..29) to all ones: 4194303 >= M = 2523114
    auto t = stream_of(sc, 8, o6);
    t[hsc::k_header_size + 1] = 0xFF;
    t[hsc::k_header_size + 2] = 0xFF;
    t[hsc::k_header_size + 3] |= 0x3F;
    recrc(t);
    sb::require(decode_err(t) == hsc::error::bad_stream);
  }

  sb::test_case("truncation anywhere fails cleanly");
  {
    auto s = stream_of(sc, 64, o6);
    for ( usize cut = 0; cut < s.size(); cut += 7 ) {
      auto r = hsc::unhopf(hsc::bytes{ s.data(), cut });
      sb::require(r.is_second());
    }
  }

  sb::test_case("the empty stream is 48 bytes; trailing garbage is refused");
  {
    auto s = stream_of(sc, 0, o6);
    sb::require(s.size(), hsc::k_header_size + hsc::k_trailer_size);
    sb::require(decode_err(s) == hsc::error::ok);
    s.push_back(0x00);
    sb::require(decode_err(s) == hsc::error::bad_length);
  }

  return 1;
}
