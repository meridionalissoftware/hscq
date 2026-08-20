//  LSB bit I/O: writer/reader round-trip under random field widths, strict end-of-input
//  accounting, fast-path/slow-path byte identity, and the consteval twin producing the same
//  bytes as the runtime path. Wide (>64-bit) fields are the pack layer's job and are tested in
//  pack.cpp, not here.

#include "../src/hsc/bits/bitreader.hpp"
#include "../src/hsc/bits/bitwriter.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

//  a fixed comptime pattern: 3+11+32+7+1 bits, written then read back in constant evaluation
constexpr bool
comptime_roundtrip()
{
  u8 buf[16]{};
  hsc::bits::bitwriter w{ .out = buf, .fast_end = buf };      //  fast_end = out: force the portable path
  w.add(0b101u, 3);
  w.flush();
  w.add(0x5A3u, 11);
  w.flush();
  w.add(0xDEADBEEFu, 32);
  w.flush();
  w.add(0x55u, 7);
  w.flush();
  w.add(1u, 1);
  const u8 *end = w.finish();

  hsc::bits::bitreader r{ .p = buf, .end = end };
  if ( !r.need(3) || r.bits(3) != 0b101u ) return false;
  if ( !r.need(11) || r.bits(11) != 0x5A3u ) return false;
  if ( !r.need(32) || r.bits(32) != 0xDEADBEEFu ) return false;
  if ( !r.need(7) || r.bits(7) != 0x55u ) return false;
  if ( !r.need(1) || r.bits(1) != 1u ) return false;
  return end - buf == 7;      //  54 bits -> 7 bytes
}

static_assert(comptime_roundtrip());

}      //  namespace

int
main()
{
  sb::test_case("random widths round-trip, fast path");
  {
    constexpr usize N = 4096;
    static u8 buf[8 * N + 64];
    u32 widths[N];
    u64 vals[N];
    tutil::rng g;
    for ( usize i = 0; i < N; ++i ) {
      widths[i] = 1 + static_cast<u32>(g.below(32));
      vals[i] = g.next() & ((1ull << widths[i]) - 1);
    }
    hsc::bits::bitwriter w{ .out = buf, .fast_end = buf + sizeof(buf) - 8 };
    for ( usize i = 0; i < N; ++i ) {
      w.add(vals[i], static_cast<i32>(widths[i]));
      w.flush();
    }
    const u8 *end = w.finish();

    hsc::bits::bitreader r{ .p = buf, .end = end };
    for ( usize i = 0; i < N; ++i ) {
      sb::require(r.need(widths[i]));
      sb::require(static_cast<u64>(r.bits(widths[i])), vals[i]);
    }
  }

  sb::test_case("fast path and portable path emit identical bytes");
  {
    constexpr usize N = 1024;
    static u8 fastb[8 * N + 64], slowb[8 * N + 64];
    tutil::rng g;
    u32 widths[N];
    u64 vals[N];
    for ( usize i = 0; i < N; ++i ) {
      widths[i] = 1 + static_cast<u32>(g.below(32));
      vals[i] = g.next() & ((1ull << widths[i]) - 1);
    }
    hsc::bits::bitwriter wf{ .out = fastb, .fast_end = fastb + sizeof(fastb) - 8 };
    hsc::bits::bitwriter ws{ .out = slowb, .fast_end = slowb };      //  never take the fast store
    for ( usize i = 0; i < N; ++i ) {
      wf.add(vals[i], static_cast<i32>(widths[i]));
      wf.flush();
      ws.add(vals[i], static_cast<i32>(widths[i]));
      ws.flush();
    }
    const u8 *fe = wf.finish();
    const u8 *se = ws.finish();
    sb::require(fe - fastb == se - slowb);
    sb::require(tutil::bytes_equal({ fastb, static_cast<usize>(fe - fastb) }, { slowb, static_cast<usize>(se - slowb) }));
  }

  sb::test_case("strict end accounting: need() fails past the last real bit");
  {
    u8 buf[4]{};
    hsc::bits::bitwriter w{ .out = buf, .fast_end = buf };
    w.add(0x7u, 3);
    const u8 *end = w.finish();      //  1 byte out, 5 pad bits
    sb::require(end - buf == 1);

    hsc::bits::bitreader r{ .p = buf, .end = end };
    sb::require(r.need(3));
    sb::require(static_cast<u64>(r.bits(3)), 0x7ull);
    sb::require(r.need(5));      //  the pad bits are readable...
    sb::require(static_cast<u64>(r.bits(5)), 0ull);
    sb::require(!r.need(1));      //  ...but nothing past them
  }

  sb::test_case("finish() zero-pads the final partial byte");
  {
    tutil::rng g;
    for ( u32 w = 1; w <= 32; ++w ) {
      u8 buf[16]{};
      hsc::bits::bitwriter bw{ .out = buf, .fast_end = buf };
      const u64 v = g.next() & ((1ull << w) - 1);
      bw.add(v, static_cast<i32>(w));
      const u8 *end = bw.finish();
      hsc::bits::bitreader r{ .p = buf, .end = end };
      sb::require(r.need(w));
      sb::require(static_cast<u64>(r.bits(w)), v);
      const u32 pad = static_cast<u32>(8 * static_cast<u64>(end - buf) - w);
      if ( pad ) {
        sb::require(r.need(pad));
        sb::require(static_cast<u64>(r.bits(pad)), 0ull);      //  decoder relies on zero pad
      }
    }
  }

  sb::test_case("consumed() counts whole bytes actually drained");
  {
    u8 buf[64]{};
    hsc::bits::bitwriter w{ .out = buf, .fast_end = buf };
    for ( i32 i = 0; i < 32; ++i ) {
      w.add(static_cast<u64>(i) & 0xFu, 4);
      w.flush();
    }
    const u8 *end = w.finish();      //  128 bits = 16 bytes
    sb::require(end - buf == 16);
    hsc::bits::bitreader r{ .p = buf, .end = end };
    u64 got = 0;
    for ( i32 i = 0; i < 32; ++i ) {
      sb::require(r.need(4));
      got += r.bits(4);
      (void)got;
    }
    sb::require(r.consumed(buf), 16ull);
  }

  return 1;
}
