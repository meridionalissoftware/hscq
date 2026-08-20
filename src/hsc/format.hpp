//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "checksum.hpp"
#include "config.hpp"
#include "error.hpp"
#include "level.hpp"

#include <micron/types.hpp>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// hsc container
//
// offset  width  field
//   0       4   magic 0x43534889 ("\x89HSC" on disk)
//   4       1   version = 1
//   5       1   flags
//               bit0 exact_pack (v1: must be 1; 0 reserved)
//               bit1 has_gain (must equal mode in {bin, vec})
//               bit2 transform (H*D pre-rotation, codec/rot.hpp; must be 0 for modes 3..5)
//               bits 3..7 reserved zero
//   6       1   mode
//               0 bin | 1 vec | 2 unit | 3 quotient | 4 quat | 5 oct
//   7       1   dim_log2
//               2..6 (the quotient family pins it: mode 3 -> 2, mode 4 -> 3, mode 5 -> 4)
//   8       4   d_q  round(d * 2^24); both sides rebuild every skeleton from this, never raw f64
//   12      1   gain_bits  1..24 (modes 0/1) | 0 (modes 2..5)
//   13      1   profile = 0  (standard construction; paper-SecV ad-hoc codes reserved)
//   14      2   bits_per_block  shape field width = ceil(log2 M)
//   16      8   n_elems  original length (mode 0: bytes; 1-5: f32 count)
//   24      8   skel_guard  M % 2^64
//   32      4   gscale  f32 bit pattern of the gain full-scale (mode 1 only, else 0)
//   36      3   reserved zero
//   39      1   hc = (xxh32(header[0..39)) >> 8) & 0xFF          (LZ4 frame HC-byte pattern)
//   40      ..  payload: one LSB-first bit record per block, [gain][shape],
//               no alignment between blocks, zero pad bits to a whole byte at the very end
//   -8      4   crc32(payload)                                                -> bad_checksum
//   -4      4   nblocks mod 2^32                                              -> bad_length

namespace hsc
{

inline constexpr u32 k_magic = 0x43534889u;
inline constexpr u8 k_version = 1;
inline constexpr usize k_header_size = 40;
inline constexpr usize k_trailer_size = 8;

struct hopf_info {
  mode m = mode::bin;
  u32 dim_log2 = 2;
  u32 d_q = 0;
  u32 gain_bits = 0;
  u32 bits_per_block = 0;
  u32 gscale_bits = 0;
  u64 n_elems = 0;
  u64 nblocks = 0;
  u64 skel_guard = 0;
  bool transform = false;      // flags bit2: H*D pre-rotation applied per block
};

constexpr bool
__has_gain(mode m) noexcept
{
  return m == mode::bin || m == mode::vec;
}

// fibration sets dim_log2 and forbids the pre-rotation; 0 = not in the family
constexpr u32
__fiber_dim_log2(mode m) noexcept
{
  return m == mode::quotient ? 2u : (m == mode::quat ? 3u : (m == mode::oct ? 4u : 0u));
}

constexpr u32
__record_bits(const hopf_info &fi) noexcept
{
  return (__has_gain(fi.m) ? fi.gain_bits : 0) + fi.bits_per_block;
}

constexpr u64
__block_elems(mode m, u32 dim_log2) noexcept
{
  return m == mode::quotient ? 4 : (1ull << dim_log2);
}

constexpr u64
__nblocks(mode m, u32 dim_log2, u64 n_elems) noexcept
{
  const u64 be = __block_elems(m, dim_log2);
  return (n_elems + be - 1) / be;
}

constexpr usize
__payload_bytes(const hopf_info &fi) noexcept
{
  return static_cast<usize>((fi.nblocks * static_cast<u64>(__record_bits(fi)) + 7) / 8);
}

namespace __format
{

constexpr void
write_header(u8 *h, const hopf_info &fi) noexcept
{
  __store32(h + 0, k_magic);
  h[4] = k_version;
  h[5] = static_cast<u8>(0x01u | (__has_gain(fi.m) ? 0x02u : 0x00u) | (fi.transform ? 0x04u : 0x00u));
  h[6] = static_cast<u8>(fi.m);
  h[7] = static_cast<u8>(fi.dim_log2);
  __store32(h + 8, fi.d_q);
  h[12] = static_cast<u8>(fi.gain_bits);
  h[13] = 0;      // profile: standard
  __store16(h + 14, fi.bits_per_block);
  __store64(h + 16, fi.n_elems);
  __store64(h + 24, fi.skel_guard);
  __store32(h + 32, fi.gscale_bits);
  h[36] = h[37] = h[38] = 0;
  h[39] = static_cast<u8>(xxh32(bytes{ h, 39 }) >> 8);
}

constexpr max_t
read_header(bytes in, hopf_info &fi) noexcept
{
  if ( in.size() < k_header_size + k_trailer_size ) [[unlikely]]
    return fail(error::short_input);
  const u8 *h = in.ptr;
  if ( __load32(h) != k_magic ) [[unlikely]]
    return fail(error::bad_container);
  if ( h[4] != k_version ) [[unlikely]]
    return fail(error::unsupported);
  if ( h[39] != static_cast<u8>(xxh32(bytes{ h, 39 }) >> 8) ) [[unlikely]]
    return fail(error::bad_container);
  const u8 flags = h[5];
  if ( flags & 0xF8u ) [[unlikely]]
    return fail(error::bad_container);
  if ( !(flags & 0x01u) ) [[unlikely]]
    return fail(error::unsupported);      // per-node wire profile reserved, not implemented
  if ( h[6] > 5 ) [[unlikely]]
    return fail(error::bad_container);
  fi.m = static_cast<mode>(h[6]);
  if ( static_cast<bool>(flags & 0x02u) != __has_gain(fi.m) ) [[unlikely]]
    return fail(error::bad_container);
  fi.transform = (flags & 0x04u) != 0;
  fi.dim_log2 = h[7];
  if ( fi.dim_log2 < 2 || fi.dim_log2 > 6 ) [[unlikely]]
    return fail(error::bad_container);
  const u32 fd = __fiber_dim_log2(fi.m);
  if ( fd != 0 && (fi.dim_log2 != fd || fi.transform) ) [[unlikely]]
    return fail(error::bad_container);
  fi.d_q = __load32(h + 8);
  if ( !dq_valid(fi.d_q) ) [[unlikely]]
    return fail(error::bad_container);
  fi.gain_bits = h[12];
  if ( __has_gain(fi.m) ? (fi.gain_bits < 1 || fi.gain_bits > 24) : (fi.gain_bits != 0) ) [[unlikely]]
    return fail(error::bad_container);
  if ( h[13] != 0 ) [[unlikely]]
    return fail(error::unsupported);      // construction profile
  fi.bits_per_block = __load16(h + 14);
  fi.n_elems = __load64(h + 16);
  fi.skel_guard = __load64(h + 24);
  fi.gscale_bits = __load32(h + 32);
  if ( h[36] || h[37] || h[38] ) [[unlikely]]
    return fail(error::bad_container);
  if ( fi.m == mode::vec ) {
    const u32 exp = (fi.gscale_bits >> 23) & 0xFFu;      // finite, non-negative f32 required
    if ( (fi.gscale_bits & 0x80000000u) || exp == 0xFFu ) [[unlikely]]
      return fail(error::bad_container);
  } else if ( fi.gscale_bits != 0 ) [[unlikely]]
    return fail(error::bad_container);

  const u64 be = __block_elems(fi.m, fi.dim_log2);
  if ( (fi.m == mode::unit || __fiber_dim_log2(fi.m) != 0) && (fi.n_elems % be) != 0 ) [[unlikely]]
    return fail(error::bad_length);
  fi.nblocks = __nblocks(fi.m, fi.dim_log2, fi.n_elems);

  const u32 rb = __record_bits(fi);
  if ( fi.nblocks != 0 && rb == 0 ) [[unlikely]]
    return fail(error::bad_container);
  if ( rb != 0 && fi.nblocks > (~0ull - 7) / rb ) [[unlikely]]
    return fail(error::bad_length);
  if ( in.size() != k_header_size + __payload_bytes(fi) + k_trailer_size ) [[unlikely]]
    return fail(error::bad_length);
  return 0;
}

constexpr void
write_trailer(u8 *t, u32 crc, u64 nblocks) noexcept
{
  __store32(t, crc);
  __store32(t + 4, static_cast<u32>(nblocks & 0xFFFFFFFFu));
}

constexpr max_t
check_trailer(bytes in, const hopf_info &fi) noexcept
{
  const u8 *t = in.ptr + in.size() - k_trailer_size;
  const bytes payload{ in.ptr + k_header_size, in.size() - k_header_size - k_trailer_size };
  if ( crc32(payload) != __load32(t) ) [[unlikely]]
    return fail(error::bad_checksum);
  if ( static_cast<u32>(fi.nblocks & 0xFFFFFFFFu) != __load32(t + 4) ) [[unlikely]]
    return fail(error::bad_length);
  return 0;
}

};      // namespace __format

constexpr result<hopf_info>
hopf_probe(bytes in) noexcept
{
  hopf_info fi{};
  const max_t r = __format::read_header(in, fi);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  return fi;
}

};      // namespace hsc
