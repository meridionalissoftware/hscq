//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "bits/bitreader.hpp"
#include "codec/block.hpp"
#include "codec/oct.hpp"
#include "codec/pack.hpp"
#include "codec/quotient.hpp"
#include "codec/rot.hpp"
#include "codec/scratch.hpp"
#include "config.hpp"
#include "error.hpp"
#include "format.hpp"

#include <micron/bits.hpp>
#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%
//  hsc decompress

namespace hsc
{

namespace __hopf
{

constexpr bool
__get64(bits::bitreader &r, u32 nbits, u64 &v) noexcept
{
  v = 0;
  for ( u32 pos = 0; pos < nbits; pos += 32 ) {
    const u32 take = nbits - pos < 32 ? nbits - pos : 32;
    if ( !r.need(take) ) [[unlikely]]
      return false;
    v |= static_cast<u64>(r.bits(take)) << pos;
  }
  return true;
}

constexpr void
__seek_bits(bits::bitreader &r, bytes payload, u64 bitoff) noexcept
{
  const usize byteoff = static_cast<usize>(bitoff >> 3);
  r.p = payload.ptr + (byteoff < payload.size() ? byteoff : payload.size());
  r.end = payload.ptr + payload.size();
  r.hold = 0;
  r.nbits = 0;
  const u32 phase = static_cast<u32>(bitoff & 7);
  if ( phase && r.need(phase) ) (void)r.bits(phase);
}

constexpr max_t
__walk_records(bytes payload, const hopf_info &fi, const tree_skeleton &sk, const pack_tables &pt, u64 first, u64 count, bool verify_tail,
               auto &&emit) noexcept
{
  const u32 n = 1u << fi.dim_log2;
  const gain_quant gq{ fi.gain_bits, fi.m == mode::bin ? gq_bin_scale(fi.dim_log2) : __u2f(fi.gscale_bits) };
  bits::bitreader r{ .p = payload.ptr, .end = payload.ptr + payload.size() };
  if ( first ) __seek_bits(r, payload, first * static_cast<u64>(__record_bits(fi)));

  vq_index a;
  pack_scratch ps;
  block_code bc;
  for ( u64 b = first; b < first + count; ++b ) {
    if ( __has_gain(fi.m) ) {
      if ( !r.need(fi.gain_bits) ) [[unlikely]]
        return fail(error::bad_stream);
      bc.gain_q = r.bits(fi.gain_bits);
    } else {
      bc.gain_q = 1;
    }
    if ( !get_wide(r, fi.bits_per_block, a) ) [[unlikely]]
      return fail(error::bad_stream);
    const max_t ur = pack_unrank(sk, pt, a, bc.shape, ps);
    if ( ur < 0 ) [[unlikely]]
      return ur;
    f64 v[64];
    if ( __has_gain(fi.m) ) {
      reconstruct_vec(sk, gq, bc, v);
    } else {
      tree_decode(sk, bc.shape, v);
    }
    if ( fi.transform ) rot_inv(v, fi.dim_log2);
    emit(b, v);
  }
  if ( verify_tail ) {

    const u64 used = fi.nblocks * static_cast<u64>(__record_bits(fi));
    const u64 pad = payload.size() * 8 - used;
    if ( pad >= 8 ) [[unlikely]]
      return fail(error::bad_stream);
    if ( pad ) {
      if ( !r.need(static_cast<u32>(pad)) || r.bits(static_cast<u32>(pad)) != 0 ) [[unlikely]]
        return fail(error::bad_stream);
    }
    if ( r.need(1) ) [[unlikely]]
      return fail(error::bad_stream);
  }
  (void)n;
  return 0;
}

constexpr max_t
__decode_records(bytes payload, const hopf_info &fi, const tree_skeleton &sk, const pack_tables &pt, auto &&emit) noexcept
{
  return __walk_records(payload, fi, sk, pt, 0, fi.nblocks, true, emit);
}

constexpr max_t
__walk_flat(bytes payload, const hopf_info &fi, u64 first, u64 count, bool verify_tail, u32 blkw, f32 *out, auto &&decode_one) noexcept
{
  bits::bitreader r{ .p = payload.ptr, .end = payload.ptr + payload.size() };
  if ( first ) __seek_bits(r, payload, first * static_cast<u64>(fi.bits_per_block));
  for ( u64 b = first; b < first + count; ++b ) {
    const max_t rr = decode_one(r, out + (b - first) * blkw);
    if ( rr < 0 ) [[unlikely]]
      return rr;
  }
  if ( verify_tail ) {
    const u64 used = fi.nblocks * static_cast<u64>(fi.bits_per_block);
    const u64 pad = payload.size() * 8 - used;
    if ( pad >= 8 ) [[unlikely]]
      return fail(error::bad_stream);
    if ( pad ) {
      if ( !r.need(static_cast<u32>(pad)) || r.bits(static_cast<u32>(pad)) != 0 ) [[unlikely]]
        return fail(error::bad_stream);
    }
    if ( r.need(1) ) [[unlikely]]
      return fail(error::bad_stream);
  }
  return 0;
}

constexpr max_t
__walk_quot(bytes payload, const hopf_info &fi, const s2_skeleton &s2, u64 first, u64 count, bool verify_tail, f32 *out) noexcept
{
  return __walk_flat(payload, fi, first, count, verify_tail, 4, out, [&](bits::bitreader &r, f32 *dst) -> max_t {
    u64 a = 0;
    if ( !__get64(r, fi.bits_per_block, a) ) [[unlikely]]
      return fail(error::bad_stream);
    if ( a >= s2.m_total ) [[unlikely]]
      return fail(error::bad_stream);
    quotient_reconstruct(s2, a, dst);
    return 0;
  });
}

constexpr max_t
__walk_quat(bytes payload, const hopf_info &fi, const susp_skeleton &ss, const tree_skeleton &tv, u64 first, u64 count, bool verify_tail,
            f32 *out) noexcept
{
  return __walk_flat(payload, fi, first, count, verify_tail, 8, out, [&](bits::bitreader &r, f32 *dst) -> max_t {
    u64 a = 0;
    if ( !__get64(r, fi.bits_per_block, a) ) [[unlikely]]
      return fail(error::bad_stream);
    if ( a >= ss.m_mod ) [[unlikely]]      //  exact radix check: the S^4 suspension M fits u64
      return fail(error::bad_stream);
    quat_reconstruct(ss, tv, a, dst);
    return 0;
  });
}

constexpr max_t
__walk_oct(bytes payload, const hopf_info &fi, const susp_skeleton &ss, const tree_skeleton &tv, const pack_tables &pt, const susp_pack &sp,
           u64 first, u64 count, bool verify_tail, f32 *out) noexcept
{
  //  hoisted out of the per-record lambda (bounded arbint ctors zero every limb)
  vq_index a;
  pack_scratch ps;
  tree_fields f;
  return __walk_flat(payload, fi, first, count, verify_tail, 16, out, [&](bits::bitreader &r, f32 *dst) -> max_t {
    if ( !get_wide(r, fi.bits_per_block, a) ) [[unlikely]]
      return fail(error::bad_stream);
    u32 band = 0;
    const max_t ur = susp_unrank(ss, tv, pt, sp, a, band, f, ps);      //  rejects a >= M -> bad_stream
    if ( ur < 0 ) [[unlikely]]
      return ur;
    oct_reconstruct(ss, tv, band, f, dst);
    return 0;
  });
}

constexpr max_t
quot_decode(bytes payload, const hopf_info &fi, const s2_skeleton &s2, f32 *out) noexcept
{
  return __walk_quot(payload, fi, s2, 0, fi.nblocks, true, out);
}

};      //  namespace __hopf

constexpr usize
__out_bytes(const hopf_info &fi) noexcept
{
  return static_cast<usize>(fi.m == mode::bin ? fi.n_elems : fi.n_elems * 4);
}

namespace __hopf
{

constexpr max_t
__guard_tree(const tree_skeleton &sk, const pack_tables &pt, const hopf_info &fi) noexcept
{
  if ( tree_m_mod64(sk) != fi.skel_guard || shape_bits(sk, pt) != fi.bits_per_block ) [[unlikely]]
    return fail(error::bad_skeleton);
  return 0;
}

constexpr max_t
__guard_s2(const s2_skeleton &s2, const hopf_info &fi) noexcept
{
  u64 mm1 = s2.m_total - 1;
  if ( s2.m_total != fi.skel_guard || static_cast<u32>(micron::bit_width(mm1)) != fi.bits_per_block ) [[unlikely]]
    return fail(error::bad_skeleton);
  return 0;
}

constexpr max_t
__guard_quat(const susp_skeleton &ss, const hopf_info &fi) noexcept
{
  u64 mm1 = ss.m_mod - 1;
  if ( ss.m_mod != fi.skel_guard || static_cast<u32>(micron::bit_width(mm1)) != fi.bits_per_block ) [[unlikely]]
    return fail(error::bad_skeleton);
  return 0;
}

constexpr max_t
__guard_oct(const susp_skeleton &ss, const susp_pack &sp, const hopf_info &fi) noexcept
{
  if ( ss.m_mod != fi.skel_guard || susp_bits(sp) != fi.bits_per_block ) [[unlikely]]
    return fail(error::bad_skeleton);
  return 0;
}

constexpr max_t
decode_core(bytes payload, const hopf_info &fi, const tree_skeleton *sk, const pack_tables *pt, const s2_skeleton *s2,
            const susp_skeleton *ss, const susp_pack *sp, u8 *outb, f32 *outf) noexcept
{
  if ( fi.m == mode::quotient ) {
    const max_t g = __guard_s2(*s2, fi);
    if ( g < 0 ) [[unlikely]]
      return g;
    const max_t r = quot_decode(payload, fi, *s2, outf);
    if ( r < 0 ) [[unlikely]]
      return r;
    return static_cast<max_t>(__out_bytes(fi));
  }
  if ( fi.m == mode::quat ) {
    const max_t g = __guard_quat(*ss, fi);
    if ( g < 0 ) [[unlikely]]
      return g;
    const max_t r = __walk_quat(payload, fi, *ss, *sk, 0, fi.nblocks, true, outf);
    if ( r < 0 ) [[unlikely]]
      return r;
    return static_cast<max_t>(__out_bytes(fi));
  }
  if ( fi.m == mode::oct ) {
    const max_t g = __guard_oct(*ss, *sp, fi);
    if ( g < 0 ) [[unlikely]]
      return g;
    const max_t r = __walk_oct(payload, fi, *ss, *sk, *pt, *sp, 0, fi.nblocks, true, outf);
    if ( r < 0 ) [[unlikely]]
      return r;
    return static_cast<max_t>(__out_bytes(fi));
  }

  const max_t g = __guard_tree(*sk, *pt, fi);
  if ( g < 0 ) [[unlikely]]
    return g;

  const u32 n = 1u << fi.dim_log2;
  if ( fi.m == mode::bin ) {
    const max_t dr = __decode_records(payload, fi, *sk, *pt, [&](u64 b, const f64 *v) {
      const u64 base = b * n;
      for ( u32 c = 0; c < n && base + c < fi.n_elems; ++c ) {
        const f64 u = __round(v[c] + 127.5);
        outb[base + c] = static_cast<u8>(u < 0.0 ? 0.0 : (u > 255.0 ? 255.0 : u));
      }
    });
    if ( dr < 0 ) [[unlikely]]
      return dr;
    return static_cast<max_t>(__out_bytes(fi));
  }

  const max_t dr = __decode_records(payload, fi, *sk, *pt, [&](u64 b, const f64 *v) {
    const u64 base = b * n;
    for ( u32 c = 0; c < n && base + c < fi.n_elems; ++c ) outf[base + c] = static_cast<f32>(v[c]);
  });
  if ( dr < 0 ) [[unlikely]]
    return dr;
  return static_cast<max_t>(__out_bytes(fi));
}

constexpr u64
__range_elems(const hopf_info &fi, u64 first, u64 count) noexcept
{
  const u64 be = __block_elems(fi.m, fi.dim_log2);
  const u64 base = first * be;
  if ( base >= fi.n_elems ) return 0;
  const u64 want = count * be;
  const u64 avail = fi.n_elems - base;
  return want < avail ? want : avail;
}

constexpr max_t
decode_range_core(bytes payload, const hopf_info &fi, const tree_skeleton *sk, const pack_tables *pt, const s2_skeleton *s2,
                  const susp_skeleton *ss, const susp_pack *sp, u64 first, u64 count, u8 *outb, f32 *outf) noexcept
{
  const u64 wrote = __range_elems(fi, first, count);
  if ( fi.m == mode::quotient ) {
    const max_t g = __guard_s2(*s2, fi);
    if ( g < 0 ) [[unlikely]]
      return g;
    const max_t r = __walk_quot(payload, fi, *s2, first, count, false, outf);
    if ( r < 0 ) [[unlikely]]
      return r;
    return static_cast<max_t>(wrote);
  }
  if ( fi.m == mode::quat ) {
    const max_t g = __guard_quat(*ss, fi);
    if ( g < 0 ) [[unlikely]]
      return g;
    const max_t r = __walk_quat(payload, fi, *ss, *sk, first, count, false, outf);
    if ( r < 0 ) [[unlikely]]
      return r;
    return static_cast<max_t>(wrote);
  }
  if ( fi.m == mode::oct ) {
    const max_t g = __guard_oct(*ss, *sp, fi);
    if ( g < 0 ) [[unlikely]]
      return g;
    const max_t r = __walk_oct(payload, fi, *ss, *sk, *pt, *sp, first, count, false, outf);
    if ( r < 0 ) [[unlikely]]
      return r;
    return static_cast<max_t>(wrote);
  }

  const max_t g = __guard_tree(*sk, *pt, fi);
  if ( g < 0 ) [[unlikely]]
    return g;

  const u32 n = 1u << fi.dim_log2;
  const u64 skip = first * n;
  if ( fi.m == mode::bin ) {
    const max_t dr = __walk_records(payload, fi, *sk, *pt, first, count, false, [&](u64 b, const f64 *v) {
      const u64 base = b * n;
      for ( u32 c = 0; c < n && base + c < fi.n_elems; ++c ) {
        const f64 u = __round(v[c] + 127.5);
        outb[base + c - skip] = static_cast<u8>(u < 0.0 ? 0.0 : (u > 255.0 ? 255.0 : u));
      }
    });
    if ( dr < 0 ) [[unlikely]]
      return dr;
    return static_cast<max_t>(wrote);
  }

  const max_t dr = __walk_records(payload, fi, *sk, *pt, first, count, false, [&](u64 b, const f64 *v) {
    const u64 base = b * n;
    for ( u32 c = 0; c < n && base + c < fi.n_elems; ++c ) outf[base + c - skip] = static_cast<f32>(v[c]);
  });
  if ( dr < 0 ) [[unlikely]]
    return dr;
  return static_cast<max_t>(wrote);
}

inline max_t
unhopf_range_run(bytes in, u64 first, u64 count, wbytes out, hopf_scratch &sc) noexcept
{
  hopf_info fi{};
  const max_t r = __format::read_header(in, fi);
  if ( r < 0 ) [[unlikely]]
    return r;
  if ( first > fi.nblocks || count > fi.nblocks - first ) [[unlikely]]
    return fail(error::bad_length);
  if ( count == 0 ) return 0;
  const u64 elems = __range_elems(fi, first, count);
  const usize need = static_cast<usize>(fi.m == mode::bin ? elems : elems * 4);
  if ( out.size() < need ) [[unlikely]]
    return fail(error::short_output);
  const bytes payload{ in.ptr + k_header_size, in.size() - k_header_size - k_trailer_size };

  if ( fi.m == mode::quotient ) {
    if ( sc.build_s2(fi.d_q) < 0 ) [[unlikely]]
      return fail(error::oom);
    return decode_range_core(payload, fi, nullptr, nullptr, &sc.s2, nullptr, nullptr, first, count, nullptr,
                             reinterpret_cast<f32 *>(out.ptr));
  }
  if ( fi.m == mode::quat || fi.m == mode::oct ) {
    if ( sc.build_susp(fi.m == mode::quat ? 2 : 3, fi.d_q) < 0 ) [[unlikely]]
      return fail(error::oom);
    return decode_range_core(payload, fi, &sc.sk, &sc.pt, nullptr, &sc.ss, &sc.sp, first, count, nullptr, reinterpret_cast<f32 *>(out.ptr));
  }
  if ( sc.build_tree(fi.dim_log2, fi.d_q) < 0 ) [[unlikely]]
    return fail(error::oom);
  return decode_range_core(payload, fi, &sc.sk, &sc.pt, nullptr, nullptr, nullptr, first, count, out.ptr, reinterpret_cast<f32 *>(out.ptr));
}

inline max_t
unhopf_run(bytes in, wbytes out, hopf_scratch &sc) noexcept
{
  hopf_info fi{};
  max_t r = __format::read_header(in, fi);
  if ( r < 0 ) [[unlikely]]
    return r;
  r = __format::check_trailer(in, fi);
  if ( r < 0 ) [[unlikely]]
    return r;
  if ( out.size() < __out_bytes(fi) ) [[unlikely]]
    return fail(error::short_output);
  const bytes payload{ in.ptr + k_header_size, in.size() - k_header_size - k_trailer_size };

  if ( fi.m == mode::quotient ) {
    if ( sc.build_s2(fi.d_q) < 0 ) [[unlikely]]
      return fail(error::oom);
    return decode_core(payload, fi, nullptr, nullptr, &sc.s2, nullptr, nullptr, nullptr, reinterpret_cast<f32 *>(out.ptr));
  }
  if ( fi.m == mode::quat || fi.m == mode::oct ) {
    if ( sc.build_susp(fi.m == mode::quat ? 2 : 3, fi.d_q) < 0 ) [[unlikely]]
      return fail(error::oom);
    return decode_core(payload, fi, &sc.sk, &sc.pt, nullptr, &sc.ss, &sc.sp, nullptr, reinterpret_cast<f32 *>(out.ptr));
  }
  if ( sc.build_tree(fi.dim_log2, fi.d_q) < 0 ) [[unlikely]]
    return fail(error::oom);
  return decode_core(payload, fi, &sc.sk, &sc.pt, nullptr, nullptr, nullptr, out.ptr, reinterpret_cast<f32 *>(out.ptr));
}

};      //  namespace __hopf

inline result<usize>
unhopf(bytes in, wbytes out, hopf_scratch &sc) noexcept
{
  const max_t r = __hopf::unhopf_run(in, out, sc);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  return static_cast<usize>(r);
}

inline result<usize>
unhopf(bytes in, wbytes out) noexcept
{
  hopf_scratch sc;
  return unhopf(in, out, sc);
}

inline result<fhsc>
unhopf(bytes in, hopf_scratch &sc) noexcept
{
  hopf_info fi{};
  const max_t hr = __format::read_header(in, fi);
  if ( hr < 0 ) [[unlikely]]
    return as_error(hr);
  fhsc out(fhsc::__uninit_t{}, __out_bytes(fi) + 1);
  const max_t r = __hopf::unhopf_run(in, wbytes{ out.first(), __out_bytes(fi) }, sc);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  out.mark(static_cast<usize>(r));
  return out;
}

inline result<fhsc>
unhopf(bytes in) noexcept
{
  hopf_scratch sc;
  return unhopf(in, sc);
}

inline result<usize>
unhopf(bytes in, wfloats out, hopf_scratch &sc) noexcept
{
  hopf_info fi{};
  const max_t hr = __format::read_header(in, fi);
  if ( hr < 0 ) [[unlikely]]
    return as_error(hr);
  if ( fi.m == mode::bin ) [[unlikely]]
    return error::bad_opts;
  const max_t r = __hopf::unhopf_run(in, wbytes{ reinterpret_cast<u8 *>(out.ptr), out.size() * 4 }, sc);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  return static_cast<usize>(r) / 4;
}

inline result<usize>
unhopf(bytes in, wfloats out) noexcept
{
  hopf_scratch sc;
  return unhopf(in, out, sc);
}

inline result<fhsc32>
unhopf_f32(bytes in) noexcept
{
  hopf_info fi{};
  const max_t hr = __format::read_header(in, fi);
  if ( hr < 0 ) [[unlikely]]
    return as_error(hr);
  if ( fi.m == mode::bin ) [[unlikely]]
    return error::bad_opts;
  fhsc32 out(fhsc32::__uninit_t{}, static_cast<usize>(fi.n_elems) + 1);
  hopf_scratch sc;
  const max_t r = __hopf::unhopf_run(in, wbytes{ reinterpret_cast<u8 *>(out.first()), fi.n_elems * 4 }, sc);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  out.mark(static_cast<usize>(r) / 4);
  return out;
}

inline result<usize>
unhopf_range(bytes in, u64 first, u64 count, wbytes out, hopf_scratch &sc) noexcept
{
  const max_t r = __hopf::unhopf_range_run(in, first, count, out, sc);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  return static_cast<usize>(r);
}

inline result<usize>
unhopf_range(bytes in, u64 first, u64 count, wfloats out, hopf_scratch &sc) noexcept
{
  hopf_info fi{};
  const max_t hr = __format::read_header(in, fi);
  if ( hr < 0 ) [[unlikely]]
    return as_error(hr);
  if ( fi.m == mode::bin ) [[unlikely]]
    return error::bad_opts;
  return unhopf_range(in, first, count, wbytes{ reinterpret_cast<u8 *>(out.ptr), out.size() * 4 }, sc);
}

inline result<hopf_info>
verify(bytes in) noexcept
{
  hopf_info fi{};
  max_t r = __format::read_header(in, fi);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  r = __format::check_trailer(in, fi);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  return fi;
}

template<byte_source C>
inline result<fhsc>
unhopf(const C &in) noexcept
{
  return unhopf(as_bytes(in));
}

template<byte_sink W>
inline result<usize>
unhopf(bytes in, W &out) noexcept
{
  if constexpr ( f32_sink<W> )
    return unhopf(in, as_wfloats(out));
  else
    return unhopf(in, as_wbytes(out));
}

template<byte_source C, byte_sink W>
inline result<usize>
unhopf(const C &in, W &out) noexcept
{
  return unhopf(as_bytes(in), out);
}

};      //  namespace hsc
