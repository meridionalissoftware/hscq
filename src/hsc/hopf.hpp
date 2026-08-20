//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "bits/bitwriter.hpp"
#include "codec/block.hpp"
#include "codec/oct.hpp"
#include "codec/pack.hpp"
#include "codec/quotient.hpp"
#include "codec/rot.hpp"
#include "codec/scratch.hpp"
#include "config.hpp"
#include "error.hpp"
#include "format.hpp"
#include "level.hpp"

#include <micron/bits.hpp>
#include <micron/string/strings.hpp>
#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%
//  hsc compress
//
//  bytes in (mode bin) cannot fail:
//  options are clamped, any byte pattern quantizes, and the owned wrappers allocate the exactly known stream size

namespace hsc
{

namespace __hopf
{

struct enc_params {
  mode m = mode::bin;
  u32 dim_log2 = 3;
  u32 dq = 0;
  u32 gain_bits = 8;
  u32 refine = 0;
  bool transform = false;
};

//  clamp, never reject
constexpr enc_params
resolve(const hopf_opts &o, mode m) noexcept
{
  enc_params p{};
  p.m = m;
  const u32 fd = __fiber_dim_log2(m);
  p.dim_log2 = fd != 0 ? fd : (o.dim_log2 < 2 ? 2 : (o.dim_log2 > 6 ? 6 : o.dim_log2));
  u32 dq = opts_dq(o);
  p.dq = dq < dq_min ? dq_min : (dq > dq_max ? dq_max : dq);
  p.gain_bits = __has_gain(m) ? (o.gain_bits < 1 ? 1 : (o.gain_bits > 24 ? 24 : o.gain_bits)) : 0;
  p.refine = o.refine > 1 ? 1 : o.refine;
  p.transform = o.transform && fd == 0;
  return p;
}

constexpr void
__put64(bits::bitwriter &w, u64 v, u32 nbits) noexcept
{
  for ( u32 pos = 0; pos < nbits; pos += 32 ) {
    const u32 take = nbits - pos < 32 ? nbits - pos : 32;
    u32 chunk = static_cast<u32>(v >> pos);
    if ( take < 32 ) chunk &= (1u << take) - 1;
    w.add(chunk, static_cast<i32>(take));
    w.flush();
  }
}

//  header + payload + trailer
constexpr usize
__emit(const hopf_info &fi, u8 *buf, auto &&record) noexcept
{
  __format::write_header(buf, fi);
  const usize pbytes = __payload_bytes(fi);
  u8 *pay = buf + k_header_size;
  bits::bitwriter w{ .acc = 0, .cnt = 0, .out = pay, .fast_end = pbytes > 8 ? pay + pbytes - 8 : pay };
  for ( u64 b = 0; b < fi.nblocks; ++b ) record(w, b);
  w.finish();
  __format::write_trailer(pay + pbytes, crc32(bytes{ pay, pbytes }), fi.nblocks);
  return k_header_size + pbytes + k_trailer_size;
}

//  bytes -> centered blocks
constexpr usize
bin_into(bytes in, const enc_params &p, const tree_skeleton &sk, const pack_tables &pt, u8 *buf) noexcept
{
  const u32 n = 1u << p.dim_log2;
  hopf_info fi{ mode::bin,          p.dim_log2, p.dq,      p.gain_bits,
                shape_bits(sk, pt), 0,          in.size(), __nblocks(mode::bin, p.dim_log2, in.size()),
                tree_m_mod64(sk) };
  fi.transform = p.transform;
  const gain_quant gq{ p.gain_bits, gq_bin_scale(p.dim_log2) };
  //  hoisted out of the per-block lambda
  vq_index a;
  pack_scratch ps;
  block_code bc;
  return __emit(fi, buf, [&](bits::bitwriter &w, u64 b) {
    f64 v[64];
    for ( u32 c = 0; c < n; ++c ) {
      const u64 idx = b * n + c;
      v[c] = idx < fi.n_elems ? static_cast<f64>(in.ptr[idx]) - 127.5 : 0.0;
    }
    if ( p.transform ) rot_fwd(v, p.dim_log2);
    quantize_vec(sk, gq, v, bc, p.refine);
    w.add(bc.gain_q, static_cast<i32>(p.gain_bits));
    w.flush();
    pack_rank(sk, pt, bc.shape, a, ps);
    put_wide(w, a, fi.bits_per_block);
  });
}

constexpr max_t
f32_into(floats in, const enc_params &p, const tree_skeleton &sk, const pack_tables &pt, u8 *buf) noexcept
{
  const u32 n = 1u << p.dim_log2;
  const bool gained = p.m == mode::vec;
  if ( !gained && (in.size() % n) != 0 ) [[unlikely]]
    return fail(error::bad_length);

  f32 scale = 0;
  if ( gained ) {
    f64 gmax = 0;
    const u64 nb = __nblocks(mode::vec, p.dim_log2, in.size());
    for ( u64 b = 0; b < nb; ++b ) {
      f64 s = 0;
      for ( u32 c = 0; c < n; ++c ) {
        const u64 idx = b * n + c;
        const f64 e = idx < in.size() ? static_cast<f64>(in.ptr[idx]) : 0.0;
        if ( micron::math::isnan(e) || micron::math::isinf(e) ) [[unlikely]]
          return fail(error::bad_value);
        s += e * e;
      }
      const f64 g = __sqrt(s);
      if ( g > gmax ) gmax = g;
    }
    scale = static_cast<f32>(gmax);
  }

  hopf_info fi{ p.m,
                p.dim_log2,
                p.dq,
                p.gain_bits,
                shape_bits(sk, pt),
                gained ? __f2u(scale) : 0,
                in.size(),
                __nblocks(p.m, p.dim_log2, in.size()),
                tree_m_mod64(sk) };
  fi.transform = p.transform;
  const gain_quant gq{ p.gain_bits, scale };
  max_t bad = 0;
  vq_index a;
  pack_scratch ps;
  block_code bc;
  const usize written = __emit(fi, buf, [&](bits::bitwriter &w, u64 b) {
    f64 v[64];
    f64 s = 0;
    for ( u32 c = 0; c < n; ++c ) {
      const u64 idx = b * n + c;
      const f64 e = idx < in.size() ? static_cast<f64>(in.ptr[idx]) : 0.0;
      if ( micron::math::isnan(e) || micron::math::isinf(e) ) [[unlikely]] {
        bad = fail(error::bad_value);
        return;
      }
      v[c] = e;
      s += e * e;
    }
    if ( p.transform ) rot_fwd(v, p.dim_log2);
    if ( gained ) {
      quantize_vec(sk, gq, v, bc, p.refine);
      w.add(bc.gain_q, static_cast<i32>(p.gain_bits));
      w.flush();
    } else {
      if ( !(s > 0.0) ) [[unlikely]] {
        bad = fail(error::bad_value);
        return;
      }
      tree_quantize(sk, v, bc.shape, p.refine);
    }
    pack_rank(sk, pt, bc.shape, a, ps);
    put_wide(w, a, fi.bits_per_block);
  });
  if ( bad < 0 ) [[unlikely]]
    return bad;
  return static_cast<max_t>(written);
}

constexpr max_t
quot_into(floats in, const enc_params &p, const s2_skeleton &s2, u8 *buf) noexcept
{
  if ( (in.size() % 4) != 0 ) [[unlikely]]
    return fail(error::bad_length);
  u64 mm1 = s2.m_total - 1;
  hopf_info fi{ mode::quotient, 2, p.dq, 0, static_cast<u32>(micron::bit_width(mm1)), 0, in.size(), __nblocks(mode::quotient, 2, in.size()),
                s2.m_total };
  max_t bad = 0;
  const usize written = __emit(fi, buf, [&](bits::bitwriter &w, u64 b) {
    f32 z[4];
    for ( u32 c = 0; c < 4; ++c ) z[c] = in.ptr[b * 4 + c];
    u64 a = 0;
    const max_t r = quotient_quantize(s2, z, a);
    if ( r < 0 ) [[unlikely]] {
      bad = r;
      return;
    }
    __put64(w, a, fi.bits_per_block);
  });
  if ( bad < 0 ) [[unlikely]]
    return bad;
  return static_cast<max_t>(written);
}

constexpr max_t
quat_into(floats in, const enc_params &p, const susp_skeleton &ss, const tree_skeleton &tv, u8 *buf) noexcept
{
  if ( (in.size() % 8) != 0 ) [[unlikely]]
    return fail(error::bad_length);
  u64 mm1 = ss.m_mod - 1;
  hopf_info fi{ mode::quat, 3, p.dq, 0, static_cast<u32>(micron::bit_width(mm1)), 0, in.size(), __nblocks(mode::quat, 3, in.size()),
                ss.m_mod };
  max_t bad = 0;
  const usize written = __emit(fi, buf, [&](bits::bitwriter &w, u64 b) {
    u64 a = 0;
    const max_t r = quat_quantize(ss, tv, in.ptr + b * 8, a, p.refine);
    if ( r < 0 ) [[unlikely]] {
      bad = r;
      return;
    }
    __put64(w, a, fi.bits_per_block);
  });
  if ( bad < 0 ) [[unlikely]]
    return bad;
  return static_cast<max_t>(written);
}

constexpr max_t
oct_into(floats in, const enc_params &p, const susp_skeleton &ss, const tree_skeleton &tv, const pack_tables &pt, const susp_pack &sp,
         u8 *buf) noexcept
{
  if ( (in.size() % 16) != 0 ) [[unlikely]]
    return fail(error::bad_length);
  hopf_info fi{ mode::oct, 4, p.dq, 0, susp_bits(sp), 0, in.size(), __nblocks(mode::oct, 4, in.size()), ss.m_mod };
  max_t bad = 0;
  //  hoisted out of the per-block lambda (the v3 arbint lesson: bounded ctors zero every limb)
  vq_index a;
  pack_scratch ps;
  tree_fields f;
  const usize written = __emit(fi, buf, [&](bits::bitwriter &w, u64 b) {
    u32 band = 0;
    const max_t r = oct_quantize(ss, tv, in.ptr + b * 16, band, f, p.refine);
    if ( r < 0 ) [[unlikely]] {
      bad = r;
      return;
    }
    susp_rank(ss, tv, pt, sp, band, f, a, ps);
    put_wide(w, a, fi.bits_per_block);
  });
  if ( bad < 0 ) [[unlikely]]
    return bad;
  return static_cast<max_t>(written);
}

};      //  namespace __hopf

//  %%%%%%%%%%%%%%%%%%%%%%%
//  bound

constexpr u32
__shape_bits_ub(u32 dim_log2, u32 dq) noexcept
{
  const u32 n = 1u << dim_log2;
  u32 lg = 1;
  while ( (dq_max >> lg) > dq ) ++lg;      //  ceil(log2(2 / d_eff)) + 1 envelope
  return (n - 1) * (lg + 1) + n;
}

//  suspension envelope for the quat/oct bases
//  M <= T * max_band M_child, every band child snaps up (d_child >= d)
//  M_child(band) <= M_child(d).  for d <= sqrt(2): T <= pi/(2 asin(d/2)) + 3 <= 8/d = 4 * (2/d)
//  (the middle inequality fails on d in (~1.843, 2), but there the band window w = pi - 2*dth is
//  already negative and T collapses to the 2 poles, well under the envelope);
//  hence bits <= (lg + 2) + child bits; generous overestimate on purpose
constexpr u32
__susp_bits_ub(u32 child_dim_log2, u32 dq) noexcept
{
  u32 lg = 1;
  while ( (dq_max >> lg) > dq ) ++lg;
  return (lg + 2) + __shape_bits_ub(child_dim_log2, dq);
}

constexpr usize
bound(usize n_elems, const hopf_opts &o = {}) noexcept
{
  const __hopf::enc_params p = __hopf::resolve(o, o.m);
  const u64 nb = __nblocks(p.m, p.dim_log2, n_elems);
  const u32 fd = __fiber_dim_log2(p.m);
  const u64 sb = (p.m == mode::quat || p.m == mode::oct) ? __susp_bits_ub(fd - 1, p.dq) : __shape_bits_ub(p.dim_log2, p.dq);
  const u64 rb = static_cast<u64>(p.gain_bits) + sb;
  return k_header_size + static_cast<usize>((nb * rb + 7) / 8) + k_trailer_size;
}

inline usize
bound(usize n_elems, const hopf_opts &o, hopf_scratch &sc)
{
  const __hopf::enc_params p = __hopf::resolve(o, o.m);
  if ( p.m == mode::quotient ) {
    if ( sc.build_s2(p.dq) < 0 ) return bound(n_elems, o);
    u64 mm1 = sc.s2.m_total - 1;
    const u64 rb = micron::bit_width(mm1);
    return k_header_size + static_cast<usize>((__nblocks(p.m, 2, n_elems) * rb + 7) / 8) + k_trailer_size;
  }
  if ( p.m == mode::quat || p.m == mode::oct ) {
    if ( sc.build_susp(p.m == mode::quat ? 2 : 3, p.dq) < 0 ) return bound(n_elems, o);
    u64 rb = 0;
    if ( p.m == mode::quat ) {
      u64 mm1 = sc.ss.m_mod - 1;
      rb = micron::bit_width(mm1);
    } else
      rb = susp_bits(sc.sp);
    return k_header_size + static_cast<usize>((__nblocks(p.m, p.dim_log2, n_elems) * rb + 7) / 8) + k_trailer_size;
  }
  if ( sc.build_tree(p.dim_log2, p.dq) < 0 ) return bound(n_elems, o);
  hopf_info fi{ p.m, p.dim_log2, p.dq, p.gain_bits, shape_bits(sc.sk, sc.pt), 0, n_elems, __nblocks(p.m, p.dim_log2, n_elems), 0 };
  return k_header_size + __payload_bytes(fi) + k_trailer_size;
}

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  compress

inline usize
hopf_into(bytes in, const hopf_opts &o, u8 *buf, usize cap, hopf_scratch &sc) noexcept
{
  const __hopf::enc_params p = __hopf::resolve(o, mode::bin);
  if ( sc.build_tree(p.dim_log2, p.dq) < 0 ) [[unlikely]]
    return 0;
  hopf_info fi{
    mode::bin, p.dim_log2, p.dq, p.gain_bits, shape_bits(sc.sk, sc.pt), 0, in.size(), __nblocks(mode::bin, p.dim_log2, in.size()), 0
  };
  if ( cap < k_header_size + __payload_bytes(fi) + k_trailer_size ) [[unlikely]]
    return 0;
  return __hopf::bin_into(in, p, sc.sk, sc.pt, buf);
}

inline fhsc
hopf(bytes in, const hopf_opts &o, hopf_scratch &sc)
{
  const usize cap = bound(in.size(), o, sc);
  fhsc out(fhsc::__uninit_t{}, cap);
  out.mark(hopf_into(in, o, out.first(), cap, sc));
  return out;
}

inline fhsc
hopf(bytes in, const hopf_opts &o = {})
{
  hopf_scratch sc;
  return hopf(in, o, sc);
}

inline fhsc
hopf(bytes in, i32 level)
{
  return hopf(in, hopf_opts{ .level = level });
}

inline micron::string
hopf_str(bytes in, const hopf_opts &o, hopf_scratch &sc)
{
  micron::string out{};
  out.reserve(bound(in.size(), o, sc) + 1);
  out.set_size(hopf_into(in, o, reinterpret_cast<u8 *>(out.data()), out.max_size(), sc));
  return out;
}

inline micron::string
hopf_str(bytes in, const hopf_opts &o = {})
{
  hopf_scratch sc;
  return hopf_str(in, o, sc);
}

template<byte_source C>
inline fhsc
hopf(const C &in, const hopf_opts &o, hopf_scratch &sc)
{
  return hopf(as_bytes(in), o, sc);
}

template<byte_source C>
inline fhsc
hopf(const C &in, const hopf_opts &o = {})
{
  return hopf(as_bytes(in), o);
}

template<byte_source C>
inline micron::string
hopf_str(const C &in, const hopf_opts &o = {})
{
  return hopf_str(as_bytes(in), o);
}

inline max_t
hopf_into(floats in, const hopf_opts &o, u8 *buf, usize cap, hopf_scratch &sc) noexcept
{
  const mode m = o.m == mode::bin ? mode::vec : o.m;      //  floats never run the byte mode
  const __hopf::enc_params p = __hopf::resolve(o, m);
  if ( p.m == mode::quotient ) {
    if ( sc.build_s2(p.dq) < 0 ) [[unlikely]]
      return fail(error::oom);
    u64 mm1 = sc.s2.m_total - 1;
    const u64 rb = micron::bit_width(mm1);
    const u64 nb = __nblocks(mode::quotient, 2, in.size());
    if ( cap < k_header_size + static_cast<usize>((nb * rb + 7) / 8) + k_trailer_size ) [[unlikely]]
      return fail(error::short_output);
    return __hopf::quot_into(in, p, sc.s2, buf);
  }
  if ( p.m == mode::quat || p.m == mode::oct ) {
    if ( sc.build_susp(p.m == mode::quat ? 2 : 3, p.dq) < 0 ) [[unlikely]]
      return fail(error::oom);
    u64 rb = 0;
    if ( p.m == mode::quat ) {
      u64 mm1 = sc.ss.m_mod - 1;
      rb = micron::bit_width(mm1);
    } else
      rb = susp_bits(sc.sp);
    const u64 nb = __nblocks(p.m, p.dim_log2, in.size());
    if ( cap < k_header_size + static_cast<usize>((nb * rb + 7) / 8) + k_trailer_size ) [[unlikely]]
      return fail(error::short_output);
    return p.m == mode::quat ? __hopf::quat_into(in, p, sc.ss, sc.sk, buf) : __hopf::oct_into(in, p, sc.ss, sc.sk, sc.pt, sc.sp, buf);
  }
  if ( sc.build_tree(p.dim_log2, p.dq) < 0 ) [[unlikely]]
    return fail(error::oom);
  hopf_info fi{ p.m, p.dim_log2, p.dq, p.gain_bits, shape_bits(sc.sk, sc.pt), 0, in.size(), __nblocks(p.m, p.dim_log2, in.size()), 0 };
  if ( cap < k_header_size + __payload_bytes(fi) + k_trailer_size ) [[unlikely]]
    return fail(error::short_output);
  return __hopf::f32_into(in, p, sc.sk, sc.pt, buf);
}

inline result<fhsc>
hopf(floats in, const hopf_opts &o, hopf_scratch &sc) noexcept
{
  const usize cap = bound(in.size(), o, sc);
  fhsc out(fhsc::__uninit_t{}, cap);
  const max_t r = hopf_into(in, o, out.first(), cap, sc);
  if ( r < 0 ) [[unlikely]]
    return as_error(r);
  out.mark(static_cast<usize>(r));
  return out;
}

inline result<fhsc>
hopf(floats in, const hopf_opts &o = {}) noexcept
{
  hopf_scratch sc;
  return hopf(in, o, sc);
}

template<f32_source C>
inline result<fhsc>
hopf(const C &in, const hopf_opts &o, hopf_scratch &sc) noexcept
{
  return hopf(as_floats(in), o, sc);
}

template<f32_source C>
inline result<fhsc>
hopf(const C &in, const hopf_opts &o = {}) noexcept
{
  return hopf(as_floats(in), o);
}

};      //  namespace hsc
