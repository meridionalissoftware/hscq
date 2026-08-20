//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "config.hpp"
#include "format.hpp"
#include "hopf.hpp"
#include "unhopf.hpp"

#include <micron/types.hpp>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// hsc::ct
//
// comptime compression
//
// NOTE: GCC's default -fconstexpr-ops-limit handles payloads up to a few KiB; larger comptime
// payloads need the limits raised, see scripts/ctbuild

namespace hsc::ct
{

template<usize N> struct bytes {
  u8 data[N ? N : 1]{};
  usize len = 0;

  constexpr usize
  size() const noexcept
  {
    return len;
  }

  constexpr const u8 *
  begin() const noexcept
  {
    return data;
  }

  constexpr const u8 *
  end() const noexcept
  {
    return data + len;
  }

  constexpr u8
  operator[](usize i) const noexcept
  {
    return data[i];
  }
};

template<usize N> struct str {
  u8 data[N]{};
  usize len = N - 1;

  consteval str(const char (&s)[N])
  {
    for ( usize i = 0; i + 1 < N; ++i ) data[i] = (u8)s[i];
  }

  constexpr usize
  size() const noexcept
  {
    return len;
  }
};

template<usize N> struct f32s {
  f32 data[N ? N : 1]{};
  usize len = 0;

  constexpr usize
  size() const noexcept
  {
    return len;
  }

  constexpr f32
  operator[](usize i) const noexcept
  {
    return data[i];
  }
};

namespace __ct
{

consteval void
require(bool ok, const char *what)
{
  if ( !ok ) throw what;
}

// comptime arena caps
// fits:
// dim4 L16 (leaves, with room)
// dim8 L11 (leaves 37024)
// dim16 L10 (rows 12295, nodes 1043)
// dim32 L8
// dim64 L8
// quat L11 (leaves 36816)
// oct L10 (rows 12204; L11 wants 36816 rows)
inline constexpr usize k_nodes = 2048;
inline constexpr usize k_rows = 16384;
inline constexpr usize k_leaves = 65536;

struct views {
  tree_arena ar{};
  tree_skeleton sk{};
  pack_tables pt{};
  s2_band *bands = nullptr;
  s2_skeleton s2{};
  susp_band *sbands = nullptr;
  vq_index *spoff = nullptr;
  susp_skeleton ss{};
  susp_pack sp{};

  consteval max_t
  build_tree(u32 dim_log2, u32 dq)
  {
    ar = tree_arena{ new tree_node[k_nodes], k_nodes, 0, new tree_row[k_rows], k_rows, 0, new s3_leaf[k_leaves], k_leaves, 0 };
    const max_t root = tree_build(dim_log2, dq, ar);
    require(root >= 0, "hsc::ct: comptime arena cap exceeded (coarsen d or raise __ct caps)");
    pt = pack_tables{ new vq_index[ar.node_count], new vq_index[ar.row_count ? ar.row_count : 1] };
    const max_t pr = pack_build(ar, pt);
    require(pr >= 0, "hsc::ct: pack_build failed");
    sk = tree_view(ar, static_cast<u32>(root), dim_log2, dq);
    return root;
  }

  consteval void
  build_susp(u32 child_dim_log2, u32 dq)
  {
    ar = tree_arena{ new tree_node[k_nodes], k_nodes, 0, new tree_row[k_rows], k_rows, 0, new s3_leaf[k_leaves], k_leaves, 0 };
    sbands = new susp_band[susp_band_count(dq)];
    const max_t r = susp_build(dq, child_dim_log2, ar, sbands, ss);
    require(r >= 0, "hsc::ct: comptime arena cap exceeded (coarsen d or raise __ct caps)");
    sk = tree_view(ar, 0, child_dim_log2, dq);      // arena view; a suspension has no single root
    if ( child_dim_log2 == 3 ) {
      pt = pack_tables{ new vq_index[ar.node_count], new vq_index[ar.row_count ? ar.row_count : 1] };
      const max_t pr = pack_build(ar, pt);
      require(pr >= 0, "hsc::ct: pack_build failed");
      spoff = new vq_index[ss.count + 1];
      sp = susp_pack{ spoff, 0 };
      require(susp_pack_build(ss, pt, sp) >= 0, "hsc::ct: susp_pack_build failed");
    }
  }

  consteval void
  build_s2(u32 dq)
  {
    bands = new s2_band[s2_band_count(dq)];
    s2 = s2_build(dq, bands);
  }

  consteval void
  drop()
  {
    delete[] ar.nodes;
    delete[] ar.rows;
    delete[] ar.leaves;
    delete[] pt.node_m;
    delete[] pt.row_off;
    delete[] bands;
    delete[] sbands;
    delete[] spoff;
  }
};

consteval usize
encode(const u8 *bp, const f32 *fp, usize len, hopf_opts o, u8 *out)
{
  const mode m = fp ? (o.m == mode::bin ? mode::vec : o.m) : mode::bin;
  const __hopf::enc_params p = __hopf::resolve(o, m);
  views v{};
  usize written = 0;
  if ( p.m == mode::quotient ) {
    v.build_s2(p.dq);
    const max_t r = __hopf::quot_into(floats{ fp, len }, p, v.s2, out);
    require(r >= 0, "hsc::ct::hopf: quotient encode failed (length/values)");
    written = static_cast<usize>(r);
  } else if ( p.m == mode::quat ) {
    v.build_susp(2, p.dq);
    const max_t r = __hopf::quat_into(floats{ fp, len }, p, v.ss, v.sk, out);
    require(r >= 0, "hsc::ct::hopf: quat encode failed (length/values)");
    written = static_cast<usize>(r);
  } else if ( p.m == mode::oct ) {
    v.build_susp(3, p.dq);
    const max_t r = __hopf::oct_into(floats{ fp, len }, p, v.ss, v.sk, v.pt, v.sp, out);
    require(r >= 0, "hsc::ct::hopf: oct encode failed (length/values)");
    written = static_cast<usize>(r);
  } else {
    v.build_tree(p.dim_log2, p.dq);
    if ( fp ) {
      const max_t r = __hopf::f32_into(floats{ fp, len }, p, v.sk, v.pt, out);
      require(r >= 0, "hsc::ct::hopf: f32 encode failed (length/values)");
      written = static_cast<usize>(r);
    } else {
      written = __hopf::bin_into(hsc::bytes{ bp, len }, p, v.sk, v.pt, out);
    }
  }
  v.drop();
  return written;
}

consteval max_t
decode(const u8 *sp, usize sn, u8 *outb, f32 *outf)
{
  hopf_info fi{};
  max_t r = __format::read_header(hsc::bytes{ sp, sn }, fi);
  if ( r < 0 ) return r;
  r = __format::check_trailer(hsc::bytes{ sp, sn }, fi);
  if ( r < 0 ) return r;
  const hsc::bytes payload{ sp + k_header_size, sn - k_header_size - k_trailer_size };
  views v{};
  if ( fi.m == mode::quotient ) {
    v.build_s2(fi.d_q);
    r = __hopf::decode_core(payload, fi, nullptr, nullptr, &v.s2, nullptr, nullptr, nullptr, outf);
  } else if ( fi.m == mode::quat || fi.m == mode::oct ) {
    v.build_susp(fi.m == mode::quat ? 2 : 3, fi.d_q);
    r = __hopf::decode_core(payload, fi, &v.sk, &v.pt, nullptr, &v.ss, &v.sp, nullptr, outf);
  } else {
    v.build_tree(fi.dim_log2, fi.d_q);
    r = __hopf::decode_core(payload, fi, &v.sk, &v.pt, nullptr, nullptr, nullptr, outb, outf);
  }
  v.drop();
  return r;
}

template<typename S> inline constexpr bool is_f32_carrier = sizeof(S::data[0]) == 4;      // unevaluated

};      // namespace __ct

// compress
template<auto S, hopf_opts O = hopf_opts{}> consteval usize hopf_size()
{
  const usize cap = hsc::bound(S.len, O);
  u8 *tmp = new u8[cap];
  usize written = 0;
  if constexpr ( __ct::is_f32_carrier<decltype(S)> )
    written = __ct::encode(nullptr, S.data, S.len, O, tmp);
  else
    written = __ct::encode(S.data, nullptr, S.len, O, tmp);
  delete[] tmp;
  return written;
}

template<auto S, hopf_opts O = hopf_opts{}> consteval auto hopf()
{
  constexpr usize m = hopf_size<S, O>();
  const usize cap = hsc::bound(S.len, O);
  u8 *tmp = new u8[cap];
  usize written = 0;
  if constexpr ( __ct::is_f32_carrier<decltype(S)> )
    written = __ct::encode(nullptr, S.data, S.len, O, tmp);
  else
    written = __ct::encode(S.data, nullptr, S.len, O, tmp);
  __ct::require(written == m, "hsc::ct::hopf: probe/value pass size mismatch");
  bytes<m> out{};
  out.len = written;
  for ( usize i = 0; i < written; ++i ) out.data[i] = tmp[i];
  delete[] tmp;
  return out;
}

// decompress
template<auto Z>
consteval usize
unhopf_size()
{
  hopf_info fi{};
  __ct::require(__format::read_header(hsc::bytes{ Z.data, Z.len }, fi) >= 0, "hsc::ct::unhopf: bad stream header");
  return __out_bytes(fi);
}

template<auto Z>
consteval auto
unhopf()
{
  constexpr usize m = unhopf_size<Z>();
  hopf_info fi{};
  __ct::require(__format::read_header(hsc::bytes{ Z.data, Z.len }, fi) >= 0, "hsc::ct::unhopf: bad stream header");
  bytes<m> out{};
  if ( fi.m == mode::bin ) {
    __ct::require(__ct::decode(Z.data, Z.len, out.data, nullptr) >= 0, "hsc::ct::unhopf: decode failed");
  } else {
    f32 *tmp = new f32[fi.n_elems ? fi.n_elems : 1];
    __ct::require(__ct::decode(Z.data, Z.len, nullptr, tmp) >= 0, "hsc::ct::unhopf: decode failed");
    for ( usize i = 0; i < fi.n_elems; ++i ) __store32(out.data + 4 * i, __f2u(tmp[i]));
    delete[] tmp;
  }
  out.len = m;
  return out;
}

template<auto Z>
consteval usize
unhopf_f32_size()
{
  hopf_info fi{};
  __ct::require(__format::read_header(hsc::bytes{ Z.data, Z.len }, fi) >= 0, "hsc::ct::unhopf_f32: bad stream header");
  __ct::require(fi.m != mode::bin, "hsc::ct::unhopf_f32: byte-mode stream");
  return static_cast<usize>(fi.n_elems);
}

template<auto Z>
consteval auto
unhopf_f32()
{
  constexpr usize m = unhopf_f32_size<Z>();
  f32s<m> out{};
  __ct::require(__ct::decode(Z.data, Z.len, nullptr, out.data) >= 0, "hsc::ct::unhopf_f32: decode failed");
  out.len = m;
  return out;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// fidelity, checked by the compiler
//
// Both take results you already computed, so nothing is encoded twice -- pass the source carrier and
// the decoded one. A lossy codec has no business promising exactness in the abstract; this is how you
// prove it for the ONE payload you are baking, at the cell you are baking it at, before the build links.
//
//   inline constexpr auto k_z    = hsc::ct::hopf<k_body, hsc::opts_exact_bytes>();
//   inline constexpr auto k_back = hsc::ct::unhopf<k_z>();
//   static_assert(hsc::ct::exact<k_body, k_back>());

// worst absolute per-byte drift, or -1 when the lengths disagree
template<auto Src, auto Back>
consteval i32
max_byte_err()
{
  if ( Src.len != Back.len ) return -1;
  i32 worst = 0;
  for ( usize i = 0; i < Src.len; ++i ) {
    const i32 e = static_cast<i32>(Back.data[i]) - static_cast<i32>(Src.data[i]);
    const i32 a = e < 0 ? -e : e;
    if ( a > worst ) worst = a;
  }
  return worst;
}

template<auto Src, auto Back>
consteval bool
exact()
{
  return max_byte_err<Src, Back>() == 0;
}

template<u32 Dq>
consteval u64
s3_m()
{
  s3_leaf *lv = new s3_leaf[s3_leaf_count(Dq)];
  const u64 m = s3_build(Dq, lv).m_total;
  delete[] lv;
  return m;
}

template<u32 DimLog2, u32 Dq>
consteval u64
tree_m_mod()
{
  __ct::views v{};
  const max_t root = v.build_tree(DimLog2, Dq);
  const u64 m = v.ar.nodes[static_cast<u32>(root)].m_mod;
  v.drop();
  return m;
}

template<u32 Dq>
consteval u64
s2_m()
{
  s2_band *bd = new s2_band[s2_band_count(Dq)];
  const u64 m = s2_build(Dq, bd).m_total;
  delete[] bd;
  return m;
}

template<u32 DimLog2, u32 Dq>
consteval u32
shape_bits()
{
  __ct::views v{};
  (void)v.build_tree(DimLog2, Dq);
  const u32 b = hsc::shape_bits(v.sk, v.pt);
  v.drop();
  return b;
}

template<u32 DimLog2, u32 Dq, u32 GainBits = 8>
consteval u32
rate_bits()
{
  return GainBits + shape_bits<DimLog2, Dq>();
}

template<u32 Dq>
consteval u32
s2_bits()
{
  s2_band *bd = new s2_band[s2_band_count(Dq)];
  const u64 m = s2_build(Dq, bd).m_total;
  delete[] bd;
  return static_cast<u32>(micron::bit_width(m - 1));
}

template<u32 ChildDimLog2, u32 Dq>
consteval u64
__susp_m()
{
  __ct::views v{};
  v.build_susp(ChildDimLog2, Dq);
  const u64 m = v.ss.m_mod;
  v.drop();
  return m;
}

template<u32 Dq>
consteval u64
s4_m()
{
  return __susp_m<2, Dq>();
}

template<u32 Dq>
consteval u64
s8_m()
{
  return __susp_m<3, Dq>();
}

template<u32 Dq>
consteval u32
s4_bits()
{
  const u64 m = __susp_m<2, Dq>();
  return static_cast<u32>(micron::bit_width(m - 1));
}

template<u32 Dq>
consteval u32
s8_bits()
{
  __ct::views v{};
  v.build_susp(3, Dq);
  const u32 b = susp_bits(v.sp);      // arbint width: stays exact past u64
  v.drop();
  return b;
}

};      // namespace hsc::ct
