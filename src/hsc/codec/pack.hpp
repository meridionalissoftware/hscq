//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../bits/bitreader.hpp"
#include "../bits/bitwriter.hpp"
#include "../config.hpp"
#include "../error.hpp"
#include "../sphere/susp.hpp"
#include "../sphere/tree.hpp"

#include <micron/math/arbint.hpp>
#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  exact index packing: per-node fields <-> one flat index in [0, M)
//
//  recursive cardinalities overflow machine u64 words by a massive margin: at dim 64, d = 0.1 (L7)
//  the exact M has bit_length 148 (log2 M ~= 147.7), and at the dq_min floor (L16) the dim-64 index
//  reaches 781 bits -- we use the stack allocd arbuint (microns bigint type)
//
//  width derivation: pack_build/susp_pack_build reject at bit_length > hsc_index_bits - 64 = 960,
//  so 1024 gives 960 usable bits against a measured worst case of 781 (+64 headroom = 845 needed);
//  widening is always safe, narrowing below ~845 starts rejecting dim-64 streams at the fine end

namespace hsc
{

//  stack
inline constexpr usize hsc_index_bits = 1024;
using vq_index = micron::arbuint<hsc_index_bits>;

//  node_m[node id], row_off[row slot]
struct pack_tables {
  vq_index *node_m = nullptr;
  vq_index *row_off = nullptr;
};

//  children live at dim_log2 - 1
constexpr max_t
pack_build(const tree_arena &ar, pack_tables &pt) noexcept
{
  for ( u32 dl = 2; dl <= 6; ++dl ) {
    for ( u32 id = 0; id < ar.node_count; ++id ) {
      const tree_node &nd = ar.nodes[id];
      if ( nd.dim_log2 != dl ) continue;
      if ( dl == 2 ) {
        pt.node_m[id] = vq_index(nd.m_mod);      //  NOTE: dim-4 cardinality fits u64
        continue;
      }
      vq_index acc(0u);
      for ( u32 x = 0; x < nd.count; ++x ) {
        const tree_row &rw = ar.rows[nd.rows_at + x];
        pt.row_off[nd.rows_at + x] = acc;
        vq_index prod = pt.node_m[rw.c1];
        prod *= pt.node_m[rw.c2];
        acc += prod;
      }
      if ( acc.bit_length() > hsc_index_bits - 64 ) [[unlikely]]
        return fail(error::bad_opts);
      pt.node_m[id] = acc;
    }
  }
  return 0;
}

constexpr const vq_index &
pack_m(const tree_skeleton &sk, const pack_tables &pt) noexcept
{
  return pt.node_m[sk.root];
}

//  ceil(log2 M) = bit_length(M - 1); the M == 1 degenerate code packs to a zero-width field
constexpr u32
shape_bits(const tree_skeleton &sk, const pack_tables &pt) noexcept
{
  vq_index m = pt.node_m[sk.root];
  m -= 1u;
  return static_cast<u32>(m.bit_length());
}

//  NOTE: micron's bounded store default-ctor is d{}, which zeroes all cap_limbs by design;
//  avoid via a scratch
struct pack_scratch {
  vq_index t[7];
};

constexpr void
__pack_rank(const tree_skeleton &sk, const pack_tables &pt, const tree_fields &f, u32 id, u32 &pi, u32 &pb, vq_index &out,
            vq_index *tmp) noexcept
{
  const tree_node &nd = sk.nodes[id];
  if ( nd.dim_log2 == 2 ) {
    out = f.base[pb++];      //  operator=(u64) wins: writes one limb, no 128-byte temporary
    return;
  }
  const u32 x = f.leaf[pi++];
  const tree_row &rw = sk.rows[nd.rows_at + x];
  vq_index &a1 = tmp[nd.dim_log2];
  __pack_rank(sk, pt, f, rw.c1, pi, pb, a1, tmp);
  __pack_rank(sk, pt, f, rw.c2, pi, pb, out, tmp);
  out *= pt.node_m[rw.c1];
  out += a1;
  out += pt.row_off[nd.rows_at + x];
}

constexpr void
pack_rank(const tree_skeleton &sk, const pack_tables &pt, const tree_fields &f, vq_index &out, pack_scratch &ps) noexcept
{
  u32 pi = 0, pb = 0;
  __pack_rank(sk, pt, f, sk.root, pi, pb, out, ps.t);
}

constexpr void
pack_rank(const tree_skeleton &sk, const pack_tables &pt, const tree_fields &f, vq_index &out) noexcept
{
  pack_scratch ps;
  pack_rank(sk, pt, f, out, ps);
}

constexpr void
__pack_unrank(const tree_skeleton &sk, const pack_tables &pt, vq_index &a, u32 id, u32 &pi, u32 &pb, tree_fields &f, vq_index *tmp) noexcept
{
  const tree_node &nd = sk.nodes[id];
  if ( nd.dim_log2 == 2 ) {
    f.base[pb++] = static_cast<u64>(a);
    return;
  }
  u32 lo = 0, hi = nd.count - 1;      //  last row with row_off <= a (prefix sums are monotone)
  while ( lo < hi ) {
    const u32 mid = (lo + hi + 1) >> 1;
    lo = pt.row_off[nd.rows_at + mid] <= a ? mid : lo;
    hi = pt.row_off[nd.rows_at + mid] <= a ? hi : mid - 1;
  }
  f.leaf[pi++] = lo;
  const tree_row &rw = sk.rows[nd.rows_at + lo];
  a -= pt.row_off[nd.rows_at + lo];
  vq_index &q = tmp[nd.dim_log2];
  a.__divmod_both(pt.node_m[rw.c1], q);      //  a = quot * M1 + rem: a becomes rem, q the quot
  __pack_unrank(sk, pt, a, rw.c1, pi, pb, f, tmp);
  __pack_unrank(sk, pt, q, rw.c2, pi, pb, f, tmp);
}

//  flat index -> per-node fields; rejects a >= M
constexpr max_t
pack_unrank(const tree_skeleton &sk, const pack_tables &pt, vq_index &a, tree_fields &f, pack_scratch &ps) noexcept
{
  if ( !(a < pt.node_m[sk.root]) ) [[unlikely]]
    return fail(error::bad_stream);
  u32 pi = 0, pb = 0;
  __pack_unrank(sk, pt, a, sk.root, pi, pb, f, ps.t);
  return 0;
}

//  convenience
constexpr max_t
pack_unrank(const tree_skeleton &sk, const pack_tables &pt, vq_index a, tree_fields &f) noexcept
{
  pack_scratch ps;
  return pack_unrank(sk, pt, a, f, ps);
}

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  suspension index accounting for arbints
//  (mode oct: S^8 over dim-8 tree children)

struct susp_pack {
  vq_index *band_off = nullptr;
  u32 count = 0;
};

constexpr max_t
susp_pack_build(const susp_skeleton &ss, const pack_tables &pt, susp_pack &sp) noexcept
{
  vq_index acc(0u);
  for ( u32 b = 0; b < ss.count; ++b ) {
    sp.band_off[b] = acc;
    if ( b == 0 || b == ss.count - 1 )
      acc += 1u;
    else
      acc += pt.node_m[ss.bd[b].child];
  }
  if ( acc.bit_length() > hsc_index_bits - 64 ) [[unlikely]]
    return fail(error::bad_opts);
  sp.band_off[ss.count] = acc;
  sp.count = ss.count;
  return 0;
}

constexpr const vq_index &
susp_m(const susp_pack &sp) noexcept
{
  return sp.band_off[sp.count];
}

//  ceil(log2 M); the suspension is never degenerate (M >= 2 at every valid d_q)
constexpr u32
susp_bits(const susp_pack &sp) noexcept
{
  vq_index m = sp.band_off[sp.count];
  m -= 1u;
  return static_cast<u32>(m.bit_length());
}

constexpr void
susp_rank(const susp_skeleton &ss, const tree_skeleton &tv, const pack_tables &pt, const susp_pack &sp, u32 band, const tree_fields &f,
          vq_index &out, pack_scratch &ps) noexcept
{
  if ( band == 0 || band == ss.count - 1 ) {
    out = sp.band_off[band];
    return;
  }
  pack_rank(susp_child_view(ss, tv, band), pt, f, out, ps);
  out += sp.band_off[band];
}

//  flat index -> (band, child fields); rejects a >= M
constexpr max_t
susp_unrank(const susp_skeleton &ss, const tree_skeleton &tv, const pack_tables &pt, const susp_pack &sp, vq_index &a, u32 &band,
            tree_fields &f, pack_scratch &ps) noexcept
{
  if ( !(a < sp.band_off[sp.count]) ) [[unlikely]]
    return fail(error::bad_stream);
  u32 lo = 0, hi = sp.count - 1;      //  last band with band_off <= a (prefix sums are monotone)
  while ( lo < hi ) {
    const u32 mid = (lo + hi + 1) >> 1;
    lo = sp.band_off[mid] <= a ? mid : lo;
    hi = sp.band_off[mid] <= a ? hi : mid - 1;
  }
  band = lo;
  a -= sp.band_off[lo];
  if ( lo == 0 || lo == sp.count - 1 ) return 0;
  return pack_unrank(susp_child_view(ss, tv, lo), pt, a, f, ps);
}

//  NOTE: the limb indexing below (pos >> 6, pos & 63) assumes 64-bit arbuint limbs -- true on every
//  target hsc supports (x86-64 Linux); a u32-limb build of micron would need this rewritten limb-
//  agnostically (get_wide below already is)
constexpr void
put_wide(bits::bitwriter &w, const vq_index &v, u32 nbits) noexcept
{
  const usize limbs = v.size();
  for ( u32 pos = 0; pos < nbits; pos += 32 ) {
    const u32 take = nbits - pos < 32 ? nbits - pos : 32;
    const usize li = pos >> 6;
    u64 limb = li < limbs ? static_cast<u64>(v[li]) : 0ull;
    u32 chunk = static_cast<u32>(limb >> (pos & 63));
    if ( take < 32 ) chunk &= (1u << take) - 1;
    w.add(chunk, static_cast<i32>(take));
    w.flush();
  }
}

constexpr bool
get_wide(bits::bitreader &r, u32 nbits, vq_index &v) noexcept
{
  v = vq_index(0u);
  for ( u32 pos = 0; pos < nbits; pos += 32 ) {
    const u32 take = nbits - pos < 32 ? nbits - pos : 32;
    if ( !r.need(take) ) [[unlikely]]
      return false;
    const u32 chunk = r.bits(take);
    if ( chunk ) {
      vq_index t(chunk);
      t <<= pos;
      v |= t;
    }
  }
  return true;
}

};      //  namespace hsc
