//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"
#include "../error.hpp"
#include "../sphere/s2.hpp"
#include "../sphere/tree.hpp"
#include "pack.hpp"

#include <micron/types.hpp>
#include <micron/vector/vector.hpp>

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// reusable scratch cache

namespace hsc
{

struct hopf_scratch {
  micron::vector<tree_node> nodes;
  micron::vector<tree_row> rows;
  micron::vector<s3_leaf> leaves;
  micron::vector<vq_index> node_m;
  micron::vector<vq_index> row_off;
  micron::vector<s2_band> bands;
  micron::vector<susp_band> sbands;
  micron::vector<vq_index> band_off;      // suspension prefix table, oct only
  tree_skeleton sk{};
  pack_tables pt{};
  s2_skeleton s2{};
  susp_skeleton ss{};
  susp_pack sp{};
  u32 built_kind = 0;      // 0 = plain tree | 1 = S^4 suspension | 2 = S^8 suspension
  u32 built_dim = 0;       // dim_log2 of the cached tree; 0 = none
  u32 built_dq = 0;
  u32 built_s2_dq = 0;      // d_q of the cached S^2 skeleton; 0 = none

  ~hopf_scratch() = default;

  hopf_scratch() = default;
  hopf_scratch(const hopf_scratch &) = delete;
  hopf_scratch &operator=(const hopf_scratch &) = delete;

  max_t
  build_tree(u32 dim_log2, u32 dq)
  {
    if ( built_kind == 0 && built_dim == dim_log2 && built_dq == dq ) return 0;
    built_dim = 0;
    built_kind = 0;
    usize nc = 256, rc = 8192, lc = 32768;
    for ( i32 retry = 0; retry < 7; ++retry ) {
      nodes.resize(nc);
      rows.resize(rc);
      leaves.resize(lc);
      tree_arena ar{ nodes.data(), static_cast<u32>(nc), 0, rows.data(), static_cast<u32>(rc), 0, leaves.data(), static_cast<u32>(lc), 0 };
      const max_t root = tree_build(dim_log2, dq, ar);
      if ( root >= 0 ) {
        node_m.resize(ar.node_count);
        row_off.resize(ar.row_count);
        pt = pack_tables{ node_m.data(), row_off.data() };
        const max_t pr = pack_build(ar, pt);
        if ( pr < 0 ) [[unlikely]]
          return pr;
        sk = tree_view(ar, static_cast<u32>(root), dim_log2, dq);
        built_dim = dim_log2;
        built_dq = dq;
        return 0;
      }
      nc *= 4;
      rc *= 4;
      lc *= 4;
    }
    return fail(error::oom);
  }

  max_t
  build_susp(u32 child_dim_log2, u32 dq)
  {
    const u32 kind = child_dim_log2 == 2 ? 1u : 2u;
    if ( built_kind == kind && built_dq == dq ) return 0;
    built_dim = 0;
    built_kind = 0;
    sbands.resize(susp_band_count(dq));
    usize nc = 256, rc = 8192, lc = 32768;
    for ( i32 retry = 0; retry < 7; ++retry ) {
      nodes.resize(nc);
      rows.resize(rc);
      leaves.resize(lc);
      tree_arena ar{ nodes.data(), static_cast<u32>(nc), 0, rows.data(), static_cast<u32>(rc), 0, leaves.data(), static_cast<u32>(lc), 0 };
      const max_t r = susp_build(dq, child_dim_log2, ar, sbands.data(), ss);
      if ( r >= 0 ) {
        sk = tree_view(ar, 0, child_dim_log2, dq);      // arena view; a suspension has no single root
        if ( kind == 2 ) {
          node_m.resize(ar.node_count);
          row_off.resize(ar.row_count ? ar.row_count : 1);
          pt = pack_tables{ node_m.data(), row_off.data() };
          const max_t pr = pack_build(ar, pt);
          if ( pr < 0 ) [[unlikely]]
            return pr;
          band_off.resize(ss.count + 1);
          sp = susp_pack{ band_off.data(), 0 };
          const max_t br = susp_pack_build(ss, pt, sp);
          if ( br < 0 ) [[unlikely]]
            return br;
        }
        built_kind = kind;
        built_dq = dq;
        return 0;
      }
      nc *= 4;
      rc *= 4;
      lc *= 4;
    }
    return fail(error::oom);
  }

  max_t
  build_s2(u32 dq)
  {
    if ( built_s2_dq == dq ) return 0;
    built_s2_dq = 0;
    bands.resize(s2_band_count(dq));
    s2 = s2_build(dq, bands.data());
    built_s2_dq = dq;
    return 0;
  }
};

};      // namespace hsc
