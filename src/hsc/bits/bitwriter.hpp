//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"

#include <micron/types.hpp>

// 64-bit batched bit output; LSB-first within each byte
namespace hsc::bits
{

struct bitwriter {
  u64 acc = 0;      // pending bits, LSB-first; bits at/above `cnt` are zero
  i32 cnt = 0;      // valid bit count, 0..63
  u8 *out = nullptr;
  u8 *fast_end = nullptr;

  constexpr void
  add(u64 v, i32 n) noexcept
  {
    acc |= v << cnt;
    cnt += n;
  }

  constexpr void
  flush() noexcept
  {
    if !consteval {
      if ( out < fast_end ) [[likely]] {
        __store64(out, acc);
        const i32 bytes = cnt >> 3;
        out += bytes;
        acc >>= (bytes << 3);
        cnt &= 7;
        return;
      }
    }
    while ( cnt >= 8 ) {
      *out++ = (u8)acc;
      acc >>= 8;
      cnt -= 8;
    }
  }

  constexpr u8 *
  finish() noexcept
  {
    while ( cnt > 0 ) {
      *out++ = (u8)acc;
      acc >>= 8;
      cnt -= 8;
    }
    acc = 0;
    cnt = 0;
    return out;
  }
};

};      // namespace hsc::bits
