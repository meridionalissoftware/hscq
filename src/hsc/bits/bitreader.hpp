//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../config.hpp"

#include <micron/types.hpp>

// 64-bit LSB-first bit reader with strict end-of-input accounting
namespace hsc::bits
{

struct bitreader {
  const u8 *p = nullptr;
  const u8 *end = nullptr;
  u64 hold = 0;
  u32 nbits = 0;

  constexpr void
  refill() noexcept
  {
    if !consteval {
      if ( end - p >= 8 ) {
        hold |= __load64(p) << nbits;
        p += 7 - ((nbits >> 3) & 7);      // whole bytes actually absorbed into hold
        nbits |= 56;                      // low 3 bits (bit phase) preserved
        return;
      }
    }
    while ( nbits <= 56 and p < end ) {
      hold |= static_cast<u64>(*p++) << nbits;
      nbits += 8;
    }
  }

  constexpr bool
  need(u32 n) noexcept
  {
    if ( nbits < n ) refill();
    return nbits >= n;
  }

  constexpr u32
  bits(u32 n) noexcept
  {
    const u32 v = static_cast<u32>(hold & ((1ull << n) - 1));
    hold >>= n;
    nbits -= n;
    return v;
  }

  constexpr void
  align_byte() noexcept
  {
    const u32 drop = nbits & 7;
    hold >>= drop;
    nbits -= drop;
  }

  constexpr u64
  consumed(const u8 *base) const noexcept
  {
    return static_cast<u64>(p - base) - (nbits >> 3);
  }
};

};      // namespace hsc::bits
