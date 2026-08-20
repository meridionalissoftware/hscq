//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "config.hpp"

#include <micron/hash/checksum.hpp>
#include <micron/types.hpp>

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  ..crc32   payload trailer
//  ..xxh32   header check byte
//  kernels live in micron; these are forwards

namespace hsc
{

constexpr u32
crc32(bytes data, u32 init = 0) noexcept
{
  return micron::crc32_gzip_refl(init, data.ptr, data.size());
}

constexpr u32
xxh32(bytes data, u32 seed = 0) noexcept
{
  return micron::hashes::xxhash32(data.ptr, data.size(), seed);
}

};      //  namespace hsc
