//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

// Shared hsc test utilities: whole-file slurp, byte-range views, and the deterministic RNG
// every test uses so failures reproduce exactly.
#pragma once

#include <micron/linux/io.hpp>
#include <micron/linux/io/ext.hpp>
#include <micron/slice.hpp>
#include <micron/std.hpp>
#include <micron/types.hpp>
#include <micron/vector.hpp>

namespace tutil
{

inline micron::vector<u8>
slurp(const char *path)
{
  micron::vector<u8> out;
  micron::posix::fd_t fd = micron::posix::open_read(path);
  if ( !fd.open() ) return out;
  micron::stat_t st{};
  if ( micron::fstat(fd, st) < 0 or st.st_size <= 0 ) {
    micron::posix::close_fd(fd);
    return out;
  }
  out.reserve(static_cast<usize>(st.st_size) + 1);
  u8 buf[4096];
  for ( ;; ) {
    const max_t n = micron::posix::read(fd, buf, sizeof(buf));
    if ( n <= 0 ) break;
    for ( max_t i = 0; i < n; i++ ) out.push_back(buf[i]);
  }
  micron::posix::close_fd(fd);
  return out;
}

inline micron::raw_slice<const u8>
view(const micron::vector<u8> &v)
{
  return { v.cbegin(), v.size() };
}

inline bool
exists(const char *path)
{
  micron::posix::fd_t fd = micron::posix::open_read(path);
  if ( !fd.open() ) return false;
  micron::posix::close_fd(fd);
  return true;
}

inline bool
bytes_equal(micron::raw_slice<const u8> a, micron::raw_slice<const u8> b)
{
  if ( a.size() != b.size() ) return false;
  for ( usize i = 0; i < a.size(); i++ )
    if ( a.ptr[i] != b.ptr[i] ) return false;
  return true;
}

// fixed-seed xorshift64: deterministic across runs and platforms, so failures reproduce exactly
struct rng {
  u64 s = 0x9E3779B97F4A7C15ull;

  constexpr u64
  next() noexcept
  {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }

  constexpr u64
  below(u64 n) noexcept
  {
    return next() % n;
  }

  // uniform-ish f64 in [-1, 1]
  constexpr f64
  unit() noexcept
  {
    return (static_cast<f64>(next() >> 11) / 9007199254740992.0) * 2.0 - 1.0;
  }
};

};      // namespace tutil
