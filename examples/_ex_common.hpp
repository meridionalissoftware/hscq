//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../src/hsc/hsc.hpp"

#include <micron/io/echo.hpp>
#include <micron/linux/io.hpp>
#include <micron/linux/io/ext.hpp>
#include <micron/types.hpp>
#include <micron/vector/vector.hpp>

namespace mc = micron;

namespace ex
{

inline void
head(const char *title)
{
  mc::echo("%%%%%%%% ", title, " %%%%%%%%");
}

inline void
put3(f64 v)
{
  const bool neg = v < 0;
  if ( neg ) v = -v;
  const u64 w = static_cast<u64>(v);
  const u64 f = static_cast<u64>((v - static_cast<f64>(w)) * 1000.0 + 0.5);
  mc::echon(neg ? "-" : "", w, ".", static_cast<u32>(f / 100), static_cast<u32>((f / 10) % 10), static_cast<u32>(f % 10));
}

inline void
line3(const char *label, f64 v, const char *suffix = "")
{
  mc::echon(label);
  put3(v);
  mc::echo(suffix);
}

// pad a label or a number out so a hand-built table lines up
inline void
pad(const char *s, usize w)
{
  usize n = 0;
  for ( const char *p = s; *p; ++p ) ++n;
  mc::echon(s);
  for ( ; n < w; ++n ) mc::echon(" ");
}

inline void
padnum(u64 v, usize w)
{
  usize n = 1;
  for ( u64 x = v; x >= 10; x /= 10 ) ++n;
  mc::echon(v);
  for ( ; n < w; ++n ) mc::echon(" ");
}

// a fixed-decimal number in a fixed-width column
inline void
padf3(f64 v, usize w)
{
  const bool neg = v < 0;
  f64 a = neg ? -v : v;
  const u64 whole = static_cast<u64>(a);
  usize n = (neg ? 1 : 0) + 4;      // sign + ".ddd"
  for ( u64 x = whole; x >= 10; x /= 10 ) ++n;
  ++n;
  put3(v);
  for ( ; n < w; ++n ) mc::echon(" ");
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// deterministic data

struct rng {
  u64 s = 0x9E3779B97F4A7C15ull;

  u64
  next() noexcept
  {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }

  f64
  unit() noexcept
  {
    return (static_cast<f64>(next() >> 11) / 9007199254740992.0) * 2.0 - 1.0;
  }
};

// a weight-matrix-shaped tensor: centered, correlated along a row, a handful of loud rows
inline mc::vector<f32>
weights(usize rows, usize cols, u64 seed = 0x9E3779B97F4A7C15ull)
{
  rng g{ seed };
  mc::vector<f32> w;
  w.reserve(rows * cols + 1);
  w.resize(rows * cols);
  for ( usize r = 0; r < rows; ++r ) {
    const f64 loud = (r % 61 == 0) ? 12.0 : 1.0;
    f64 acc = 0;
    for ( usize c = 0; c < cols; ++c ) {
      acc = acc * 0.75 + g.unit() * 0.25;
      w[r * cols + c] = static_cast<f32>(acc * loud);
    }
  }
  return w;
}

// an embedding table
inline mc::vector<f32>
embeddings(usize rows, usize cols, u64 seed = 0xD1B54A32D192ED03ull)
{
  rng g{ seed };
  mc::vector<f32> e;
  e.reserve(rows * cols + 1);
  e.resize(rows * cols);
  for ( usize r = 0; r < rows; ++r ) {
    f64 s = 0;
    for ( usize c = 0; c < cols; ++c ) {
      const f64 v = g.unit();
      e[r * cols + c] = static_cast<f32>(v);
      s += v * v;
    }
    const f64 inv = s > 0 ? 1.0 / micron::math::fsqrt(s) : 0.0;
    for ( usize c = 0; c < cols; ++c ) e[r * cols + c] = static_cast<f32>(static_cast<f64>(e[r * cols + c]) * inv);
  }
  return e;
}

inline f64
cosine(const f32 *a, const f32 *b, usize n)
{
  f64 d = 0, x = 0, y = 0;
  for ( usize i = 0; i < n; ++i ) {
    d += static_cast<f64>(a[i]) * static_cast<f64>(b[i]);
    x += static_cast<f64>(a[i]) * static_cast<f64>(a[i]);
    y += static_cast<f64>(b[i]) * static_cast<f64>(b[i]);
  }
  return (x > 0 && y > 0) ? d / micron::math::fsqrt(x * y) : 0.0;
}

inline max_t
slurp_prefix(const char *path, u8 *dst, usize cap)
{
  mc::posix::fd_t fd = mc::posix::open_read(path);
  if ( !fd.open() ) return -1;
  usize got = 0;
  while ( got < cap ) {
    const max_t n = mc::posix::read(fd, dst + got, cap - got);
    if ( n <= 0 ) break;
    got += static_cast<usize>(n);
  }
  mc::posix::close_fd(fd);
  return static_cast<max_t>(got);
}

};      // namespace ex
