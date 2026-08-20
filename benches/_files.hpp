//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include "../src/hsc/hsc.hpp"

#include <micron/linux/io.hpp>
#include <micron/linux/io/ext.hpp>
#include <micron/types.hpp>

namespace hf
{

inline constexpr usize k_path_max = 512;

inline bool
join(char *dst, usize cap, const char *root, const char *name) noexcept
{
  usize k = 0;
  for ( const char *p = root; *p && k + 2 < cap; ++p ) dst[k++] = *p;
  if ( k && dst[k - 1] != '/' && k + 2 < cap ) dst[k++] = '/';
  for ( const char *p = name; *p && k + 1 < cap; ++p ) dst[k++] = *p;
  dst[k] = '\0';
  return k + 1 < cap;
}

inline max_t
slurp_into(const char *path, u8 *dst, usize cap) noexcept
{
  micron::posix::fd_t fd = micron::posix::open_read(path);
  if ( !fd.open() ) return -1;
  usize got = 0;
  for ( ;; ) {
    if ( got == cap ) {
      u8 probe = 0;
      const max_t extra = micron::posix::read(fd, &probe, 1);
      micron::posix::close_fd(fd);
      return extra > 0 ? -2 : static_cast<max_t>(got);
    }
    const max_t n = micron::posix::read(fd, dst + got, cap - got);
    if ( n < 0 ) {
      micron::posix::close_fd(fd);
      return -1;
    }
    if ( n == 0 ) break;
    got += static_cast<usize>(n);
  }
  micron::posix::close_fd(fd);
  return static_cast<max_t>(got);
}

inline max_t
slurp_prefix(const char *path, u8 *dst, usize cap) noexcept
{
  micron::posix::fd_t fd = micron::posix::open_read(path);
  if ( !fd.open() ) return -1;
  usize got = 0;
  while ( got < cap ) {
    const max_t n = micron::posix::read(fd, dst + got, cap - got);
    if ( n < 0 ) {
      micron::posix::close_fd(fd);
      return -1;
    }
    if ( n == 0 ) break;
    got += static_cast<usize>(n);
  }
  micron::posix::close_fd(fd);
  return static_cast<max_t>(got);
}

inline max_t
slurp_prefix_at(const char *root, const char *name, u8 *dst, usize cap) noexcept
{
  char path[k_path_max];
  if ( !join(path, k_path_max, root, name) ) return -1;
  return slurp_prefix(path, dst, cap);
}

inline max_t
slurp_at(const char *root, const char *name, u8 *dst, usize cap) noexcept
{
  char path[k_path_max];
  if ( !join(path, k_path_max, root, name) ) return -1;
  return slurp_into(path, dst, cap);
}

inline max_t
spill(const char *path, const u8 *src, usize n) noexcept
{
  micron::posix::fd_t fd = micron::posix::open_write(path);
  if ( !fd.open() ) return -1;
  usize put = 0;
  while ( put < n ) {
    const max_t w = micron::posix::write(fd, src + put, n - put);
    if ( w <= 0 ) {
      micron::posix::close_fd(fd);
      return -1;
    }
    put += static_cast<usize>(w);
  }
  micron::posix::close_fd(fd);
  return static_cast<max_t>(put);
}

enum class kind : u8 {
  bytes,
};

struct entry {
  const char *label;
  const char *file;
  kind k;
  const char *note;
};

inline constexpr entry corpus_files[] = {
  { "urandom", "urandom.bin", kind::bytes, "/dev/urandom -- incompressible" },
  { "random", "random.bin", kind::bytes, "/dev/random -- same CSPRNG" },
  { "image-raw", "photo.ppm", kind::bytes, "raw RGB pixels" },
  { "image-png", "photo.png", kind::bytes, "already entropy-coded" },
  { "json", "data.json", kind::bytes, "structured text" },
  { "text", "text.txt", kind::bytes, "natural language" },
  { "source", "micron_src.txt", kind::bytes, "C++ source (micron.cpp)" },
  { "records", "records.bin", kind::bytes, "fixed-schema binary" },
  { "mixed", "mixed.bin", kind::bytes, "unstructured heterogeneous" },
  { "runs", "runs.bin", kind::bytes, "synthetic long runs" },
  { "words", "words.bin", kind::bytes, "synthetic Zipf words" },
};
inline constexpr usize corpus_count = sizeof(corpus_files) / sizeof(corpus_files[0]);

};      //  namespace hf
