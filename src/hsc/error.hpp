//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include <micron/sum.hpp>
#include <micron/types.hpp>

namespace hsc
{

enum class error : i32 {
  ok = 0,
  bad_container,      // framing invalid: magic, version-reserved rules, flag bits, hc byte, cross-mode header rules
  bad_stream,         // payload malformed: bit accounting off, index >= its radix, nonzero pad bits
  bad_checksum,       // payload crc32 trailer mismatch
  bad_length,         // n_elems vs payload/trailer count mismatch; input not block-divisible on encode
  short_output,       // caller-supplied output buffer too small for the decoded data
  short_input,        // stream truncated: header, payload bits or trailer ran out
  unsupported,        // stream version or reserved profile this build does not implement
  oom,                // allocation failed while sizing an owned output or scratch
  bad_skeleton,       // d_q-rebuilt skeleton fails the M-mod-2^64 / bits_per_block guard (float drift)
  bad_value,          // NaN/Inf input, or a zero-norm block where unit input is required
  bad_opts,           // d/dim_log2/gain_bits out of range, bin mode on floats (quotient-family width
                      // mismatches are bad_length: input not block-divisible, see above)
};

template<typename T> using result = micron::option<T, error>;

constexpr max_t
fail(error e) noexcept
{
  return -static_cast<max_t>(static_cast<i32>(e));
}

constexpr error
as_error(max_t r) noexcept
{
  return r < 0 ? static_cast<error>(static_cast<i32>(-r)) : error::ok;
}

constexpr const char *
error_name(error e) noexcept
{
  switch ( e ) {
  case error::ok:
    return "ok";
  case error::bad_container:
    return "bad_container";
  case error::bad_stream:
    return "bad_stream";
  case error::bad_checksum:
    return "bad_checksum";
  case error::bad_length:
    return "bad_length";
  case error::short_output:
    return "short_output";
  case error::short_input:
    return "short_input";
  case error::unsupported:
    return "unsupported";
  case error::oom:
    return "oom";
  case error::bad_skeleton:
    return "bad_skeleton";
  case error::bad_value:
    return "bad_value";
  case error::bad_opts:
    return "bad_opts";
  }
  return "unknown";
}

};      // namespace hsc
