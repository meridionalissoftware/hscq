//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

#include <micron/concepts.hpp>
#include <micron/math/__asm/hw.hpp>
#include <micron/math/bits.hpp>
#include <micron/math/ieee.hpp>
#include <micron/math/mk.hpp>
#include <micron/math/sqrt.hpp>
#include <micron/memory/cmemory.hpp>
#include <micron/slice.hpp>
#include <micron/type_traits.hpp>
#include <micron/types.hpp>

namespace hsc
{

//  type aliases
using bytes = micron::raw_slice<const u8>;
using wbytes = micron::raw_slice<u8>;
using fhsc = micron::slice<u8>;

using floats = micron::raw_slice<const f32>;
using wfloats = micron::raw_slice<f32>;
using fhsc32 = micron::slice<f32>;

//  containers
template<typename C>
concept byte_source = (micron::is_iterable_container<micron::remove_cvref_t<C>> || micron::is_string<micron::remove_cvref_t<C>>)
                      && micron::is_trivially_copyable_v<typename micron::remove_cvref_t<C>::value_type>;

//  write side
template<typename C>
concept byte_sink = byte_source<C> && !micron::is_const_v<micron::remove_pointer_t<typename micron::remove_cvref_t<C>::pointer>>;

template<typename C>
concept f32_source = byte_source<C> && micron::is_floating_point_v<typename micron::remove_cvref_t<C>::value_type>
                     && sizeof(typename micron::remove_cvref_t<C>::value_type) == 4;

template<typename C>
concept f32_sink = f32_source<C> && byte_sink<C>;

//  NOTE: for sizeof(value_type) > 1 the byte view is this machine's byte order
//  NOTE: array<T, N>::size() is N (capacity; arrays carry no fill count), so the whole buffer is read
//  NOTE: for the string types size() is the logical length in elements and excludes the NUL
template<byte_source C>
inline bytes
as_bytes(const C &c) noexcept
{
  return bytes{ reinterpret_cast<const u8 *>(c.data()), c.size() * sizeof(typename micron::remove_cvref_t<C>::value_type) };
}

template<byte_sink C>
inline wbytes
as_wbytes(C &c) noexcept
{
  return wbytes{ reinterpret_cast<u8 *>(c.data()), c.size() * sizeof(typename micron::remove_cvref_t<C>::value_type) };
}

template<f32_source C>
inline floats
as_floats(const C &c) noexcept
{
  return floats{ reinterpret_cast<const f32 *>(c.data()), c.size() };
}

template<f32_sink C>
inline wfloats
as_wfloats(C &c) noexcept
{
  return wfloats{ reinterpret_cast<f32 *>(c.data()), c.size() };
}

//  NOTE on naming: the loads/stores here are bit-named (__load16 reads 2 bytes, __store32 writes 4)
//  NOTE: ONE body each, no consteval split. gcc's bswap/store-merging pass folds the shift-or form to
//  the same single movl/movq a memcpy would emit, so consteval == runtime holds by construction here
//  rather than by agreement between two bodies -- and the byte order is the STREAM's, not the host's
constexpr u32
__load16(const u8 *p) noexcept
{
  return u32(p[0]) | (u32(p[1]) << 8);
}

constexpr u32
__load32(const u8 *p) noexcept
{
  return u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24);
}

constexpr u64
__load64(const u8 *p) noexcept
{
  return u64(p[0]) | (u64(p[1]) << 8) | (u64(p[2]) << 16) | (u64(p[3]) << 24) | (u64(p[4]) << 32) | (u64(p[5]) << 40) | (u64(p[6]) << 48)
         | (u64(p[7]) << 56);
}

constexpr void
__store16(u8 *p, u32 v) noexcept
{
  p[0] = u8(v);
  p[1] = u8(v >> 8);
}

constexpr void
__store32(u8 *p, u32 v) noexcept
{
  for ( i32 i = 0; i < 4; ++i ) p[i] = u8(v >> (8 * i));
}

constexpr void
__store64(u8 *p, u64 v) noexcept
{
  for ( i32 i = 0; i < 8; ++i ) p[i] = u8(v >> (8 * i));
}

//  NOTE: micron::memcpy counts ELEMENTS, not bytes, and when sizeof(F) != sizeof(D) it falls through to an
//  element-wise `dest[n] = static_cast<F>(src[n])` CONVERSION loop -- so it is a byte copy only when the two
//  pointee types are the same width, as they are here (u8 -> u8). Never reach for it to punch bytes into a
//  wider scalar: micron::memcpy(&u32_v, p, 4) compiles to a one-byte movzbl. Use ieee::to_bits/from_bits or
//  micron::math::bits::bit_cast for that.
constexpr void
__copy(u8 *dst, const u8 *src, usize n) noexcept
{
  micron::memcpy(dst, src, n);
}

#if !defined(HSC_ROUND_IMPL)
#define HSC_ROUND_IMPL 2
#endif

//  impl 0 is the reference form (micron's portable round kernel); 1 and 2 are the branchless rewrites.
//  mk::round_ns::round is deliberate over mkbits::: its consteval branch IS __builtin_round, so the golden
//  static_asserts in tests/exact.cpp cannot move.
//  hw::round_sd<3> is a single roundsd, the instruction __builtin_trunc used to emit -- but micron's hw:: layer
//  is LEGACY-SSE inline asm, so it is non-VEX inside an AVX2 codec. MEASURED on kernel_bench (Haswell-E, 4
//  interleaved rounds, medians), the whole-bench totals for the three trunc forms are:
//      __builtin_trunc (vroundsd, VEX)        baseline
//      hw::round_sd<3> (roundsd, legacy SSE)  -1.4% instructions, +3.1% cycles
//      mkbits::round_ns::trunc (portable C++) +11.8% instructions, +8.9% cycles
//  so hw:: is the better of the two micron forms by a wide margin, and the ~3% is the honest price of dropping
//  the builtin: micron exposes no form that is BOTH VEX-encoded and one instruction (simd::round_sd is not
//  reachable from the freestanding intrin layer). Do not swap this without re-running that A/B.
[[gnu::always_inline]] constexpr f64
__round(f64 x) noexcept
{
  if consteval {
    return micron::math::mk::round_ns::round<f64>(x);
  }
#if HSC_ROUND_IMPL == 0
  return micron::math::mk::round_ns::round<f64>(x);
#elif HSC_ROUND_IMPL == 1
  const f64 t = micron::math::hw::round_sd<3>(x);
  return micron::math::fabs(x - t) >= 0.5 ? f64(t + micron::math::copysign(f64(1.0), x)) : t;
#else
  const f64 t = micron::math::hw::round_sd<3>(x);
  const u64 m = -static_cast<u64>(micron::math::fabs(x - t) >= 0.5);
  const f64 mag = micron::math::bits::bit_cast<f64>(micron::math::bits::bit_cast<u64>(f64(1.0)) & m);
  return f64(t + micron::math::copysign(mag, x));
#endif
}

#if !defined(HSC_SQRT_IMPL)
#define HSC_SQRT_IMPL 1
#endif

//  The sqrt twin, and the reason it exists: __builtin_sqrt is NOT a bare sqrtsd here. duck's recipes carry no
//  -fno-math-errno, so gcc emits `vucomisd; ja` plus a TAIL CALL to libm around every one -- a real call inside
//  the codec's inner loops, in a library that links -nostdlib and only resolves it against micron's own weak
//  symbol (micron/math/__gcc_math_syms.hpp, whose header says: NEVER USE THIS IN REGULAR CODE).
//  IEEE-754 square root is correctly rounded and uniquely defined, so the consteval fold and the hardware
//  instruction agree bit for bit -- the same argument __round rests on. impl 0 is the constexpr-callable
//  legacy-SSE form; it is kept only as the kernel_bench A/B partner and it MEASURED FAR WORSE -- 45% more
//  cycles across the sqrt kernels on Haswell-E (quat_lift +309%, oct_lift +208%, quantize_unit +45..55%) for
//  0.4% FEWER instructions. That is an AVX/SSE transition penalty, and it is why sqrt goes through the VEX
//  intrinsic while __round's trunc (a much shorter op, measured at +3.1%) can afford the hw:: form. Note the
//  two ops disagree about which encoding wins: measure, never generalize. Leave the default at 1.
[[gnu::always_inline]] constexpr f64
__sqrt(f64 x) noexcept
{
  if consteval {
    return micron::math::hw::sqrt<f64>(x);
  }
#if HSC_SQRT_IMPL == 0
  return micron::math::hw::sqrt<f64>(x);
#else
  return micron::math::sd_sqrt(x);
#endif
}

//  f32 <-> u32 bit punning
constexpr u32
__f2u(f32 v) noexcept
{
  return micron::math::ieee::to_bits<f32>(v);
}

constexpr f32
__u2f(u32 v) noexcept
{
  return micron::math::ieee::from_bits<f32>(v);
}

};      //  namespace hsc
