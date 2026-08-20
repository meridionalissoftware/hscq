// The energy-balancing pre-rotation (codec/rot.hpp). Guards: the k_dsign_mask literal against
// its xorshift64 derivation, consteval rot_fwd of a fixed ramp against pinned f64 bit patterns
// with the runtime path held to the same pins (the identity that extends the determinism
// contract over the transform), norm preservation and involution across dims 4..64, the point
// of the feature -- every basis vector's recursive half-energy split is exactly 1/2 at every
// level -- and the flat-block split staying inside the fan. Container-level behavior (flags
// bit2, stream bytes, the spiky win) is tested in container/roundtrip/comptime.cpp, not here.

#include "../src/hsc/codec/rot.hpp"
#include "tutil.hpp"

#include <micron/types.hpp>

#include <snowball/snowball.hpp>

namespace
{

// the sign diagonal is one xorshift64 step from the golden-ratio constant; the LITERAL is the
// format constant, and this derivation must never drift from it
consteval u64
dsign_step()
{
  u64 s = 0x9E3779B97F4A7C15ull;
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

static_assert(hsc::k_dsign_mask == dsign_step());
static_assert(__builtin_popcountll(hsc::k_dsign_mask) == 38);

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// consteval rot_fwd of the dim-8 ramp {1..8}, pinned as exact bit patterns

struct r8 {
  f64 d[8];
};

consteval r8
fwd_ramp8()
{
  r8 r{};
  for ( u32 c = 0; c < 8; ++c ) r.d[c] = static_cast<f64>(c + 1);
  hsc::rot_fwd(r.d, 3);
  return r;
}

inline constexpr u64 k_ramp8_pins[8] = {
  0xC006A09E667F3BCCull, 0x4020F876CCDF6CD9ull, 0x4006A09E667F3BCCull, 0xC006A09E667F3BCCull,
  0xBFF6A09E667F3BCCull, 0xC023CC8A99AF5452ull, 0x4006A09E667F3BCCull, 0x0000000000000000ull,
};

consteval bool
ramp8_pinned()
{
  const r8 r = fwd_ramp8();
  for ( u32 c = 0; c < 8; ++c )
    if ( __builtin_bit_cast(u64, r.d[c]) != k_ramp8_pins[c] ) return false;
  return true;
}

static_assert(ramp8_pinned());

// worst deviation from 1/2 of the lower-half energy fraction, over every aligned even-length
// segment -- i.e. every node of the codec's recursive halving, all the way down
f64
worst_split_dev(const f64 *v, u32 n)
{
  f64 worst = 0;
  for ( u32 len = n; len >= 2; len >>= 1 ) {
    for ( u32 base = 0; base < n; base += len ) {
      f64 lo = 0, tot = 0;
      for ( u32 c = 0; c < len; ++c ) {
        const f64 e = v[base + c] * v[base + c];
        tot += e;
        if ( c < len / 2 ) lo += e;
      }
      const f64 frac = tot > 0.0 ? lo / tot : 1.0;
      const f64 dev = frac > 0.5 ? frac - 0.5 : 0.5 - frac;
      if ( dev > worst ) worst = dev;
    }
  }
  return worst;
}

}      // namespace

int
main()
{
  tutil::rng g;

  sb::test_case("runtime rot_fwd of the ramp equals the consteval pins bit for bit");
  {
    f64 v[8];
    for ( u32 c = 0; c < 8; ++c ) v[c] = static_cast<f64>(c + 1);
    hsc::rot_fwd(v, 3);
    for ( u32 c = 0; c < 8; ++c ) sb::require(__builtin_bit_cast(u64, v[c]), k_ramp8_pins[c]);
  }

  sb::test_case("norm preserved <= 1e-12 rel and rot_inv(rot_fwd(x)) == x <= 1e-12 abs, dims 4..64");
  {
    for ( u32 dl = 2; dl <= 6; ++dl ) {
      const u32 n = 1u << dl;
      for ( i32 t = 0; t < 500; ++t ) {
        f64 x[64], y[64];
        f64 nx = 0;
        for ( u32 c = 0; c < n; ++c ) {
          x[c] = g.unit() * 3.0;
          y[c] = x[c];
          nx += x[c] * x[c];
        }
        hsc::rot_fwd(y, dl);
        f64 ny = 0;
        for ( u32 c = 0; c < n; ++c ) ny += y[c] * y[c];
        sb::require(__builtin_fabs(__builtin_sqrt(ny) - __builtin_sqrt(nx)) <= 1e-12 * __builtin_sqrt(nx));
        hsc::rot_inv(y, dl);
        for ( u32 c = 0; c < n; ++c ) sb::require(__builtin_fabs(y[c] - x[c]) <= 1e-12);
      }
    }
  }

  sb::test_case("every basis vector splits exactly 50/50 at every recursion level, dims 4..64");
  {
    for ( u32 dl = 2; dl <= 6; ++dl ) {
      const u32 n = 1u << dl;
      for ( u32 i = 0; i < n; ++i ) {
        f64 v[64]{};
        v[i] = 1.0;
        hsc::rot_fwd(v, dl);
        sb::require(worst_split_dev(v, n) <= 1e-15);
      }
    }
  }

  sb::test_case("flat dim-8 block lands off-center but inside the fan: min-side split in [0.15, 0.45]");
  {
    f64 v[8];
    for ( u32 c = 0; c < 8; ++c ) v[c] = 1.0;
    hsc::rot_fwd(v, 3);
    f64 lo = 0, up = 0;
    for ( u32 c = 0; c < 4; ++c ) lo += v[c] * v[c];
    for ( u32 c = 4; c < 8; ++c ) up += v[c] * v[c];
    const f64 frac = lo / (lo + up);
    const f64 mn = frac < 0.5 ? frac : 1.0 - frac;
    sb::require(mn >= 0.15);
    sb::require(mn <= 0.45);
  }

  return 1;
}
