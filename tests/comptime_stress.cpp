// Comptime stress: a 4 KiB bin-mode container round-trip in constant evaluation. Needs the
// raised constexpr limits from scripts/ctbuild (-DHSC_CT_STRESS there); in a plain suite
// build the heavy asserts compile out and this binary is a trivial pass, czlib's pattern.

#include "../src/hsc/hsc.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

#ifdef HSC_CT_STRESS

namespace
{

inline constexpr auto kBig = []() {
  hsc::ct::bytes<4096> b{};
  u64 s = 0x243F6A8885A308D3ull;
  for ( usize i = 0; i < 4096; ++i ) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    b.data[i] = static_cast<u8>(s);
  }
  b.len = 4096;
  return b;
}();

inline constexpr hsc::hopf_opts kOpts{ .level = 6, .dim_log2 = 3 };
inline constexpr auto kZ = hsc::ct::hopf<kBig, kOpts>();
inline constexpr auto kBack = hsc::ct::unhopf<kZ>();

static_assert(kZ.len < 4096 + hsc::k_header_size + hsc::k_trailer_size);      // it compresses
static_assert(kBack.len == 4096);

consteval bool
survives()
{
  // fixed-rate lossy: every byte within the level-6 envelope, none wildly off
  for ( usize i = 0; i < kBig.len; ++i ) {
    const i32 e = static_cast<i32>(kBack.data[i]) - static_cast<i32>(kBig.data[i]);
    if ( e > 160 || e < -160 ) return false;
  }
  return true;
}

static_assert(survives());

// the suspension lanes at their comptime arena caps (quat L11 leaves-bound, oct L10 rows-bound;
// one level past each blows the ct.hpp require) -- values from the validated Python mirror
static_assert(hsc::ct::s4_m<hsc::level_dq(11)>() == 48143258803ull);
static_assert(hsc::ct::s8_bits<hsc::level_dq(10)>() == 59);      // M_S8 = 378280722294130556

}      // namespace

#endif

int
main()
{
#ifdef HSC_CT_STRESS
  sb::test_case("4 KiB comptime round-trip proven by the static_asserts above");
  sb::require(kZ.len > 0);
#else
  sb::test_case("stress asserts compiled out (build via scripts/ctbuild to prove them)");
#endif
  return 1;
}
