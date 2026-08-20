//  The comptime contract. Building this file IS most of the test: the static_asserts prove the
//  consteval codec (golden cardinalities, whole-container round-trips in constant evaluation
//  for every mode). main() adds the determinism half: the consteval-built stream must be
//  byte-identical to the runtime encoder's output for the same input -- compiler constant-fold
//  trig on one side, micron kernels on the other, one wire format out of both.

#include "../src/hsc/hsc.hpp"
#include "tutil.hpp"

#include <micron/std.hpp>

#include <snowball/snowball.hpp>

namespace
{

//  golden cardinalities through the ct probes
static_assert(hsc::ct::s3_m<hsc::level_dq(3)>() == 138);
static_assert(hsc::ct::s3_m<hsc::level_dq(6)>() == 2588);
static_assert(hsc::ct::s3_m<hsc::level_dq(9)>() == 2828294);
static_assert(hsc::ct::tree_m_mod<3, hsc::level_dq(3)>() == 2310);
static_assert(hsc::ct::tree_m_mod<4, hsc::level_dq(3)>() == 60316);
static_assert(hsc::ct::tree_m_mod<4, hsc::level_dq(6)>() == 73150212400ull);
static_assert(hsc::ct::s2_m<hsc::level_dq(3)>() == 46);
static_assert(hsc::ct::s2_m<hsc::level_dq(7)>() == 1236);
static_assert(hsc::ct::s4_m<hsc::level_dq(3)>() == 332);
static_assert(hsc::ct::s4_m<hsc::level_dq(6)>() == 16720);
static_assert(hsc::ct::s4_bits<hsc::level_dq(6)>() == 15);
static_assert(hsc::ct::s8_m<hsc::level_dq(3)>() == 4552);
static_assert(hsc::ct::s8_m<hsc::level_dq(6)>() == 10923842);
static_assert(hsc::ct::s8_bits<hsc::level_dq(6)>() == 24);

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  bin: compress a body at compile time, decode it back at compile time

inline constexpr hsc::ct::str kBody{ "the quick brown fox jumps over the lazy dog, hopfed and unhopfed at compile time" };
inline constexpr hsc::hopf_opts kBinOpts{ .level = 6, .dim_log2 = 3 };
inline constexpr auto kZ = hsc::ct::hopf<kBody, kBinOpts>();
inline constexpr auto kBack = hsc::ct::unhopf<kZ>();

static_assert(kZ.len > hsc::k_header_size + hsc::k_trailer_size);
static_assert(kZ.len <= hsc::bound(kBody.len, kBinOpts));
static_assert(kBack.len == kBody.len);

//  lossy but bounded: per-byte error under the coarse level 6 envelope, verified consteval
consteval bool
bin_close()
{
  for ( usize i = 0; i < kBody.len; ++i ) {
    const i32 e = static_cast<i32>(kBack.data[i]) - static_cast<i32>(kBody.data[i]);
    if ( e > 120 || e < -120 ) return false;
  }
  return true;
}

static_assert(bin_close());

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  the bit-exact cell, proven in the type system
//
//  hsc::opts_exact_bytes (dim 4, L13) is the cell BENCHMARKS.md measures at rmse 0.00 / psnr 99.99 --
//  the bench clamps psnr to 99.99 below rmse 1e-9 and an integer rmse is 0 or >= 1/sqrt(n), so that
//  column means EXACT. Here the compiler proves it for one payload rather than trusting the sweep.
//  Kept short on purpose: the cost is the 1571-row s3 skeleton the constant evaluator has to build,
//  and this file compiles at the default -fconstexpr-ops-limit.
inline constexpr hsc::ct::str kExactBody{ "exact bytes, at compile time." };
inline constexpr auto kZe = hsc::ct::hopf<kExactBody, hsc::opts_exact_bytes>();
inline constexpr auto kBe = hsc::ct::unhopf<kZe>();

static_assert(hsc::exact_bytes(hsc::opts_exact_bytes));      //  the cell claim
static_assert(hsc::ct::exact<kExactBody, kBe>());            //  the payload proof
static_assert(hsc::ct::max_byte_err<kExactBody, kBe>() == 0);
static_assert(kZe.len > kExactBody.len);      //  10.75 bits/byte: exactness EXPANDS

//  and the coarse cell above is genuinely not exact, so the assert above is testing something
static_assert(hsc::ct::max_byte_err<kBody, kBack>() > 0);

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  the transform lane: bit2 streams are part of the determinism contract too

inline constexpr hsc::hopf_opts kBinRotOpts{ .level = 6, .dim_log2 = 3, .transform = true };
inline constexpr auto kZr = hsc::ct::hopf<kBody, kBinRotOpts>();
inline constexpr auto kBackr = hsc::ct::unhopf<kZr>();

static_assert(kZr.len == kZ.len);        //  analog-side: identical rate
static_assert(kZr.data[5] == 0x07);      //  exact_pack | has_gain | transform
static_assert(kBackr.len == kBody.len);

consteval bool
bin_rot_close()
{
  for ( usize i = 0; i < kBody.len; ++i ) {
    const i32 e = static_cast<i32>(kBackr.data[i]) - static_cast<i32>(kBody.data[i]);
    if ( e > 120 || e < -120 ) return false;
  }
  return true;
}

static_assert(bin_rot_close());

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  float lanes at compile time: vec, unit, quotient

inline constexpr auto kF = []() {
  hsc::ct::f32s<16> f{};
  for ( usize i = 0; i < 16; ++i ) f.data[i] = static_cast<f32>(static_cast<f64>(i % 7) * 0.25 - 0.5);
  f.len = 16;
  return f;
}();

inline constexpr auto kU = []() {
  hsc::ct::f32s<16> f{};
  for ( usize b = 0; b < 4; ++b ) {
    f64 v[4], s = 0;
    for ( usize c = 0; c < 4; ++c ) {
      v[c] = static_cast<f64>(b + c + 1);
      s += v[c] * v[c];
    }
    const f64 n = __builtin_sqrt(s);
    for ( usize c = 0; c < 4; ++c ) f.data[b * 4 + c] = static_cast<f32>(v[c] / n);
  }
  f.len = 16;
  return f;
}();

inline constexpr hsc::hopf_opts kVecOpts{ .m = hsc::mode::vec, .level = 5, .dim_log2 = 2, .gain_bits = 6 };
inline constexpr auto kZv = hsc::ct::hopf<kF, kVecOpts>();
inline constexpr auto kBv = hsc::ct::unhopf_f32<kZv>();
static_assert(kBv.len == 16);

inline constexpr hsc::hopf_opts kUnitOpts{ .m = hsc::mode::unit, .level = 5, .dim_log2 = 2 };
inline constexpr auto kZu = hsc::ct::hopf<kU, kUnitOpts>();
inline constexpr auto kBu = hsc::ct::unhopf_f32<kZu>();
static_assert(kBu.len == 16);

inline constexpr hsc::hopf_opts kQuotOpts{ .m = hsc::mode::quotient, .level = 5 };
inline constexpr auto kZq = hsc::ct::hopf<kU, kQuotOpts>();
inline constexpr auto kBq = hsc::ct::unhopf_f32<kZq>();
static_assert(kBq.len == 16);

//  the sibling quotient lanes reuse the same 16-float unit fixture: 2 quat blocks / 1 oct block
inline constexpr hsc::hopf_opts kQuatOpts{ .m = hsc::mode::quat, .level = 5 };
inline constexpr auto kZq4 = hsc::ct::hopf<kU, kQuatOpts>();
inline constexpr auto kBq4 = hsc::ct::unhopf_f32<kZq4>();
static_assert(kBq4.len == 16);

inline constexpr hsc::hopf_opts kOctOpts{ .m = hsc::mode::oct, .level = 5 };
inline constexpr auto kZq8 = hsc::ct::hopf<kU, kOctOpts>();
inline constexpr auto kBq8 = hsc::ct::unhopf_f32<kZq8>();
static_assert(kBq8.len == 16);

//  the raw-bytes ct surface serializes float-mode output as little-endian f32s
inline constexpr auto kBvRaw = hsc::ct::unhopf<kZv>();
static_assert(kBvRaw.len == 64);

consteval bool
raw_matches_typed()
{
  for ( usize i = 0; i < 16; ++i )
    if ( hsc::__load32(kBvRaw.data + 4 * i) != hsc::__f2u(kBv.data[i]) ) return false;
  return true;
}

static_assert(raw_matches_typed());

}      //  namespace

int
main()
{
  hsc::hopf_scratch sc;

  sb::test_case("consteval stream == runtime stream, byte for byte (bin)");
  {
    micron::vector<u8> src;
    src.reserve(kBody.len + 1);
    for ( usize i = 0; i < kBody.len; ++i ) src.push_back(kBody.data[i]);
    hsc::fhsc z = hsc::hopf(tutil::view(src), kBinOpts, sc);
    sb::require(z.size(), kZ.len);
    sb::require(tutil::bytes_equal({ z.first(), z.size() }, { kZ.data, kZ.len }));

    hsc::fhsc zr = hsc::hopf(tutil::view(src), kBinRotOpts, sc);
    sb::require(zr.size(), kZr.len);
    sb::require(tutil::bytes_equal({ zr.first(), zr.size() }, { kZr.data, kZr.len }));

    auto br = hsc::unhopf(hsc::bytes{ kZr.data, kZr.len }, sc);
    sb::require(br.is_first());
    sb::require(tutil::bytes_equal({ br.cast<hsc::fhsc>().first(), br.cast<hsc::fhsc>().size() }, { kBackr.data, kBackr.len }));
  }

  sb::test_case("the bit-exact cell agrees consteval and runtime, and is exact on both sides");
  {
    micron::vector<u8> src;
    src.reserve(kExactBody.len + 1);
    for ( usize i = 0; i < kExactBody.len; ++i ) src.push_back(kExactBody.data[i]);
    hsc::fhsc z = hsc::hopf(tutil::view(src), hsc::opts_exact_bytes, sc);
    sb::require(z.size(), kZe.len);
    sb::require(tutil::bytes_equal({ z.first(), z.size() }, { kZe.data, kZe.len }));

    auto b = hsc::unhopf(hsc::bytes{ kZe.data, kZe.len }, sc);
    sb::require(b.is_first());
    sb::require(tutil::bytes_equal({ b.cast<hsc::fhsc>().first(), b.cast<hsc::fhsc>().size() }, { kBe.data, kBe.len }));
    //  the consteval side already static_assert'd exactness; this is the runtime half of the same claim
    sb::require(tutil::bytes_equal({ b.cast<hsc::fhsc>().first(), b.cast<hsc::fhsc>().size() }, { kExactBody.data, kExactBody.len }));
  }

  sb::test_case("consteval stream == runtime stream (vec, unit, quotient)");
  {
    micron::vector<f32> f;
    f.reserve(17);
    for ( usize i = 0; i < 16; ++i ) f.push_back(kF.data[i]);
    auto zv = hsc::hopf(hsc::as_floats(f), kVecOpts, sc);
    sb::require(zv.is_first());
    sb::require(zv.cast<hsc::fhsc>().size(), kZv.len);
    sb::require(tutil::bytes_equal({ zv.cast<hsc::fhsc>().first(), kZv.len }, { kZv.data, kZv.len }));

    micron::vector<f32> u;
    u.reserve(17);
    for ( usize i = 0; i < 16; ++i ) u.push_back(kU.data[i]);
    auto zu = hsc::hopf(hsc::as_floats(u), kUnitOpts, sc);
    sb::require(zu.is_first());
    sb::require(tutil::bytes_equal({ zu.cast<hsc::fhsc>().first(), kZu.len }, { kZu.data, kZu.len }));

    auto zq = hsc::hopf(hsc::as_floats(u), kQuotOpts, sc);
    sb::require(zq.is_first());
    sb::require(tutil::bytes_equal({ zq.cast<hsc::fhsc>().first(), kZq.len }, { kZq.data, kZq.len }));

    auto zq4 = hsc::hopf(hsc::as_floats(u), kQuatOpts, sc);
    sb::require(zq4.is_first());
    sb::require(zq4.cast<hsc::fhsc>().size(), kZq4.len);
    sb::require(tutil::bytes_equal({ zq4.cast<hsc::fhsc>().first(), kZq4.len }, { kZq4.data, kZq4.len }));

    auto zq8 = hsc::hopf(hsc::as_floats(u), kOctOpts, sc);
    sb::require(zq8.is_first());
    sb::require(zq8.cast<hsc::fhsc>().size(), kZq8.len);
    sb::require(tutil::bytes_equal({ zq8.cast<hsc::fhsc>().first(), kZq8.len }, { kZq8.data, kZq8.len }));
  }

  sb::test_case("runtime decode of the consteval streams matches the consteval decode");
  {
    auto b = hsc::unhopf(hsc::bytes{ kZ.data, kZ.len }, sc);
    sb::require(b.is_first());
    sb::require(tutil::bytes_equal({ b.cast<hsc::fhsc>().first(), b.cast<hsc::fhsc>().size() }, { kBack.data, kBack.len }));

    auto bv = hsc::unhopf_f32(hsc::bytes{ kZv.data, kZv.len });
    sb::require(bv.is_first());
    for ( usize i = 0; i < 16; ++i ) sb::require(hsc::__f2u(bv.cast<hsc::fhsc32>().first()[i]), hsc::__f2u(kBv.data[i]));

    auto bq = hsc::unhopf_f32(hsc::bytes{ kZq.data, kZq.len });
    sb::require(bq.is_first());
    for ( usize i = 0; i < 16; ++i ) sb::require(hsc::__f2u(bq.cast<hsc::fhsc32>().first()[i]), hsc::__f2u(kBq.data[i]));

    auto bq4 = hsc::unhopf_f32(hsc::bytes{ kZq4.data, kZq4.len });
    sb::require(bq4.is_first());
    for ( usize i = 0; i < 16; ++i ) sb::require(hsc::__f2u(bq4.cast<hsc::fhsc32>().first()[i]), hsc::__f2u(kBq4.data[i]));

    auto bq8 = hsc::unhopf_f32(hsc::bytes{ kZq8.data, kZq8.len });
    sb::require(bq8.is_first());
    for ( usize i = 0; i < 16; ++i ) sb::require(hsc::__f2u(bq8.cast<hsc::fhsc32>().first()[i]), hsc::__f2u(kBq8.data[i]));
  }

  return 1;
}
