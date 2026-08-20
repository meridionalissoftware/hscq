// 07_comptime.cpp
// See also:
//   examples/01_quickstart.cpp — the runtime surface
//   tests/comptime.cpp         — the proof this example is standing on
//
// The whole codec is constexpr. hsc::ct runs encode and decode inside constant evaluation, with
// transient arenas instead of the runtime scratch, and tests/comptime.cpp proves the consteval
// stream BYTE-IDENTICAL to the runtime one in both directions.
//
// Build (from the repo root):
//   duck batch examples.duck && ./bin/07_comptime

#include "_ex_common.hpp"

#include <micron/std.hpp>

namespace
{

// a) bytes, baked
// hsc::ct::str is the NTTP carrier: the payload has to be a structural template argument, because
// the codec runs in the type system
inline constexpr hsc::ct::str k_body{ "the quick brown fox jumps over the lazy dog, hopfed at compile time and unhopfed "
                                      "again, so that the stream is long enough for the 48-byte frame to stop dominating "
                                      "the ratio -- at 67 bytes the container costs more than the payload saves, which is "
                                      "worth seeing once before you compress anything small." };
// dim 4 / L13: the cell the corpus sweep measures bit-exact (BENCHMARKS.md prints psnr 99.99, which
// is the bench's clamp for rmse < 1e-9 -- an integer rmse is either 0 or >= 1/sqrt(n), so it means
// exact). Level is the axis that buys fidelity; dim is not -- kappa(dim) ~= 0.29*sqrt(dim), so a
// WIDER block is less accurate at the same d, and hsc::ct's fixed arenas cap dim8 at L11 anyway.
inline constexpr hsc::hopf_opts k_opts = hsc::opts_exact_bytes;
inline constexpr auto k_z = hsc::ct::hopf<k_body, k_opts>();
inline constexpr auto k_back = hsc::ct::unhopf<k_z>();

static_assert(k_z.len > hsc::k_header_size + hsc::k_trailer_size);
static_assert(k_z.len <= hsc::bound(k_body.len, k_opts));
static_assert(k_back.len == k_body.len);

// checked at compile time, so a build in which the text came back altered never links. exact_bytes()
// is the library's claim about the CELL -- what the corpus sweep measured -- and ct::exact is the
// proof for THIS payload, which is the one that matters: SCHF is packing, not covering, so the cell
// being measured exact is an envelope, not a guarantee about arbitrary bytes.
static_assert(hsc::exact_bytes(k_opts));
static_assert(hsc::ct::max_byte_err<k_body, k_back>() == 0);
static_assert(hsc::ct::exact<k_body, k_back>());

// b) floats, baked
inline constexpr auto k_vec = []() {
  hsc::ct::f32s<32> f{};
  for ( usize i = 0; i < 32; ++i ) f.data[i] = static_cast<f32>(static_cast<f64>(i % 7) * 0.25 - 0.75);
  f.len = 32;
  return f;
}();
inline constexpr hsc::hopf_opts k_vopts{ .m = hsc::mode::vec, .level = 6, .dim_log2 = 2, .gain_bits = 8 };
inline constexpr auto k_vz = hsc::ct::hopf<k_vec, k_vopts>();
inline constexpr auto k_vback = hsc::ct::unhopf_f32<k_vz>();

static_assert(k_vback.len == 32);

// c) the skeleton, probed at compile time
// every stream-defining integer comes out of d_q, and the compiler can derive it. These are the
// golden cardinalities tests/s3.cpp pins the runtime kernels against.
static_assert(hsc::ct::s3_m<hsc::level_dq(3)>() == 138);
static_assert(hsc::ct::s3_m<hsc::level_dq(6)>() == 2588);
static_assert(hsc::ct::tree_m_mod<3, hsc::level_dq(3)>() == 2310);

};      // namespace

int
main()
{
  // a) it is all already in the binary
  ex::head("baked at compile time");

  mc::echo("source  = ", k_body.len, " bytes");
  mc::echo("stream  = ", k_z.len, " bytes, computed by the compiler");
  mc::echo("decoded = ", k_back.len, " bytes, also by the compiler");
  ex::line3("ratio   = ", static_cast<f64>(k_body.len) / static_cast<f64>(k_z.len), "x");
  ex::line3("payload = ", static_cast<f64>(k_body.len) / static_cast<f64>(k_z.len - 48), "x without the 48-byte frame");
  mc::echo("(both below 1: this EXPANDS. the 48-byte frame is one reason and the small input is");
  mc::echo(" another, but the payload alone is 10.75 bits per byte -- exactness costs rate, and this");
  mc::echo(" cell spends more bits than the byte it reproduces. never quote it as compression.)");

  mc::echon("text back: \"");
  for ( usize i = 0; i < k_back.len; ++i ) {
    const char c[2] = { static_cast<char>(k_back.data[i]), '\0' };
    mc::echon(c);
  }
  mc::echo("\"");
  mc::echo("(byte for byte the source string: hsc::ct::exact<> above proved it at compile time)");

  // b) and the runtime agrees, byte for byte
  // this is the contract tests/comptime.cpp asserts across the whole matrix; here it is once,
  // visibly, so the claim is not just a comment
  ex::head("consteval == runtime");

  mc::vector<u8> src;
  src.reserve(k_body.len + 1);
  for ( usize i = 0; i < k_body.len; ++i ) src.push_back(k_body.data[i]);
  const hsc::fhsc rt = hsc::hopf(src, k_opts);

  bool same = rt.size() == k_z.len;
  if ( same )
    for ( usize i = 0; i < k_z.len; ++i )
      if ( rt.begin()[i] != k_z.data[i] ) same = false;
  mc::echo("runtime stream  = ", rt.size(), " bytes");
  mc::echo("comptime stream = ", k_z.len, " bytes");
  mc::echo("byte-identical  : ", same ? "yes" : "NO -- the determinism contract is broken");

  // c) the float lane, and the same check through the bit patterns
  ex::head("floats");

  mc::vector<f32> fsrc;
  fsrc.reserve(k_vec.len + 1);
  for ( usize i = 0; i < k_vec.len; ++i ) fsrc.push_back(k_vec.data[i]);
  auto rv = hsc::hopf(fsrc, k_vopts);
  if ( rv.is_first() ) {
    mc::vector<f32> rb;
    rb.reserve(k_vec.len + 1);
    rb.resize(k_vec.len);
    const hsc::fhsc &z = rv.cast<hsc::fhsc>();
    if ( hsc::unhopf(hsc::bytes{ z.begin(), z.size() }, hsc::wfloats{ rb.begin(), rb.size() }).is_first() ) {
      bool bits = true;
      for ( usize i = 0; i < k_vec.len; ++i )
        if ( hsc::__f2u(rb[i]) != hsc::__f2u(k_vback.data[i]) ) bits = false;
      mc::echo("32 f32 -> ", z.size(), " bytes; decode bit-identical to the consteval one: ", bits ? "yes" : "NO");
      mc::echo("(compared through __f2u, not ==: one ulp apart would be a format bug, not rounding)");
    }
  }

  // d) skeleton cardinalities, straight out of the type system
  ex::head("skeleton probes");

  mc::echo("M(S^3) at L3 = ", hsc::ct::s3_m<hsc::level_dq(3)>());
  mc::echo("M(S^3) at L6 = ", hsc::ct::s3_m<hsc::level_dq(6)>());
  mc::echo("M mod 2^64, dim 8 L3 = ", hsc::ct::tree_m_mod<3, hsc::level_dq(3)>());
  mc::echo("no codebook is stored anywhere: both sides rebuild these from d_q alone,");
  mc::echo("and every stream carries M mod 2^64 so a drifted decoder fails bad_skeleton");
  mc::echo("instead of quietly emitting the wrong floats.");

  return 0;
}
