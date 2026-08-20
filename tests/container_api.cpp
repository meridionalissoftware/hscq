// container overloads: the byte_source/byte_sink seam plus hsc's f32_source/f32_sink twins.
// Proves the compile-time contract (what the seams admit and reject), the as_bytes/as_floats
// view identities, that container verb forms are byte-identical to the bytes/floats forms they
// forward to, and -- the compatibility linchpin -- that braced-init/raw_slice call sites are
// NOT hijacked by the container overloads.

#include "../src/hsc/hsc.hpp"
#include "tutil.hpp"

#include <micron/array/array.hpp>
#include <micron/span.hpp>
#include <micron/std.hpp>
#include <micron/string/istring.hpp>
#include <micron/string/rope.hpp>
#include <micron/string/strings.hpp>
#include <micron/vector/fvector.hpp>
#include <micron/vector/svector.hpp>
#include <micron/vector/vector.hpp>

#include <snowball/snowball.hpp>

namespace
{

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// the concept must admit containers and strings and reject the raw_slice views -- that
// disjointness is what makes the overload sets unambiguous, so it is a hard compile-time
// contract, not a runtime check. This block IS the documentation of what the seam accepts.

// contiguous containers (micron's contiguous_tag family)
static_assert(hsc::byte_source<micron::vector<u8>>);
static_assert(hsc::byte_source<micron::vector<u32>>);
static_assert(hsc::byte_source<micron::fvector<u8>>);
static_assert(hsc::byte_source<micron::svector<u8, 64>>);      // stack vector
static_assert(hsc::byte_source<micron::array<u32, 16>>);
static_assert(hsc::byte_source<micron::span<u8, 32>>);
static_assert(hsc::byte_source<hsc::fhsc>);      // slice<u8>: a decode result feeds straight back in

// strings. hstring/sstring reach us through is_iterable_container as well; rope reaches us ONLY
// through the is_string arm.
static_assert(hsc::byte_source<micron::string>);
static_assert(hsc::byte_source<micron::sstring<128, char>>);
static_assert(hsc::byte_source<micron::rope<>>);

// rejected, and why
static_assert(!hsc::byte_source<hsc::bytes>);       // raw_slice<const u8>: no .data(), no size_type
static_assert(!hsc::byte_source<hsc::wbytes>);      // raw_slice<u8>
static_assert(!hsc::byte_source<hsc::floats>);      // raw_slice<const f32>: the float views stay disjoint too
static_assert(!hsc::byte_source<hsc::wfloats>);
static_assert(!hsc::byte_source<micron::istring<>>);      // const-only iteration satisfies neither concept

// const-hostility of is_iterable_container is absorbed by remove_cvref_t
static_assert(hsc::byte_source<const micron::vector<u8> &>);
static_assert(hsc::byte_source<const micron::string &>);

// the write side is strictly narrower: a const-qualified pointer can be read but never written into
static_assert(hsc::byte_sink<micron::vector<u8>>);
static_assert(hsc::byte_sink<micron::string>);
static_assert(hsc::byte_sink<micron::array<u32, 16>>);
static_assert(!hsc::byte_sink<micron::rope<>>);      // pointer is const T *

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// the f32 lane: only 4-byte floating element containers, so a vector<f32> selects the typed
// overloads and every integer/8-byte container falls through to byte_source

static_assert(hsc::f32_source<micron::vector<f32>>);
static_assert(hsc::f32_source<micron::array<f32, 16>>);
static_assert(hsc::f32_source<micron::fvector<f32>>);
static_assert(hsc::f32_source<const micron::vector<f32> &>);
static_assert(hsc::f32_sink<micron::vector<f32>>);

static_assert(!hsc::f32_source<micron::vector<u32>>);      // 4 bytes but not floating
static_assert(!hsc::f32_source<micron::vector<f64>>);      // floating but not 4 bytes
static_assert(!hsc::f32_source<micron::vector<u8>>);
static_assert(!hsc::f32_source<hsc::floats>);      // the view stays a view
static_assert(!hsc::f32_source<micron::string>);

// every f32_source is a byte_source (byte entry points remain reachable by explicit view)
static_assert(hsc::byte_source<micron::vector<f32>>);

}      // namespace

int
main()
{
  sb::test_case("as_bytes / as_wbytes view identity");
  {
    micron::vector<u32> v;
    v.reserve(65);
    for ( u32 i = 0; i < 64; ++i ) v.push_back(i * 0x01010101u);
    const hsc::bytes b = hsc::as_bytes(v);
    sb::require(b.size() == 64 * sizeof(u32));
    sb::require(b.ptr == reinterpret_cast<const u8 *>(v.data()));

    hsc::wbytes w = hsc::as_wbytes(v);
    sb::require(w.size() == b.size());
    sb::require(w.ptr == reinterpret_cast<u8 *>(v.data()));
  }

  sb::test_case("as_floats / as_wfloats view identity and element count");
  {
    micron::vector<f32> v;
    v.reserve(33);
    for ( u32 i = 0; i < 32; ++i ) v.push_back(static_cast<f32>(i) * 0.25f);
    const hsc::floats f = hsc::as_floats(v);
    sb::require(f.size() == 32);      // elements, not bytes
    sb::require(f.ptr == reinterpret_cast<const f32 *>(v.data()));
    sb::require(f.ptr[7] == 1.75f);

    hsc::wfloats wf = hsc::as_wfloats(v);
    sb::require(wf.size() == 32);
    wf.ptr[3] = -2.5f;
    sb::require(v[3] == -2.5f);
  }

  sb::test_case("f2u/u2f punning is bit-exact both ways");
  {
    sb::require(hsc::__f2u(0.0f) == 0u);
    sb::require(hsc::__f2u(1.0f) == 0x3F800000u);
    sb::require(hsc::__u2f(0x40490FDBu) > 3.14159f);
    tutil::rng g;
    for ( i32 i = 0; i < 1000; ++i ) {
      const u32 u = static_cast<u32>(g.next());
      // NaN patterns must survive the round-trip bit-exactly too
      sb::require(hsc::__f2u(hsc::__u2f(u)) == u);
    }
  }

  sb::test_case("container forms are byte-identical to the bytes forms they forward to");
  {
    micron::vector<u8> src;
    src.reserve(2049);
    tutil::rng g;
    for ( usize i = 0; i < 2048; ++i ) src.push_back(static_cast<u8>(g.next()));
    const hsc::bytes view{ src.cbegin(), src.size() };
    for ( i32 lvl : { 3, 6, 8 } ) {
      hsc::fhsc a = hsc::hopf(src, hsc::hopf_opts{ .level = lvl });       // container overload
      hsc::fhsc e = hsc::hopf(view, hsc::hopf_opts{ .level = lvl });      // the bytes overload
      sb::require(a.size() == e.size());
      sb::require(tutil::bytes_equal({ a.first(), a.size() }, { e.first(), e.size() }));
    }
  }

  sb::test_case("a vector<f32> selects the typed float lane and matches the floats form");
  {
    micron::vector<f32> src;
    src.reserve(257);
    tutil::rng g;
    for ( usize i = 0; i < 256; ++i ) src.push_back(static_cast<f32>(g.unit()));
    const hsc::hopf_opts o{ .m = hsc::mode::vec, .level = 6, .dim_log2 = 3 };
    auto a = hsc::hopf(src, o);                      // f32_source overload
    auto e = hsc::hopf(hsc::as_floats(src), o);      // the floats overload it forwards to
    sb::require(a.is_first());
    sb::require(e.is_first());
    auto &za = a.cast<hsc::fhsc>();
    auto &ze = e.cast<hsc::fhsc>();
    sb::require(za.size() == ze.size());
    sb::require(tutil::bytes_equal({ za.first(), za.size() }, { ze.first(), ze.size() }));
    // and the stream is a float-mode stream, not a punned byte stream
    auto pi = hsc::hopf_probe(hsc::bytes{ za.first(), za.size() });
    sb::require(pi.is_first());
    sb::require(pi.cast<hsc::hopf_info>().m == hsc::mode::vec);
  }

  sb::test_case("braced-init and raw_slice call sites are NOT hijacked");
  {
    micron::vector<u8> src;
    src.reserve(513);
    tutil::rng g;
    for ( usize i = 0; i < 512; ++i ) src.push_back(static_cast<u8>(g.next()));
    // braced-init cannot deduce a template parameter, so this binds the bytes overload
    hsc::fhsc a = hsc::hopf({ src.cbegin(), src.size() }, hsc::hopf_opts{});
    hsc::fhsc e = hsc::hopf(src, hsc::hopf_opts{});
    sb::require(tutil::bytes_equal({ a.first(), a.size() }, { e.first(), e.size() }));
    // an fhsc (slice<u8>) decode result feeds straight back in as a byte_source
    hsc::fhsc twice = hsc::hopf(a, hsc::hopf_opts{});
    sb::require(twice.size() > 0);
  }

  sb::test_case("container sinks: byte and f32 output containers both bind");
  {
    micron::vector<u8> src;
    src.reserve(257);
    tutil::rng g;
    for ( usize i = 0; i < 256; ++i ) src.push_back(static_cast<u8>(g.next()));
    hsc::fhsc z = hsc::hopf(src, hsc::hopf_opts{ .level = 6, .dim_log2 = 3 });

    micron::vector<u8> out;
    out.reserve(257);
    out.resize(256);
    auto r = hsc::unhopf(hsc::bytes{ z.first(), z.size() }, out);
    sb::require(r.is_first());
    sb::require(r.cast<usize>(), 256ull);

    micron::vector<f32> fsrc;
    fsrc.reserve(129);
    for ( usize i = 0; i < 128; ++i ) fsrc.push_back(static_cast<f32>(g.unit()));
    auto zf = hsc::hopf(fsrc, hsc::hopf_opts{ .m = hsc::mode::vec, .dim_log2 = 3 });
    sb::require(zf.is_first());
    auto &zff = zf.cast<hsc::fhsc>();
    micron::vector<f32> fout;
    fout.reserve(129);
    fout.resize(128);
    auto rf = hsc::unhopf(hsc::bytes{ zff.first(), zff.size() }, fout);
    sb::require(rf.is_first());
    sb::require(rf.cast<usize>(), 128ull);      // element count through the f32_sink lane
  }

  return 1;
}
