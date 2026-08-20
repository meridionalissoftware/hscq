

#pragma once

#include "../../micron/external/bbench/bench.hpp"

#include <micron/io/console.hpp>
#include <micron/io/stdout.hpp>
#include <micron/linux/sys/sched.hpp>
#include <micron/std.hpp>

namespace mb
{

constexpr u32 K_MEASUREMENTS = 7;
constexpr u64 WARMUP_REPS = 4;
constexpr u64 TARGET_BYTES_PER_MEAS = 1ULL << 22;
constexpr u64 MIN_REPS = 32;
constexpr u64 MAX_REPS = 1ULL << 18;

using mem_events = bbench::event_group<bbench::hardware_cycles, bbench::hardware_instructions, bbench::branches, bbench::branch_misses>;

struct row {
  const char *op;
  const char *impl;
  u64 size;
  f64 cyc_per_op;
  f64 ipc;
  f64 bmiss_rate;
};

struct fmt2 {
  u64 whole;
  u32 frac_x100;
};

[[gnu::always_inline]] inline fmt2
to_fmt2(f64 v)
{
  if ( v < 0 ) v = 0;
  u64 scaled = static_cast<u64>(v * 100.0 + 0.5);
  return { scaled / 100, static_cast<u32>(scaled % 100) };
}

struct line {
  char buf[256];
  u32 pos;

  constexpr line() noexcept : pos(0) { }

  void
  s(const char *p) noexcept
  {
    while ( *p ) buf[pos++] = *p++;
  }

  // a field that already reaches (or overruns) its column still gets one separating space --
  // otherwise a wide value runs straight into the next one ("11915.671.05").
  void
  pad_to(u32 end_col, u32 written) noexcept
  {
    const u32 want = end_col >= written ? end_col - written : 0;
    if ( want <= pos )
      buf[pos++] = ' ';
    else
      while ( pos < want ) buf[pos++] = ' ';
  }

  void
  u_at(u64 v, u32 end_col) noexcept
  {
    char tmp[24];
    u32 n = 0;
    if ( v == 0 )
      tmp[n++] = '0';
    else {
      u64 vv = v;
      while ( vv ) {
        tmp[n++] = '0' + (vv % 10);
        vv /= 10;
      }
    }
    pad_to(end_col, n);
    while ( n ) buf[pos++] = tmp[--n];
  }

  void
  f2_at(fmt2 f, u32 end_col) noexcept
  {
    char tmp[24];
    u32 n = 0;
    u64 w = f.whole;
    if ( w == 0 )
      tmp[n++] = '0';
    else
      while ( w ) {
        tmp[n++] = '0' + (w % 10);
        w /= 10;
      }
    pad_to(end_col, n + 3);
    while ( n ) buf[pos++] = tmp[--n];
    buf[pos++] = '.';
    buf[pos++] = '0' + static_cast<char>(f.frac_x100 / 10);
    buf[pos++] = '0' + static_cast<char>(f.frac_x100 % 10);
  }

  void
  s_at(const char *p, u32 end_col) noexcept
  {
    u32 n = 0;
    while ( p[n] ) ++n;
    pad_to(end_col, n);
    while ( *p ) buf[pos++] = *p++;
  }

  // NOTE: a field wider than its column still gets one separating space -- without this an
  // over-long impl name runs straight into the next number ("quantize refine1753.45").
  void
  s_lj_at(const char *p, u32 end_col) noexcept
  {
    while ( *p ) buf[pos++] = *p++;
    if ( pos >= end_col )
      buf[pos++] = ' ';
    else
      while ( pos < end_col ) buf[pos++] = ' ';
  }

  const char *
  str() noexcept
  {
    buf[pos] = '\0';
    return buf;
  }
};

[[gnu::cold]] inline void
print_header()
{
  line h;
  h.s_at("size(B)", 10);
  h.pad_to(12, 0);
  h.s_lj_at("op", 40);
  h.s_lj_at("impl", 50);
  h.s_at("cyc/op", 62);
  h.s_at("IPC", 72);
  h.s_at("bmiss%", 82);
  micron::io::println(h.str());
  micron::io::println("----------------------------------------------------------------------------------");
}

[[gnu::cold]] inline void
print_row(const row &r)
{
  fmt2 cpo = to_fmt2(r.cyc_per_op);
  fmt2 ipc = to_fmt2(r.ipc);
  fmt2 bm = to_fmt2(r.bmiss_rate * 100.0);
  line ln;
  ln.u_at(r.size, 10);
  ln.pad_to(12, 0);
  ln.s_lj_at(r.op, 40);
  ln.s_lj_at(r.impl, 50);
  ln.f2_at(cpo, 62);
  ln.f2_at(ipc, 72);
  ln.f2_at(bm, 82);
  micron::io::println(ln.str());
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%
// throughput reporting: cyc/byte next to MB/s, encode next to decode. cyc/op alone hides the
// absolute rate, which is the number a consumer actually budgets against.

struct tput {
  const char *corpus;
  const char *impl;
  u64 out_bytes;
  f64 ratio;      // in / out
  f64 enc_cyc_b;
  u64 enc_mbps;
  f64 dec_cyc_b;
  u64 dec_mbps;
  f64 ipc;
  f64 bmiss_rate;
  f64 ghz;      // cycles/ns observed during the encode cell -- MB/s is only comparable at equal clock
};

[[gnu::cold]] inline void
print_tput_header()
{
  line h;
  h.s_lj_at("corpus", 14);
  h.s_lj_at("impl", 34);
  h.s_at("out(B)", 44);
  h.s_at("ratio", 52);
  h.s_at("e cyc/B", 62);
  h.s_at("e MB/s", 71);
  h.s_at("d cyc/B", 81);
  h.s_at("d MB/s", 90);
  h.s_at("IPC", 97);
  h.s_at("bmiss%", 105);
  h.s_at("GHz", 112);
  micron::io::println(h.str());
  micron::io::println("----------------------------------------------------------------------------------------------------------------");
}

[[gnu::cold]] inline void
print_tput_row(const tput &t)
{
  line ln;
  ln.s_lj_at(t.corpus, 14);
  ln.s_lj_at(t.impl, 34);
  ln.u_at(t.out_bytes, 44);
  ln.f2_at(to_fmt2(t.ratio), 52);
  ln.f2_at(to_fmt2(t.enc_cyc_b), 62);
  ln.u_at(t.enc_mbps, 71);
  ln.f2_at(to_fmt2(t.dec_cyc_b), 81);
  ln.u_at(t.dec_mbps, 90);
  ln.f2_at(to_fmt2(t.ipc), 97);
  ln.f2_at(to_fmt2(t.bmiss_rate * 100.0), 105);
  ln.f2_at(to_fmt2(t.ghz), 112);
  micron::io::println(ln.str());
}

inline volatile u64 sink_u64 = 0;

[[gnu::always_inline]] inline void
clobber(const void *p) noexcept
{
  asm volatile("" : : "r"(p) : "memory");
}

[[gnu::always_inline]] inline void
sink_bool(bool b) noexcept
{
  sink_u64 += static_cast<u64>(b);
}

[[gnu::always_inline]] inline void
sink_size(usize v) noexcept
{
  sink_u64 += static_cast<u64>(v);
}

[[gnu::always_inline]] inline void
sink_ptr(const void *p) noexcept
{
  sink_u64 += reinterpret_cast<u64>(p);
}

inline f64
median_f64(f64 *xs, u32 n) noexcept
{
  for ( u32 i = 1; i < n; i++ ) {
    f64 key = xs[i];
    u32 j = i;
    while ( j > 0 && xs[j - 1] > key ) {
      xs[j] = xs[j - 1];
      --j;
    }
    xs[j] = key;
  }
  return xs[n / 2];
}

struct sample {
  u64 cyc;
  u64 inst;
  u64 br;
  u64 bm;
};

template<typename Fn>
[[gnu::noinline]] sample
measure_once(Fn &&fn, u64 reps) noexcept
{
  mem_events evs{ bbench::quiet{} };
  evs.open();
  evs.begin();
  for ( u64 i = 0; i < reps; i++ ) fn();
  evs.end();
  return { static_cast<u64>(evs.get<bbench::hardware_cycles>().retrieve()),
           static_cast<u64>(evs.get<bbench::hardware_instructions>().retrieve()), static_cast<u64>(evs.get<bbench::branches>().retrieve()),
           static_cast<u64>(evs.get<bbench::branch_misses>().retrieve()) };
}

template<typename Fn>
row
bench_one(const char *op, const char *impl, u64 size, u64 bytes_per_op, Fn &&fn, u64 max_reps_override = MAX_REPS) noexcept
{
  for ( u64 i = 0; i < WARMUP_REPS; i++ ) fn();

  u64 reps = TARGET_BYTES_PER_MEAS / (bytes_per_op == 0 ? 1 : bytes_per_op);
  if ( reps < MIN_REPS ) reps = MIN_REPS;
  if ( reps > max_reps_override ) reps = max_reps_override;

  f64 cpo_samples[K_MEASUREMENTS];
  f64 ipc_samples[K_MEASUREMENTS];
  f64 bm_samples[K_MEASUREMENTS];

  for ( u32 m = 0; m < K_MEASUREMENTS; m++ ) {
    sample s = measure_once(fn, reps);
    cpo_samples[m] = static_cast<f64>(s.cyc) / static_cast<f64>(reps);
    ipc_samples[m] = s.cyc > 0 ? static_cast<f64>(s.inst) / static_cast<f64>(s.cyc) : 0.0;
    bm_samples[m] = s.br > 0 ? static_cast<f64>(s.bm) / static_cast<f64>(s.br) : 0.0;
  }
  return row{
    op, impl, size, median_f64(cpo_samples, K_MEASUREMENTS), median_f64(ipc_samples, K_MEASUREMENTS), median_f64(bm_samples, K_MEASUREMENTS)
  };
}

// Busy-spin to pull the core up to its turbo clock before any wall-clock measurement. Without
// this the first rows of a long table are timed at the idle frequency and their MB/s reads low
// against identical cyc/byte -- the giveaway that a number is a clock artefact, not a speedup.
[[gnu::noinline]] inline void
spin_up(u64 ms = 300) noexcept
{
  bbench::time_clock_mono ck{};
  ck.begin();
  u64 acc = 0;
  for ( ;; ) {
    for ( u64 i = 0; i < (1u << 16); i++ ) acc += i * i + (acc >> 3);
    ck.end();
    if ( ck.template elapsed<bbench::time_resolution::ms>() >= static_cast<f64>(ms) ) break;
  }
  sink_u64 += acc;
}

inline void
pin_cpu0()
{
  micron::posix::cpu_set_t set;
  set.cpu_zero();
  set.cpu_set(0);
  micron::posix::sched_setaffinity(0, sizeof(set), set);
}

template<typename Fn>
f64
time_one_ns(Fn &&fn, u64 reps) noexcept
{
  f64 samples[K_MEASUREMENTS];
  for ( u32 m = 0; m < K_MEASUREMENTS; m++ ) {
    bbench::time_clock_mono ck{};
    ck.begin();
    for ( u64 i = 0; i < reps; i++ ) fn();
    ck.end();
    samples[m] = ck.template elapsed<bbench::time_resolution::ns>() / static_cast<f64>(reps);
  }
  return median_f64(samples, K_MEASUREMENTS);
}

inline u64
mbps(u64 bytes_per_op, f64 ns_per_op) noexcept
{
  if ( ns_per_op <= 0.0 ) return 0;
  return static_cast<u64>((static_cast<f64>(bytes_per_op) / ns_per_op) * 1000.0);
}

struct lat {
  u64 min_ns;
  u64 p50_ns;
  u64 p90_ns;
  u64 p99_ns;
  u64 max_ns;
};

inline constexpr u32 LAT_SAMPLES = 2000;
inline f64 lat_buf[LAT_SAMPLES];

template<typename Fn>
lat
latency_one(Fn &&fn, u32 samples = LAT_SAMPLES) noexcept
{
  if ( samples > LAT_SAMPLES ) samples = LAT_SAMPLES;
  for ( u32 i = 0; i < 16; i++ ) fn();
  for ( u32 i = 0; i < samples; i++ ) {
    bbench::time_clock_mono ck{};
    ck.begin();
    fn();
    ck.end();
    lat_buf[i] = ck.template elapsed<bbench::time_resolution::ns>();
  }

  for ( u32 i = 1; i < samples; i++ ) {
    const f64 key = lat_buf[i];
    u32 j = i;
    while ( j > 0 && lat_buf[j - 1] > key ) {
      lat_buf[j] = lat_buf[j - 1];
      --j;
    }
    lat_buf[j] = key;
  }
  auto at = [&](f64 q) -> u64 {
    u32 idx = (u32)(q * (f64)(samples - 1));
    return (u64)lat_buf[idx];
  };
  return lat{ (u64)lat_buf[0], at(0.50), at(0.90), at(0.99), (u64)lat_buf[samples - 1] };
}

struct soak_result {
  f64 mbps_decile[10];
  f64 mbps_median;
  f64 drift_pct;
  u64 total_reps;
};

inline constexpr u32 SOAK_MAX_SLICES = 4096;
inline f64 soak_slice_mbps[SOAK_MAX_SLICES];

template<typename Fn>
soak_result
soak_one(Fn &&fn, u64 bytes_per_op, u64 target_ms = 5000) noexcept
{
  for ( u64 i = 0; i < WARMUP_REPS; i++ ) fn();

  bbench::time_clock_mono cal{};
  cal.begin();
  fn();
  cal.end();
  f64 ns_call = cal.template elapsed<bbench::time_resolution::ns>();
  if ( ns_call < 1.0 ) ns_call = 1.0;
  u64 slice_reps = static_cast<u64>(20.0e6 / ns_call);
  if ( slice_reps < 1 ) slice_reps = 1;

  u32 slices = 0;
  u64 total_reps = 0;
  f64 elapsed_ns = 0.0;
  const f64 target_ns = static_cast<f64>(target_ms) * 1.0e6;
  while ( elapsed_ns < target_ns && slices < SOAK_MAX_SLICES ) {
    bbench::time_clock_mono ck{};
    ck.begin();
    for ( u64 i = 0; i < slice_reps; i++ ) fn();
    ck.end();
    const f64 ns = ck.template elapsed<bbench::time_resolution::ns>();
    soak_slice_mbps[slices++] = ns > 0.0 ? (static_cast<f64>(bytes_per_op * slice_reps) / ns) * 1000.0 : 0.0;
    total_reps += slice_reps;
    elapsed_ns += ns;
  }

  soak_result r{};
  r.total_reps = total_reps;
  const u32 per = slices >= 10 ? slices / 10 : 1;
  for ( u32 d = 0; d < 10; d++ ) {
    const u32 lo = d * per;
    u32 hi = (d == 9) ? slices : (d + 1) * per;
    if ( lo >= slices ) {
      r.mbps_decile[d] = 0.0;
      continue;
    }
    if ( hi > slices ) hi = slices;
    f64 sum = 0.0;
    for ( u32 i = lo; i < hi; i++ ) sum += soak_slice_mbps[i];
    r.mbps_decile[d] = sum / static_cast<f64>(hi - lo);
  }
  r.mbps_median = median_f64(soak_slice_mbps, slices ? slices : 1);
  r.drift_pct = r.mbps_decile[0] > 0.0 ? (r.mbps_decile[0] - r.mbps_decile[9]) / r.mbps_decile[0] * 100.0 : 0.0;
  return r;
}

[[gnu::cold]] inline void
print_soak_row(const char *op, const char *impl, const soak_result &r)
{
  line ln;
  ln.s_lj_at(op, 28);
  ln.s_lj_at(impl, 44);
  ln.u_at(static_cast<u64>(r.mbps_median), 54);
  ln.u_at(static_cast<u64>(r.mbps_decile[0]), 64);
  ln.u_at(static_cast<u64>(r.mbps_decile[9]), 74);
  ln.f2_at(to_fmt2(r.drift_pct < 0 ? -r.drift_pct : r.drift_pct), 84);
  ln.s(r.drift_pct < 0 ? " (rising)" : "");
  micron::io::println(ln.str());
}

[[gnu::cold]] inline void
print_soak_header()
{
  line h;
  h.s_lj_at("soak op", 28);
  h.s_lj_at("impl", 44);
  h.s_at("MB/s", 54);
  h.s_at("d0", 64);
  h.s_at("d9", 74);
  h.s_at("drift%", 84);
  micron::io::println(h.str());
  micron::io::println("------------------------------------------------------------------------------------");
}

[[gnu::cold]] inline void
print_preamble(const char *title)
{
  micron::io::println("=== ", title, " ===");
  micron::io::println("warmup ", WARMUP_REPS, " reps; ", K_MEASUREMENTS, " median samples per cell");
  micron::io::println("perf events: cycles + instructions + branches + branch-misses");
  micron::io::println("");
}

[[gnu::cold]] inline void
print_epilogue()
{
  micron::io::println("");
  micron::io::println("=== done ===");
  micron::io::println("(anti-DCE sink: ", sink_u64, ")");
}

}      // namespace mb
