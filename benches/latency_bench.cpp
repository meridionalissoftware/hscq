// hsc single-call latency — ns percentiles for small payloads, scratch hot vs cold (cold
// includes the whole skeleton + pack-table build; hot is the steady-state cost).
//   duck build benches/latency_bench.cpp --perf --fp -i ../micron -i ../micron/src && ./bin/latency_bench
#include "_corpus.hpp"

#include "_bench_common.hpp"

static u8 g_z[1 << 18];
static u8 g_out[1 << 16];

int
main()
{
  mb::pin_cpu0();
  mb::print_preamble("hsc latency — single 4 KiB call, ns percentiles");
  hc::generate();

  const hsc::hopf_opts o{ .level = 6, .dim_log2 = 3 };
  hsc::hopf_scratch hot;
  (void)hsc::hopf_into(hsc::bytes{ hc::g_smooth, 4096 }, o, g_z, sizeof(g_z), hot);

  {
    auto l = mb::latency_one(
        [&]() {
          const usize zn = hsc::hopf_into(hsc::bytes{ hc::g_smooth, 4096 }, o, g_z, sizeof(g_z), hot);
          mb::sink_size(zn);
        },
        512);
    micron::io::println("hopf 4KiB hot   : p50=", l.p50_ns, "ns p90=", l.p90_ns, "ns p99=", l.p99_ns, "ns max=", l.max_ns, "ns");
  }
  {
    auto l = mb::latency_one(
        [&]() {
          hsc::hopf_scratch cold;
          const usize zn = hsc::hopf_into(hsc::bytes{ hc::g_smooth, 4096 }, o, g_z, sizeof(g_z), cold);
          mb::sink_size(zn);
        },
        128);
    micron::io::println("hopf 4KiB cold  : p50=", l.p50_ns, "ns p90=", l.p90_ns, "ns p99=", l.p99_ns, "ns max=", l.max_ns, "ns");
  }
  {
    const usize zn = hsc::hopf_into(hsc::bytes{ hc::g_smooth, 4096 }, o, g_z, sizeof(g_z), hot);
    auto l = mb::latency_one(
        [&]() {
          auto r = hsc::unhopf(hsc::bytes{ g_z, zn }, hsc::wbytes{ g_out, sizeof(g_out) }, hot);
          mb::sink_bool(r.is_first());
        },
        512);
    micron::io::println("unhopf 4KiB hot : p50=", l.p50_ns, "ns p90=", l.p90_ns, "ns p99=", l.p99_ns, "ns max=", l.max_ns, "ns");
  }

  return 0;
}
