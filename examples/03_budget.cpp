// 03_budget.cpp
// See also:
//   examples/02_tensor.cpp      — the plan, field by field
//   examples/04_embeddings.cpp  — spending a budget on a real retrieval task
//
// hsc is fixed rate, which changes the question you get to ask. With an entropy coder you compress
// and then find out what you got. Here the size is a closed-form function of (mode, dim, d_q,
// gain_bits) and the shape -- so the useful question is not "what ratio did I get?" but "what does
// this quality target cost?", and it is answerable before the data exists
//
// Nothing in this example runs the codec. Not one float is compressed.
//
// Build (from the repo root):
//   duck batch examples.duck && ./bin/03_budget

#include "_ex_common.hpp"

#include <micron/std.hpp>

int
main()
{
  hsc::hopf_scratch sc;
  const usize rows = 4096, cols = 384;      // a plausible embedding table, never allocated

  // a) preset ladder
  // named cells to pick from instead of solving for. bits_per_weight here is the exact per-block
  // accounting, pinned against hsc::rate() by tests/quant.cpp so it cannot drift into a lie.
  ex::head("presets");

  mc::echo("name        dim   L    bits/w   est rel-rmse   vs fp32");
  for ( usize i = 0; i < hsc::k_qpresets; ++i ) {
    const hsc::qpreset &q = hsc::q_presets[i];
    ex::pad(q.name, 12);
    ex::padnum(1u << q.dim_log2, 6);
    mc::echon("L");
    ex::padnum(static_cast<u64>(q.level), 5);
    ex::padf3(q.bits_per_weight, 9);
    ex::padf3(q.est_rel_rmse, 15);
    ex::put3(32.0 / q.bits_per_weight);
    mc::echo("x");
  }
  mc::echo("");
  mc::echo("the ladder walks the DIM axis at L5/L6/L7: dimension gain buys more than a finer");
  mc::echo("distance costs, which is why every entry is on the (rate, error) frontier.");

  // b) pick one off the table
  // no skeleton build, no search, no allocation -- just the constexpr ladder
  ex::head("pick");

  for ( f64 budget : { 1.0, 1.5, 2.5, 3.5, 8.0 } ) {
    const hsc::qpreset *q = hsc::pick(budget);
    mc::echon("budget ");
    ex::put3(budget);
    mc::echo(" bits/weight -> ", q ? q->name : "nothing fits (even q_min is dearer)");
  }

  // c) the exact accounting for a real shape
  // pick()'s number is the per-block rate; plan_for() adds this tensor's row padding and framing,
  // so it is the number that shows up on disk. They differ, and the difference is not hidden.
  ex::head("plan vs preset");

  mc::echo("preset      preset b/w   planned b/w   bytes for ", rows, "x", cols);
  for ( usize i = 0; i < hsc::k_qpresets; ++i ) {
    const hsc::qpreset &qp = hsc::q_presets[i];
    hsc::target tg = hsc::as_target(qp);
    tg.min_dim_log2 = 2;
    tg.max_dim_log2 = 6;      // the ladder names dim 64; the solver's default window stops at 32
    const auto pr = hsc::plan_for(rows, cols, tg, sc);
    ex::pad(qp.name, 12);
    if ( !pr.is_first() ) {
      mc::echo("refused: ", hsc::error_name(pr.cast<hsc::error>()));
      continue;
    }
    const hsc::qplan &p = pr.cast<hsc::qplan>();
    ex::padf3(qp.bits_per_weight, 13);
    ex::padf3(p.bits_per_weight, 14);
    mc::echo(p.bytes);
  }

  // d) the trade-off surface
  // dim buys rate, level buys accuracy; the solver walks this grid so you do not have to
  ex::head("the dim x level grid");

  mc::echo("        L4      L5      L6      L7        (bits/weight)");
  for ( u32 dl : { 2u, 3u, 4u, 5u } ) {
    mc::echon("dim ");
    ex::padnum(1u << dl, 4);
    for ( i32 lvl : { 4, 5, 6, 7 } ) {
      hsc::target tg{};
      tg.level = lvl;
      tg.dim_log2 = dl;
      const auto pr = hsc::plan_for(rows, cols, tg, sc);
      if ( pr.is_first() )
        ex::put3(pr.cast<hsc::qplan>().bits_per_weight);
      else
        mc::echon("  --  ");
      mc::echon("  ");
    }
    mc::echo("");
  }

  // e) budgets and goals, solved
  ex::head("solve");

  for ( f64 budget : { 1.2, 2.0, 3.0, 4.5 } ) {
    hsc::target tg{};
    tg.bits_per_weight = budget;
    tg.max_dim_log2 = 6;
    const auto pr = hsc::plan_for(rows, cols, tg, sc);
    mc::echon("<= ");
    ex::put3(budget);
    if ( !pr.is_first() ) {
      mc::echo(" b/w -> ", hsc::error_name(pr.cast<hsc::error>()));
      continue;
    }
    const hsc::qplan &p = pr.cast<hsc::qplan>();
    mc::echon(" b/w -> dim ");
    mc::echon(1u << p.dim_log2);
    mc::echon(" L");
    mc::echon(static_cast<i32>(p.level));
    mc::echon("  costs ");
    ex::put3(p.bits_per_weight);
    mc::echon("  est rel-rmse ");
    ex::put3(p.est_rel_rmse);
    mc::echo("");
  }

  for ( f64 goal : { 0.4, 0.2, 0.1 } ) {
    hsc::target tg{};
    tg.rel_rmse = goal;
    tg.max_dim_log2 = 6;
    const auto pr = hsc::plan_for(rows, cols, tg, sc);
    if ( !pr.is_first() ) continue;
    const hsc::qplan &p = pr.cast<hsc::qplan>();
    mc::echon("rel-rmse <= ");
    ex::put3(goal);
    mc::echon(" -> dim ");
    mc::echon(1u << p.dim_log2);
    mc::echon(" L");
    mc::echon(static_cast<i32>(p.level));
    mc::echon("  cheapest at ");
    ex::put3(p.bits_per_weight);
    mc::echo(" bits/weight");
  }

  // f) the cell that is not compression
  ex::head("the cell that lies");

  hsc::target deg{};
  deg.level = 1;
  deg.dim_log2 = 4;
  const auto bad = hsc::plan_for(rows, cols, deg, sc);
  mc::echo("dim 16 L1 -> ", bad.is_first() ? "accepted" : hsc::error_name(bad.cast<hsc::error>()));

  deg.allow_degenerate = true;
  const auto shown = hsc::plan_for(rows, cols, deg, sc);
  if ( shown.is_first() ) {
    const hsc::qplan &p = shown.cast<hsc::qplan>();
    mc::echo("forced:  shape_bits = ", p.shape_bits, "  record_bits = ", p.record_bits);
    ex::line3("         apparent ratio ", p.ratio, "x -- and the direction is GONE");
    ex::line3("         est rel-rmse ", p.est_rel_rmse, "   (sqrt(2) is the no-information level)");
  }

  return 0;
}
