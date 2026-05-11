"""
comparison.py
-------------
Runs all three strategies on identical problem instances and produces
a side-by-side comparison table — the main deliverable for the project report.

Strategies compared:
  1. Greedy         — myopic, no look-ahead
  2. Pure Minimax   — adversarial search, no Alpha-Beta pruning
  3. Minimax + AB   — full hybrid adversarial-planning system

Metrics reported (per report §4.5 & §6):
  - Total social welfare (sum of all agents' utility)
  - Individual utility per agent
  - Goal completion rate (agents with utility > 0)
  - Nodes explored in game tree
  - Nodes pruned by Alpha-Beta
  - Pruning efficiency %
  - Wall-clock time (ms)
  - Jain Fairness Index (JFI)
"""

from __future__ import annotations
import copy

from environment import build_default_environment, Environment
from agent import Agent, AGENT_IDS
from simulation import run_simulation, SimResult


# ──────────────────────────────────────────────
#  Jain Fairness Index
# ──────────────────────────────────────────────

def jain_fairness_index(utilities: list[int]) -> float:
    """
    JFI = (Σxᵢ)² / (n · Σxᵢ²)
    Perfect fairness = 1.0, maximum unfairness = 1/n.
    """
    n = len(utilities)
    if n == 0 or all(u == 0 for u in utilities):
        return 0.0
    sum_x  = sum(utilities)
    sum_x2 = sum(u * u for u in utilities)
    if sum_x2 == 0:
        return 0.0
    return (sum_x ** 2) / (n * sum_x2)


# ──────────────────────────────────────────────
#  Fresh environment + agents builder
# ──────────────────────────────────────────────

def _make_run(strategy: str) -> tuple[Environment, list[Agent]]:
    env    = build_default_environment()
    agents = [Agent(aid, strategy=strategy) for aid in AGENT_IDS]
    return env, agents


# ──────────────────────────────────────────────
#  Table printer
# ──────────────────────────────────────────────

def _bar(n: int, total: int, width: int = 20) -> str:
    if total == 0:
        return "─" * width
    filled = round(n / total * width)
    return "█" * filled + "░" * (width - filled)


def print_comparison(results: list[SimResult]) -> None:
    labels = {
        "greedy":     "Greedy         ",
        "minimax":    "Pure Minimax   ",
        "minimax_ab": "Minimax + AB   ",
    }

    W = 58
    print("\n" + "═" * W)
    print("  MARAS — STRATEGY COMPARISON REPORT")
    print("═" * W)

    # ── Header row ──
    col_w = 17
    print(f"\n  {'Metric':<26}", end="")
    for r in results:
        print(f"  {labels[r.strategy]}", end="")
    print()
    print("  " + "─" * (26 + (col_w + 2) * len(results)))

    # ── Social welfare ──
    max_possible = sum(t.value for t in build_default_environment().tasks.values())
    print(f"\n  {'Total Social Welfare':<26}", end="")
    for r in results:
        print(f"  {r.total_utility:>5d} / {max_possible}       ", end="")
    print()

    # Welfare bar
    print(f"  {'':26}", end="")
    for r in results:
        print(f"  {_bar(r.total_utility, max_possible)}  ", end="")
    print()

    # ── Per-agent utility ──
    print(f"\n  {'Per-Agent Utility':<26}")
    for i, aid in enumerate(AGENT_IDS):
        print(f"  {'  ' + aid:<26}", end="")
        for r in results:
            u = r.agent_summaries[i]["utility"]
            tasks = ",".join(r.agent_summaries[i]["tasks_claimed"])
            print(f"  {u:>3d}  ({tasks:<8s})    ", end="")
        print()

    # ── Goal completion ──
    print(f"\n  {'Goals Met (agents>0)':<26}", end="")
    for r in results:
        rate = r.goals_met() / len(AGENT_IDS) * 100
        print(f"  {r.goals_met()}/{len(AGENT_IDS)} agents ({rate:.0f}%)   ", end="")
    print()

    # ── Jain Fairness Index ──
    print(f"\n  {'Jain Fairness Index':<26}", end="")
    for r in results:
        utils = [s["utility"] for s in r.agent_summaries]
        jfi = jain_fairness_index(utils)
        print(f"  {jfi:.3f}              ", end="")
    print()

    # ── Game tree stats ──
    print(f"\n  {'Nodes Explored':<26}", end="")
    for r in results:
        ne = r.total_nodes_explored()
        print(f"  {ne:>6d}             ", end="")
    print()

    print(f"  {'Nodes Pruned':<26}", end="")
    for r in results:
        np_ = r.total_nodes_pruned()
        print(f"  {np_:>6d}             ", end="")
    print()

    print(f"  {'Pruning Efficiency':<26}", end="")
    for r in results:
        ne  = r.total_nodes_explored()
        np_ = r.total_nodes_pruned()
        total = ne + np_
        eff = (np_ / total * 100) if total > 0 else 0.0
        print(f"  {eff:>5.1f}%             ", end="")
    print()

    # ── Time ──
    print(f"\n  {'Time Elapsed (ms)':<26}", end="")
    for r in results:
        print(f"  {r.elapsed_ms:>7.2f} ms        ", end="")
    print()

    # ── Rounds ──
    print(f"\n  {'Rounds Played':<26}", end="")
    for r in results:
        print(f"  {r.rounds_played:>2d}                 ", end="")
    print()

    print("\n" + "═" * W)


# ──────────────────────────────────────────────
#  Allocation trace printer
# ──────────────────────────────────────────────

def print_allocation_trace(result: SimResult) -> None:
    """Show which agent got which task each round for one strategy."""
    label = {
        "greedy":     "GREEDY",
        "minimax":    "PURE MINIMAX",
        "minimax_ab": "MINIMAX + ALPHA-BETA",
    }.get(result.strategy, result.strategy)

    print(f"\n  Allocation trace — {label}")
    print(f"  {'Round':<8} {'Agent':<6} {'Wanted':<8} {'Got':<8}")
    print("  " + "─" * 32)
    for rlog in result.round_logs:
        for action in rlog["actions"]:
            w = action["wanted"] or "—"
            g = action["got"]    or "✗"
            print(f"  {rlog['round']:<8} {action['agent']:<6} {w:<8} {g:<8}")
    print()


# ──────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────

def run_comparison(verbose_sims: bool = True) -> list[SimResult]:
    strategies = ["greedy", "minimax", "minimax_ab"]
    results    = []

    for strat in strategies:
        env, agents = _make_run(strat)
        result = run_simulation(env, agents, verbose=verbose_sims)
        results.append(result)

    print_comparison(results)
    return results
