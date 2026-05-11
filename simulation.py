"""
simulation.py
-------------
Main simulation loop for MARAS.

Each round:
  1. Every agent runs its decide() cycle and nominates one task.
  2. The Resource Arbiter resolves conflicts (first-priority order wins).
  3. Env state is updated; tasks are claimed.
  4. Loop until all tasks claimed or T_max rounds reached.

Returns a SimResult with per-round logs and final summaries.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

from environment import Environment
from agent import Agent


T_MAX = 20   # max rounds before forced termination


# ──────────────────────────────────────────────
#  Result container
# ──────────────────────────────────────────────

@dataclass
class SimResult:
    strategy:      str
    agent_ids:     list[str]
    rounds_played: int
    round_logs:    list[dict]           # one entry per round
    agent_summaries: list[dict]         # one entry per agent
    total_utility: int
    elapsed_ms:    float

    def goals_met(self) -> int:
        """Count agents who claimed at least one task."""
        return sum(1 for s in self.agent_summaries if s["utility"] > 0)

    def total_nodes_explored(self) -> int:
        return sum(s["nodes_explored"] for s in self.agent_summaries)

    def total_nodes_pruned(self) -> int:
        return sum(s["nodes_pruned"] for s in self.agent_summaries)


# ──────────────────────────────────────────────
#  Simulation runner
# ──────────────────────────────────────────────

def run_simulation(env: Environment, agents: list[Agent],
                   verbose: bool = True) -> SimResult:
    """
    Run the full MARAS simulation.

    Parameters
    ----------
    env     : fresh Environment (not shared across runs — always pass a new one)
    agents  : list of Agent objects (strategy already set)
    verbose : print round-by-round trace if True
    """
    agent_ids = [a.agent_id for a in agents]
    strategy  = agents[0].strategy   # all agents share the same strategy in one run

    # Initialise adversarial engines with the full agent list
    for agent in agents:
        agent.init_engine(agent_ids)

    round_logs: list[dict] = []
    t0 = time.perf_counter()

    if verbose:
        strat_label = {
            "greedy":     "GREEDY",
            "minimax":    "PURE MINIMAX",
            "minimax_ab": "MINIMAX + ALPHA-BETA",
        }.get(strategy, strategy.upper())
        print(f"\n{'='*58}")
        print(f"  Strategy: {strat_label}")
        print(f"{'='*58}")
        print(f"  Initial state: {[str(t) for t in env.available_tasks()]}\n")

    for rnd in range(1, T_MAX + 1):
        if env.all_claimed():
            break

        # ── Phase 1: All agents decide simultaneously ──
        nominations: dict[str, str | None] = {}   # agent_id → task_id
        for agent in agents:
            nominations[agent.agent_id] = agent.decide(env, rnd)

        # ── Phase 2: Arbiter resolves conflicts ──
        # Priority order = list order; ties go to first agent
        claimed_this_round: dict[str, str] = {}   # task_id → agent_id
        log_entries: list[dict] = []

        for agent in agents:
            chosen = nominations[agent.agent_id]
            if chosen is None:
                log_entries.append({"agent": agent.agent_id,
                                    "wanted": None, "got": None})
                continue

            success = env.claim(chosen, agent.agent_id)
            actual  = chosen if success else None

            # If blocked, claim the best still-available task by preference score
            if not success:
                from planner import score_task
                remaining = env.available_tasks()
                if remaining:
                    best_fallback = max(remaining, key=lambda t: score_task(t, agent.agent_id))
                    if env.claim(best_fallback.task_id, agent.agent_id):
                        actual = best_fallback.task_id

            log_entries.append({"agent": agent.agent_id,
                                 "wanted": chosen, "got": actual})
            if actual:
                claimed_this_round[actual] = agent.agent_id

        round_logs.append({"round": rnd, "actions": log_entries})

        if verbose:
            print(f"  Round {rnd}:")
            for entry in log_entries:
                w = entry["wanted"] or "—"
                g = entry["got"]    or "✗ blocked"
                print(f"    {entry['agent']}  wanted={w:4s}  got={g}")
            remaining = [t.task_id for t in env.available_tasks()]
            print(f"    → Remaining: {remaining}\n")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    summaries  = [a.summary(env) for a in agents]
    total_util = sum(s["utility"] for s in summaries)

    if verbose:
        print(f"  {'─'*54}")
        print(f"  Final allocations:")
        for s in summaries:
            print(f"    {s['agent']}  tasks={s['tasks_claimed']}  "
                  f"utility={s['utility']}")
        print(f"  Total utility : {total_util}")
        print(f"  Time elapsed  : {elapsed_ms:.1f} ms")
        if strategy != "greedy":
            ne = sum(s["nodes_explored"] for s in summaries)
            np_ = sum(s["nodes_pruned"] for s in summaries)
            print(f"  Nodes explored: {ne}   Nodes pruned: {np_}")
        print()

    return SimResult(
        strategy=strategy,
        agent_ids=agent_ids,
        rounds_played=rnd,
        round_logs=round_logs,
        agent_summaries=summaries,
        total_utility=total_util,
        elapsed_ms=elapsed_ms,
    )
