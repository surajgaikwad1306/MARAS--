"""
agent.py
--------
Agent class: ties the planning layer and adversarial layer together.

Each agent executes a two-phase decision cycle per round (see report §4.3):

  Phase 1 — Plan Generation (STRIPS planner)
      Generate an ordered list of tasks the agent wants to claim.

  Phase 2 — Adversarial Evaluation (Minimax-AB)
      From the plan, choose the single best task to grab RIGHT NOW,
      taking into account what opponents will likely do.

Three agent strategies are supported (for comparison):
  - "greedy"       : claim the highest-value available task, no look-ahead
  - "minimax"      : Minimax without Alpha-Beta pruning
  - "minimax_ab"   : Minimax WITH Alpha-Beta pruning  (full hybrid system)
"""

from __future__ import annotations
from environment import Environment
from planner import STRIPSPlanner, score_task
from minimax import MinimaxEngine


AGENT_IDS  = ["A1", "A2", "A3"]
MINIMAX_DEPTH = 4   # look-ahead depth for adversarial search


# ──────────────────────────────────────────────
#  Agent
# ──────────────────────────────────────────────

class Agent:
    """
    A single agent in the MARAS simulation.

    Parameters
    ----------
    agent_id  : "A1" | "A2" | "A3"
    strategy  : "greedy" | "minimax" | "minimax_ab"
    """

    def __init__(self, agent_id: str, strategy: str = "minimax_ab"):
        self.agent_id = agent_id
        self.strategy = strategy

        # Planning layer
        self.planner = STRIPSPlanner(agent_id)

        # Adversarial layer (created lazily when all_agents is known)
        self._engine: MinimaxEngine | None = None

        # Logging
        self.total_nodes_explored = 0
        self.total_nodes_pruned   = 0
        self.decisions: list[dict] = []   # one dict per round

    def init_engine(self, all_agents: list[str]) -> None:
        """Call once before the simulation loop, passing the full agent list."""
        use_ab = (self.strategy == "minimax_ab")
        self._engine = MinimaxEngine(
            max_agent=self.agent_id,
            all_agents=all_agents,
            depth=MINIMAX_DEPTH,
            use_alpha_beta=use_ab,
        )

    # ── Decision cycle ───────────────────────

    def decide(self, env: Environment, round_num: int) -> str | None:
        """
        Execute one decision cycle. Returns the task_id to claim, or None.
        """
        # Phase 1 — Plan generation
        plan = self.planner.generate_plan(env)

        if not plan:
            self._log(round_num, plan, None, 0, 0)
            return None

        # Phase 2 — Choose action
        if self.strategy == "greedy":
            chosen = self._greedy_choice(env)
            nodes_exp, nodes_pruned = 0, 0

        elif self.strategy in ("minimax", "minimax_ab"):
            chosen, _ = self._engine.choose_task(env, plan)
            nodes_exp   = self._engine.nodes_explored
            nodes_pruned = self._engine.nodes_pruned
            self.total_nodes_explored += nodes_exp
            self.total_nodes_pruned   += nodes_pruned

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._log(round_num, plan, chosen, nodes_exp if self.strategy != "greedy" else 0,
                  nodes_pruned if self.strategy != "greedy" else 0)
        return chosen

    # ── Greedy fallback ──────────────────────

    def _greedy_choice(self, env: Environment) -> str | None:
        """
        Greedy: pick the available task with the highest score for this agent.
        No look-ahead — purely myopic.
        """
        available = env.available_tasks()
        if not available:
            return None
        best = max(available, key=lambda t: score_task(t, self.agent_id))
        return best.task_id

    # ── Logging ──────────────────────────────

    def _log(self, round_num: int, plan: list[str],
             chosen: str | None, nodes_exp: int, nodes_pruned: int) -> None:
        self.decisions.append({
            "round":         round_num,
            "plan":          plan,
            "chosen":        chosen,
            "nodes_explored": nodes_exp,
            "nodes_pruned":   nodes_pruned,
        })

    # ── Stats ────────────────────────────────

    def summary(self, env: Environment) -> dict:
        held  = env.tasks_held_by(self.agent_id)
        return {
            "agent":           self.agent_id,
            "strategy":        self.strategy,
            "tasks_claimed":   [t.task_id for t in held],
            "utility":         env.utility_of(self.agent_id),
            "nodes_explored":  self.total_nodes_explored,
            "nodes_pruned":    self.total_nodes_pruned,
        }

    def __repr__(self) -> str:
        return f"Agent({self.agent_id}, strategy={self.strategy})"
