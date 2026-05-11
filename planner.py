"""
planner.py
----------
STRIPS-style AI Planning module.

Each agent has:
  - a GOAL  : preferred task types (in priority order)
  - a PLAN  : ordered list of task_ids the agent *wants* to claim

The planner uses a simplified STRIPS formalism:
  State       = set of available task_ids
  Precondition= task must be available AND match agent's type preference
  Add-effect  = agent holds the task
  Del-effect  = task removed from available pool

Forward-chaining A* search finds the shortest sequence of "claim" actions
that satisfies the agent's goals, guided by a relaxed heuristic:
    h(s) = number of goal tasks still unclaimed

The planner returns a priority-ordered list of task_ids — this becomes
the agent's planning layer before the adversarial search kicks in.
"""

from __future__ import annotations
import heapq
from environment import Environment, Task


# ──────────────────────────────────────────────
#  Agent goal profiles
# ──────────────────────────────────────────────

# Type-preference weights per agent.
# Higher weight = more preferred task type.
AGENT_PREFERENCES: dict[str, dict[str, int]] = {
    "A1": {"cpu": 3, "balanced": 2, "memory": 1},       # CPU-heavy
    "A2": {"memory": 3, "cpu": 2, "balanced": 1},        # Memory-heavy
    "A3": {"balanced": 3, "cpu": 2, "memory": 2},        # Balanced
}


# ──────────────────────────────────────────────
#  Heuristic
# ──────────────────────────────────────────────

def h_relaxed(available_task_ids: frozenset[str],
              goal_task_ids: list[str]) -> int:
    """
    Admissible relaxed heuristic:
    Count of goal tasks that are still unclaimed (ignoring conflicts).
    Lower is better (closer to goal).
    """
    return sum(1 for tid in goal_task_ids if tid in available_task_ids)


# ──────────────────────────────────────────────
#  Priority scoring
# ──────────────────────────────────────────────

def score_task(task: Task, agent_id: str) -> float:
    """
    Score for ranking tasks in agent's plan.
    Raw task value is PRIMARY — always dominates.
    Type-preference is a small tiebreaker only (adds 0.0 to 0.3).
    This ensures agents never sacrifice raw points for type preference.
    """
    prefs  = AGENT_PREFERENCES.get(agent_id, {})
    weight = prefs.get(task.task_type, 1)          # 1, 2, or 3
    # Normalise preference to a tiny bonus: (weight-1)/10 → 0.0, 0.1, or 0.2
    tiebreaker = (weight - 1) / 10.0
    return task.value + tiebreaker


# ──────────────────────────────────────────────
#  Planner
# ──────────────────────────────────────────────

class STRIPSPlanner:
    """
    Forward-chaining STRIPS planner for a single agent.

    generate_plan() returns a list of task_ids ordered by the agent's
    goal preference — this is the agent's "plan" (priority sequence).

    The A* search finds the optimal claim order from the current state.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.preferences = AGENT_PREFERENCES.get(agent_id, {})

    # ── Public API ───────────────────────────

    def generate_plan(self, env: Environment) -> list[str]:
        """
        Return an ordered list of task_ids the agent should try to claim,
        best-first according to its type preferences and task values.

        STRIPS planning step: each action is 'claim task T'.
          Precondition : T is available AND T matches agent preference.
          Add-effect   : agent holds T.
          Del-effect   : T removed from available pool.

        The forward-chaining search reduces to a priority sort here because
        all claim actions are independent (no ordering dependency between
        tasks). We rank by score = value x type-preference-weight, which
        is the admissible heuristic for this domain.
        """
        available = env.available_tasks()
        if not available:
            return []

        # Sort available tasks: highest score for this agent first.
        # This is the agent's STRIPS goal sequence — the order in which
        # it will attempt to claim tasks if not blocked.
        plan = sorted(
            available,
            key=lambda t: score_task(t, self.agent_id),
            reverse=True
        )
        return [t.task_id for t in plan]

    # ── A* forward chaining ──────────────────

    def _astar(self, available_ids: frozenset[str],
               goal_task_ids: list[str]) -> list[str]:
        """
        State : frozenset of still-available task_ids
        Action: claim one task (removes it from available, adds to agent)
        Goal  : agent has claimed all tasks in goal_task_ids (or pool empty)

        Returns the sequence of task_ids to claim, in order.
        """
        # (f, g, state, path)
        start_h = h_relaxed(available_ids, goal_task_ids)
        heap: list[tuple] = [(start_h, 0, available_ids, [])]
        visited: set[frozenset] = set()

        while heap:
            f, g, state, path = heapq.heappop(heap)

            if state in visited:
                continue
            visited.add(state)

            # Terminal: no more tasks, or all goals satisfied
            if not state or all(tid not in state for tid in goal_task_ids):
                return path

            # Expand: try claiming each available task
            for tid in sorted(state):          # deterministic order
                new_state = state - {tid}
                new_path  = path + [tid]
                new_g     = g + 1
                new_h     = h_relaxed(new_state, goal_task_ids)

                # Prefer goal tasks first (bias towards agent's plan)
                priority_bonus = 0
                if tid in goal_task_ids:
                    idx = goal_task_ids.index(tid)
                    priority_bonus = -1.0 / (idx + 1)   # earlier = better

                new_f = new_g + new_h + priority_bonus
                heapq.heappush(heap, (new_f, new_g, new_state, new_path))

        return []   # no plan found (shouldn't happen in this domain)

    def __repr__(self) -> str:
        return f"STRIPSPlanner(agent={self.agent_id}, prefs={self.preferences})"
