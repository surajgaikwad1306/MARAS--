"""
minimax.py
----------
Minimax with Alpha-Beta pruning for the adversarial search layer.

Multi-agent generalisation:
  - The current agent is MAX
  - All other agents are collectively MIN (they try to minimise MAX's utility)
  - Agents take turns in a fixed order within the game tree

The game tree node = (env_state_snapshot, depth, whose_turn)
Terminal condition = no tasks left OR depth limit reached

Evaluation function h(s):
    w1 * ResourceUtility  +  w2 * GoalProximity  -  w3 * OpponentAdvantage

    ResourceUtility  = sum of values held by MAX agent
    GoalProximity    = score of best available task for MAX (plan proximity)
    OpponentAdvantage= sum of values held by all other agents

Pruning stats (nodes explored, nodes pruned) are tracked for the
comparison report.
"""

from __future__ import annotations
from environment import Environment
from planner import score_task, AGENT_PREFERENCES

# Heuristic weights
# W1 (own utility) is dominant. W2 (future potential) is secondary.
# W3 (opponent penalty) is minimal — stops agents from over-sacrificing
# their own points just to block others.
W1 = 3.0   # ResourceUtility   ← own points, highest priority
W2 = 1.0   # GoalProximity     ← best remaining task value
W3 = 0.3   # OpponentAdvantage ← slight awareness of opponents


# ──────────────────────────────────────────────
#  Evaluation function
# ──────────────────────────────────────────────

def evaluate(env: Environment, max_agent: str, all_agents: list[str]) -> float:
    """Static heuristic h(s) from the MAX agent's perspective."""

    # w1 · ResourceUtility
    resource_utility = env.utility_of(max_agent)

    # w2 · GoalProximity  (best score MAX can still get from available tasks)
    available = env.available_tasks()
    if available:
        best_available = max(score_task(t, max_agent) for t in available)
    else:
        best_available = 0.0
    goal_proximity = best_available

    # w3 · OpponentAdvantage
    opponent_advantage = sum(
        env.utility_of(a) for a in all_agents if a != max_agent
    )

    return W1 * resource_utility + W2 * goal_proximity - W3 * opponent_advantage


# ──────────────────────────────────────────────
#  Minimax engine
# ──────────────────────────────────────────────

class MinimaxEngine:
    """
    Minimax with Alpha-Beta pruning.

    Usage:
        engine = MinimaxEngine(max_agent="A1", all_agents=["A1","A2","A3"],
                               depth=4, use_alpha_beta=True)
        best_task, score = engine.choose_task(env, candidate_tasks)
    """

    def __init__(self, max_agent: str, all_agents: list[str],
                 depth: int = 4, use_alpha_beta: bool = True):
        self.max_agent      = max_agent
        self.all_agents     = all_agents          # fixed turn order
        self.depth          = depth
        self.use_alpha_beta = use_alpha_beta

        # Stats reset each call to choose_task
        self.nodes_explored = 0
        self.nodes_pruned   = 0

    # ── Public API ───────────────────────────

    def choose_task(self, env: Environment,
                    candidate_tasks: list[str]) -> tuple[str | None, float]:
        """
        From the candidate task list (agent's plan), pick the best task
        to claim *right now* using Minimax look-ahead.

        Returns (best_task_id, minimax_value).
        """
        self.nodes_explored = 0
        self.nodes_pruned   = 0

        if not candidate_tasks:
            return None, 0.0

        best_task  = None
        best_value = float("-inf")

        # Save current env state before simulating
        root_state = env.get_state()

        for task_id in candidate_tasks:
            task = env.tasks.get(task_id)
            if task is None or not task.is_available():
                continue

            # Simulate MAX claiming this task
            env.apply_action(task_id, self.max_agent)

            # Determine next player (first opponent after MAX in turn order)
            next_player_idx = (self.all_agents.index(self.max_agent) + 1) % len(self.all_agents)
            next_player = self.all_agents[next_player_idx]

            value = self._minimax(
                env=env,
                depth=self.depth - 1,
                alpha=float("-inf"),
                beta=float("+inf"),
                current_agent=next_player,
                is_max=(next_player == self.max_agent),
            )

            # Restore
            env.set_state(root_state)

            if value > best_value:
                best_value = value
                best_task  = task_id

        return best_task, best_value

    # ── Recursive Minimax ────────────────────

    def _minimax(self, env: Environment, depth: int,
                 alpha: float, beta: float,
                 current_agent: str, is_max: bool) -> float:

        self.nodes_explored += 1

        available = env.available_tasks()

        # Terminal: no tasks left or depth exhausted
        if depth == 0 or not available:
            return evaluate(env, self.max_agent, self.all_agents)

        saved_state = env.get_state()

        if is_max:
            # MAX node — maximise over current_agent's moves
            value = float("-inf")
            for task in available:
                env.apply_action(task.task_id, current_agent)

                next_idx   = (self.all_agents.index(current_agent) + 1) % len(self.all_agents)
                next_agent = self.all_agents[next_idx]
                next_is_max = (next_agent == self.max_agent)

                child_val = self._minimax(env, depth - 1, alpha, beta,
                                          next_agent, next_is_max)
                env.set_state(saved_state)

                value = max(value, child_val)

                if self.use_alpha_beta:
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        self.nodes_pruned += (len(available) - available.index(task) - 1)
                        break   # Beta cut-off
        else:
            # MIN node — minimise to hurt MAX
            value = float("+inf")
            for task in available:
                env.apply_action(task.task_id, current_agent)

                next_idx   = (self.all_agents.index(current_agent) + 1) % len(self.all_agents)
                next_agent = self.all_agents[next_idx]
                next_is_max = (next_agent == self.max_agent)

                child_val = self._minimax(env, depth - 1, alpha, beta,
                                          next_agent, next_is_max)
                env.set_state(saved_state)

                value = min(value, child_val)

                if self.use_alpha_beta:
                    beta = min(beta, value)
                    if beta <= alpha:
                        self.nodes_pruned += (len(available) - available.index(task) - 1)
                        break   # Alpha cut-off

        return value
