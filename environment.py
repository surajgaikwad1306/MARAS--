"""
environment.py
--------------
Defines the task pool, global state, and resource arbiter for MARAS.

A "task" here is the resource to be allocated. Each task has:
  - an id          : unique string label  (e.g. "T1")
  - a value        : integer utility payoff for the agent that claims it
  - a type         : "cpu" | "memory" | "balanced"  (agent preference tag)
  - claimed_by     : None while available, agent-id once taken
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import copy


# ──────────────────────────────────────────────
#  Task
# ──────────────────────────────────────────────

@dataclass
class Task:
    task_id:    str
    value:      int
    task_type:  str                     # "cpu" | "memory" | "balanced"
    claimed_by: Optional[str] = None   # None = available

    def is_available(self) -> bool:
        return self.claimed_by is None

    def __repr__(self) -> str:
        status = "free" if self.is_available() else f"→{self.claimed_by}"
        return f"{self.task_id}(v={self.value},{self.task_type},{status})"

    def copy(self) -> "Task":
        return copy.copy(self)


# ──────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────

class Environment:
    """
    Holds all tasks and mediates allocation.

    State snapshot used by the Minimax engine is a lightweight dict:
        { task_id: claimed_by }   (None = still free)
    """

    def __init__(self, tasks: list[Task]):
        # keyed by task_id for O(1) lookup
        self.tasks: dict[str, Task] = {t.task_id: t for t in tasks}

    # ── Queries ──────────────────────────────

    def available_tasks(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.is_available()]

    def tasks_held_by(self, agent_id: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.claimed_by == agent_id]

    def utility_of(self, agent_id: str) -> int:
        return sum(t.value for t in self.tasks_held_by(agent_id))

    def total_utility(self) -> int:
        return sum(t.value for t in self.tasks.values() if not t.is_available())

    def all_claimed(self) -> bool:
        return all(not t.is_available() for t in self.tasks.values())

    # ── Mutation (real execution) ─────────────

    def claim(self, task_id: str, agent_id: str) -> bool:
        """
        Attempt to claim a task. Returns True on success, False if
        already taken (arbiter enforces first-come-first-served within
        a round; ties resolved by agent priority order in simulation.py).
        """
        task = self.tasks.get(task_id)
        if task is None or not task.is_available():
            return False
        task.claimed_by = agent_id
        return True

    # ── State snapshot / restore (for Minimax) ──

    def get_state(self) -> dict:
        """Return a lightweight snapshot: {task_id: claimed_by}."""
        return {tid: t.claimed_by for tid, t in self.tasks.items()}

    def set_state(self, state: dict) -> None:
        """Restore from snapshot."""
        for tid, owner in state.items():
            self.tasks[tid].claimed_by = owner

    def apply_action(self, task_id: str, agent_id: str) -> None:
        """Apply without the 'already-taken' guard — used inside Minimax."""
        self.tasks[task_id].claimed_by = agent_id

    # ── Pretty print ─────────────────────────

    def __repr__(self) -> str:
        lines = ["Environment:"]
        for t in self.tasks.values():
            lines.append(f"  {t}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
#  Default problem instance (from project spec)
# ──────────────────────────────────────────────

def build_default_environment() -> Environment:
    """
    Tasks from the project outline:
        T1(value=10,cpu)  T2(value=7,balanced)  T3(value=4,memory)
        T4(value=9,cpu)   T5(value=3,balanced)
    """
    tasks = [
        Task("T1", 10, "cpu"),
        Task("T2",  7, "balanced"),
        Task("T3",  4, "memory"),
        Task("T4",  9, "cpu"),
        Task("T5",  3, "balanced"),
    ]
    return Environment(tasks)
