"""GEPA core: candidates are strings, examples are opaque. See docs/gepa_design.md section 4."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class Rollout:
    score: float  # 0 to 1
    trace: str  # what the reflector sees: inputs and output, as text
    feedback: str  # one line on the outcome, in the task's chosen feedback mode
    record: dict  # task-specific detail for the log (id, prediction, raw reply)


class Task(Protocol):
    def evaluate(self, candidate: str, examples: list) -> list[Rollout]:
        """Run the candidate on each example. One rollout per example, same order."""


# Parent candidate and its rollouts on the minibatch -> child candidate.
Propose = Callable[[str, list[Rollout]], str]


@dataclass
class Settings:
    budget: int  # task-model calls, counted as examples evaluated
    minibatch_size: int = 5  # tickets are short, so 5 fit in the reflection prompt
    selection: Literal["pareto", "best"] = "pareto"
    seed: int = 0
