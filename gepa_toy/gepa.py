"""GEPA core: candidates are strings, examples are opaque."""

import json
import math
import os
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from gepa_toy.utils import git_version


@dataclass
class Rollout:
    score: float  # 0 to 1
    trace: str  # what the reflector sees: inputs and output, as text
    feedback: str  # the outcome as text, in the task's chosen feedback mode
    record: dict  # task-specific detail for the log (id, prediction, raw reply)


class Task(Protocol):
    def evaluate(self, candidate: str, examples: list) -> list[Rollout]:
        """Run the candidate on each example. One rollout per example, same order."""

    def describe(self) -> dict:
        """Model configs and feedback mode, recorded in the log's metadata."""


# Parent candidate and its rollouts on the minibatch -> child candidate.
Propose = Callable[[str, list[Rollout]], str]


@dataclass
class Settings:
    budget: int  # task-model calls, counted as examples evaluated
    minibatch_size: int = 5  # tickets are short, so 5 fit in the reflection prompt
    selection: Literal["pareto", "best"] = "pareto"
    seed: int = 0


def optimise(
    seed_candidate: str,
    train: list,
    pareto: list,
    task: Task,
    propose: Propose,
    settings: Settings,
    log_path: Path,
    metadata: dict | None = None,
) -> dict:
    # An empty train would loop forever in shuffled_positions, an empty pareto has no mean,
    # and a budget below len(pareto) cannot even score the seed.
    if (
        not train
        or not pareto
        or settings.minibatch_size < 1
        or settings.budget < len(pareto)
    ):
        raise ValueError(
            "need non-empty train and pareto, minibatch_size >= 1, budget >= len(pareto)"
        )

    rng = random.Random(settings.seed)
    positions = shuffled_positions(len(train), rng)
    started = time.time()
    calls_used = 0
    candidates: list[dict] = []
    log = {
        "metadata": {
            "git": git_version(),
            "settings": asdict(settings),
            "task": task.describe(),
            "train_size": len(train),
            "pareto_size": len(pareto),
            "started": datetime.now(UTC).isoformat(),
            "runtime_seconds": 0.0,
            # Caller's metadata and seed go in from the first write, so a crashed run is still labelled.
            **(metadata or {}),
        },
        "candidates": candidates,
        "iterations": [],
        "best": None,
    }

    def evaluate(candidate: str, examples: list) -> list[Rollout]:
        nonlocal calls_used
        calls_used += len(examples)
        rollouts = task.evaluate(candidate, examples)
        # A short or empty list would read as a perfect parent (all([]) is True) and a NaN
        # score would pass the acceptance test, so both become plausible-looking log rows.
        if len(rollouts) != len(examples) or any(
            not 0 <= r.score <= 1 for r in rollouts
        ):
            raise ValueError(
                "task must return one rollout per example with scores in 0..1"
            )
        return rollouts

    def budget_left() -> int:
        return settings.budget - calls_used

    def write_log() -> None:
        log["metadata"]["runtime_seconds"] = time.time() - started
        log["best"] = best_candidate(candidates)["id"]
        # Write-then-rename: a crash mid-write would otherwise leave a truncated log.
        tmp = log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(log, indent=2))
        os.replace(tmp, log_path)

    add_candidate(
        candidates, seed_candidate, None, None, evaluate(seed_candidate, pareto)
    )
    write_log()

    while budget_left() >= 2 * settings.minibatch_size:
        parent = select_parent(candidates, settings, rng)
        batch = next_minibatch(positions, settings.minibatch_size)
        examples = [train[i] for i in batch]
        parent_rollouts = evaluate(parent["text"], examples)
        iteration = {
            "parent": parent["id"],
            "minibatch": batch,
            "parent_scores": scores(parent_rollouts),
            "parent_rollouts": [asdict(r) for r in parent_rollouts],
            "child": None,
            "child_text": None,
            "child_scores": None,
            "child_rollouts": None,
            "reflection_prompt": None,
            "reflection_reply": None,
            "outcome": None,
            "error": None,
            "calls_used": None,
        }
        log["iterations"].append(iteration)

        if all(s == 1 for s in iteration["parent_scores"]):
            iteration["outcome"] = "skipped"
        else:
            try:
                child_text = propose(parent["text"], parent_rollouts)
            except Exception as e:  # noqa: BLE001 - any reflector failure is logged and the loop goes on
                iteration["outcome"] = "proposal failed"
                iteration["error"] = f"{type(e).__name__}: {e}"
            # make_reflector records these on itself, even when it raises. Fakes may not.
            iteration["reflection_prompt"] = getattr(propose, "last_prompt", None)
            iteration["reflection_reply"] = getattr(propose, "last_reply", None)
            if iteration["outcome"] is None:
                child_rollouts = evaluate(child_text, examples)
                iteration["child_text"] = child_text
                iteration["child_scores"] = scores(child_rollouts)
                iteration["child_rollouts"] = [asdict(r) for r in child_rollouts]
                child_sum = sum(iteration["child_scores"])
                parent_sum = sum(iteration["parent_scores"])
                # isclose: sums of real gepa_scores such as 0.6 and 3 * 0.2 differ only by rounding.
                if child_sum <= parent_sum or math.isclose(child_sum, parent_sum):
                    iteration["outcome"] = "rejected"
                elif budget_left() < len(pareto):
                    iteration["outcome"] = "out of budget"
                else:
                    created = len(log["iterations"]) - 1
                    child_rollouts = evaluate(child_text, pareto)
                    child = add_candidate(
                        candidates, child_text, parent["id"], created, child_rollouts
                    )
                    iteration["child"] = child["id"]
                    iteration["outcome"] = "accepted"

        iteration["calls_used"] = calls_used
        write_log()
        if iteration["outcome"] == "out of budget":
            break

    return log


def select_parent(
    candidates: list[dict], settings: Settings, rng: random.Random
) -> dict:
    if settings.selection == "best":
        return best_candidate(candidates)
    front = pareto_front(candidates)
    return rng.choices(front, weights=[len(c["pareto_wins"]) for c in front])[0]


def pareto_front(candidates: list[dict]) -> list[dict]:
    """Candidates that win at least one pareto example, minus the dominated ones.

    A candidate is dominated when every example it wins is also won by some other
    candidate still in the running, not necessarily the same one. This is the reference
    implementation's `remove_dominated_programs` (gepa-ai/gepa, gepa_utils.py), which
    DSPy's GEPA runs. Checked from the lowest mean upwards, so of two with the same win set
    the weaker goes (the earlier one on equal means). The reference rescans after each
    drop; dropping only shrinks the cover for later checks, so one pass gives the same
    result. A candidate that wins an example nobody else wins can never be dropped.
    """
    front = [c for c in candidates if c["pareto_wins"]]
    for c in sorted(front, key=lambda c: c["mean_pareto_score"]):
        covered = {j for o in front if o is not c for j in o["pareto_wins"]}
        if set(c["pareto_wins"]) <= covered:
            front.remove(c)
    return front


def best_candidate(candidates: list[dict]) -> dict:
    # max keeps the first of equal keys, so ties go to the earliest candidate.
    return max(candidates, key=lambda c: c["mean_pareto_score"])


def shuffled_positions(n: int, rng: random.Random) -> Iterator[int]:
    """Every position once per epoch, reshuffled each epoch.

    A minibatch that straddles two epochs can hold the same position twice; accepted,
    since it is rare for a train split much larger than the minibatch.
    """
    while True:
        order = list(range(n))
        rng.shuffle(order)
        yield from order


def next_minibatch(positions: Iterator[int], size: int) -> list[int]:
    return [next(positions) for _ in range(size)]


def scores(rollouts: list[Rollout]) -> list[float]:
    return [r.score for r in rollouts]


def add_candidate(
    candidates: list[dict],
    text: str,
    parent: int | None,
    iteration: int | None,
    pareto_rollouts: list[Rollout],
) -> dict:
    pareto_scores = scores(pareto_rollouts)
    candidate = {
        "id": len(candidates),
        "text": text,
        "parent": parent,
        "iteration": iteration,
        "pareto_scores": pareto_scores,
        "mean_pareto_score": sum(pareto_scores) / len(pareto_scores),
        "pareto_wins": [],
    }
    candidates.append(candidate)
    update_wins(candidates)
    return candidate


def update_wins(candidates: list[dict]) -> None:
    """A new candidate can take the top score on an example, so every win list is recomputed."""
    for c in candidates:
        c["pareto_wins"] = []
    for j in range(len(candidates[0]["pareto_scores"])):
        top = max(c["pareto_scores"][j] for c in candidates)
        for c in candidates:
            # isclose: gepa_score gives 0.7 or 0.7000000000000001 for the same result,
            # depending on the tag counts, and ties must all win.
            if math.isclose(c["pareto_scores"][j], top):
                c["pareto_wins"].append(j)
