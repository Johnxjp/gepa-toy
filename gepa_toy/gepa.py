"""GEPA core: candidates are strings, examples are opaque."""

import json
import math
import os
import random
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from gepa_toy.model import Model
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


@dataclass
class Candidate:
    id: int
    text: str
    parent: int | None  # None for the seed
    iteration: int | None  # iteration that created it; None for the seed
    pareto_scores: list[float]  # one per pareto example, same order as the pareto list
    # A field, not a property, so asdict writes it to the log.
    mean_pareto_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean_pareto_score = sum(self.pareto_scores) / len(self.pareto_scores)


class CandidatePool:
    """Accepted candidates and their pareto scores. No model calls, budget or files."""

    def __init__(self) -> None:
        self.candidates: list[Candidate] = []

    def add(
        self,
        text: str,
        parent: int | None,
        iteration: int | None,
        pareto_scores: list[float],
    ) -> Candidate:
        """Append a candidate with the next free id.

        Args:
            text: The candidate instruction.
            parent: Id of the candidate it was proposed from, None for the seed.
            iteration: Index of the iteration that created it, None for the seed.
            pareto_scores: Its score on each pareto example, in pareto-list order.

        Returns:
            The new candidate, with the next free id.
        """
        candidate = Candidate(
            id=len(self.candidates),
            text=text,
            parent=parent,
            iteration=iteration,
            pareto_scores=pareto_scores,
        )
        self.candidates.append(candidate)
        return candidate

    def best(self) -> Candidate:
        """The candidate with the highest mean pareto score."""
        # max keeps the first of equal keys, so ties go to the earliest candidate.
        return max(self.candidates, key=lambda c: c.mean_pareto_score)

    def pareto_front(self) -> list[Candidate]:
        """Candidates that win at least one pareto example, minus the dominated ones.

        A candidate is dominated when every example it wins is also won by some other
        candidate still in the running, not necessarily the same one. This is the reference
        implementation's `remove_dominated_programs` (gepa-ai/gepa, gepa_utils.py), which
        DSPy's GEPA runs. Checked from the lowest mean upwards, so of two with the same win set
        the weaker goes (the earlier one on equal means). The reference rescans after each
        drop; dropping only shrinks the cover for later checks, so one pass gives the same
        result. A candidate that wins an example nobody else wins can never be dropped.
        """
        wins = self.wins()
        front = [c for c in self.candidates if wins[c.id]]
        for c in sorted(front, key=lambda c: c.mean_pareto_score):
            covered = {j for o in front if o is not c for j in wins[o.id]}
            if wins[c.id] <= covered:
                front.remove(c)
        return front

    def wins(self) -> dict[int, set[int]]:
        """The pareto examples on which each candidate has the top score, keyed by id.

        Recomputed on every call: a new candidate can take the top score on an example
        from every candidate before it. Ties all win.
        """
        wins = {c.id: set() for c in self.candidates}
        for j in range(len(self.candidates[0].pareto_scores)):
            top = max(c.pareto_scores[j] for c in self.candidates)
            for c in self.candidates:
                # isclose: gepa_score gives 0.7 or 0.7000000000000001 for the same result,
                # depending on the tag counts, and ties must all win.
                if math.isclose(c.pareto_scores[j], top):
                    wins[c.id].add(j)
        return wins

    def select_parent(
        self, selection: Literal["pareto", "best"], rng: random.Random
    ) -> Candidate:
        """Pick the candidate to improve next.

        Args:
            selection: "best" takes the top mean; "pareto" samples the front, weighted by
                the number of examples each candidate wins.
            rng: Drawn from in "pareto" mode even when the front has one candidate, so the
                draw order is the same for every run with a given seed.
        """
        if selection == "best":
            return self.best()
        front = self.pareto_front()
        wins = self.wins()
        return rng.choices(front, weights=[len(wins[c.id]) for c in front])[0]


class GEPA:
    def __init__(
        self,
        task: Task,
        reflection_model: Model,
        reflection_prompt: str,
        log_path: Path,
        *,
        budget: int,
        minibatch_size: int = 5,
        selection: Literal["pareto", "best"] = "pareto",
        seed: int = 0,
    ):
        """Store the settings; nothing runs until `run`.

        Args:
            task: Evaluates a candidate on examples.
            reflection_model: The model that proposes new instructions. Never cached.
            reflection_prompt: Template with `{instruction}` and `{rollouts}` placeholders.
            log_path: Where every `run` writes its log, overwriting the last run's file.
            budget: Task-model calls, counted as examples evaluated.
            minibatch_size: Train examples the parent is evaluated on each iteration.
            selection: How the parent of each iteration is chosen.
            seed: Seeds parent sampling and the minibatch order.
        """
        self.task = task
        self.reflection_model = reflection_model
        self.reflection_prompt = reflection_prompt
        self.log_path = log_path
        self.budget = budget
        self.minibatch_size = minibatch_size
        self.selection = selection
        self.seed = seed
        self.calls_used = 0

    def run(self, seed_candidate: str, train: list, pareto: list) -> Candidate:
        """Run GEPA from scratch and return the best candidate.

        Builds all run state fresh, so calling `run` again starts a new run. The finished
        run stays in `self.log` (the dict written to `log_path`) and `self.pool`.

        Args:
            seed_candidate: The instruction the search starts from.
            train: Examples that minibatches are drawn from.
            pareto: Examples every accepted candidate is scored on.

        Returns:
            The candidate with the highest mean pareto score, earliest on ties.

        Raises:
            ValueError: If the inputs or settings cannot support a run.
        """
        # An empty train would loop forever in shuffled_positions, an empty pareto has no mean,
        # and a budget below len(pareto) cannot even score the seed.
        if (
            not train
            or not pareto
            or self.minibatch_size < 1
            or self.budget < len(pareto)
        ):
            raise ValueError(
                "need non-empty train and pareto, minibatch_size >= 1, budget >= len(pareto)"
            )

        rng = random.Random(self.seed)
        positions = shuffled_positions(len(train), rng)
        self.calls_used = 0
        self.pool = CandidatePool()
        self._started = time.time()
        self.log = {
            "metadata": {
                "git": git_version(),
                "settings": {
                    "budget": self.budget,
                    "minibatch_size": self.minibatch_size,
                    "selection": self.selection,
                    "seed": self.seed,
                },
                "task": self.task.describe(),
                "train_size": len(train),
                "pareto_size": len(pareto),
                "started": datetime.now(UTC).isoformat(),
                "runtime_seconds": 0.0,
                "reflection_model": self.reflection_model.config,
                "reflection_prompt": self.reflection_prompt,
            },
            "candidates": [],
            "iterations": [],
            "best": None,
        }

        seed_rollouts = self._evaluate(seed_candidate, pareto)
        self.pool.add(seed_candidate, None, None, [r.score for r in seed_rollouts])
        self._write_log()

        while self._budget_left() >= 2 * self.minibatch_size:
            # Parent before minibatch: both draw from rng, so swapping them changes every
            # minibatch after the first selection.
            parent = self.pool.select_parent(self.selection, rng)
            batch_ids = [next(positions) for _ in range(self.minibatch_size)]
            iteration = self._step(parent, batch_ids, train, pareto)
            self._write_log()
            if iteration["outcome"] == "out of budget":
                break

        return self.pool.best()

    def _step(
        self, parent: Candidate, batch_ids: list[int], train: list, pareto: list
    ) -> dict:
        """Run one iteration: reflect on the parent's minibatch, maybe add a child.

        Args:
            parent: The candidate to improve.
            batch_ids: Positions in `train` of this iteration's minibatch; logged as `minibatch`.
            train: The train examples `batch_ids` index into.
            pareto: Examples a new child is scored on.

        Returns:
            The iteration record, already appended to `self.log["iterations"]`.
        """
        examples = [train[i] for i in batch_ids]
        parent_rollouts = self._evaluate(parent.text, examples)
        iteration = {
            "parent": parent.id,
            "minibatch": batch_ids,
            "parent_scores": [r.score for r in parent_rollouts],
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
        self.log["iterations"].append(iteration)

        if all(s == 1 for s in iteration["parent_scores"]):
            iteration["outcome"] = "skipped"
        else:
            try:
                child_text = self._propose(parent, parent_rollouts, iteration)
            except Exception as e:  # noqa: BLE001 - any reflector failure is logged and the loop goes on
                iteration["outcome"] = "proposal failed"
                iteration["error"] = f"{type(e).__name__}: {e}"
            if iteration["outcome"] is None:
                child_rollouts = self._evaluate(child_text, examples)
                iteration["child_text"] = child_text
                iteration["child_scores"] = [r.score for r in child_rollouts]
                iteration["child_rollouts"] = [asdict(r) for r in child_rollouts]
                child_sum = sum(iteration["child_scores"])
                parent_sum = sum(iteration["parent_scores"])
                # isclose: sums of real gepa_scores such as 0.6 and 3 * 0.2 differ only by rounding.
                if child_sum <= parent_sum or math.isclose(child_sum, parent_sum):
                    iteration["outcome"] = "rejected"
                elif self._budget_left() < len(pareto):
                    iteration["outcome"] = "out of budget"
                else:
                    created = len(self.log["iterations"]) - 1
                    child_rollouts = self._evaluate(child_text, pareto)
                    child = self.pool.add(
                        child_text,
                        parent.id,
                        created,
                        [r.score for r in child_rollouts],
                    )
                    iteration["child"] = child.id
                    iteration["outcome"] = "accepted"

        iteration["calls_used"] = self.calls_used
        return iteration

    def _propose(
        self, parent: Candidate, rollouts: list[Rollout], iteration: dict
    ) -> str:
        """Ask the reflection model for an improved instruction.

        The prompt and reply go into `iteration` as soon as each exists, so a failed
        proposal still logs whatever it sent and got back.

        Args:
            parent: The candidate being improved.
            rollouts: The parent's rollouts on the minibatch.
            iteration: The iteration record to write `reflection_prompt` and `reflection_reply` into.

        Returns:
            The proposed instruction.
        """
        prompt = build_reflection_prompt(self.reflection_prompt, parent.text, rollouts)
        iteration["reflection_prompt"] = prompt
        reply = self.reflection_model.complete(prompt)
        iteration["reflection_reply"] = reply
        return extract_instruction(reply)

    def _evaluate(self, candidate: str, examples: list) -> list[Rollout]:
        """Run the task on the examples, count the calls and check the rollouts."""
        self.calls_used += len(examples)
        rollouts = self.task.evaluate(candidate, examples)
        # A short or empty list would read as a perfect parent (all([]) is True) and a NaN
        # score would pass the acceptance test, so both become plausible-looking log rows.
        if len(rollouts) != len(examples) or any(
            not 0 <= r.score <= 1 for r in rollouts
        ):
            raise ValueError(
                "task must return one rollout per example with scores in 0..1"
            )
        return rollouts

    def _budget_left(self) -> int:
        return self.budget - self.calls_used

    def _write_log(self) -> None:
        """Refresh the log's candidates, best and runtime, and write it to `log_path`."""
        self.log["metadata"]["runtime_seconds"] = time.time() - self._started
        self.log["candidates"] = [asdict(c) for c in self.pool.candidates]
        self.log["best"] = self.pool.best().id
        # Write-then-rename: a crash mid-write would otherwise leave a truncated log.
        tmp = self.log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.log, indent=2))
        os.replace(tmp, self.log_path)


# Both fences must start a line: a ``` mentioned mid-sentence before the real
# block would otherwise pair with its opening fence and the block would be lost.
FENCED_BLOCK = re.compile(
    r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL
)


def build_reflection_prompt(
    template: str, instruction: str, rollouts: list[Rollout]
) -> str:
    """Fill the template with the instruction and the numbered rollouts."""
    examples = "\n\n".join(
        f"Example {i}\n{r.trace}\n{r.feedback}" for i, r in enumerate(rollouts, 1)
    )
    values = {"instruction": instruction, "rollouts": examples}
    # One pass over the template, so a placeholder inside the instruction or a
    # trace is never substituted (sequential str.replace would do that).
    return re.sub(r"\{(instruction|rollouts)\}", lambda m: values[m[1]], template)


def extract_instruction(reply: str) -> str:
    """Return the last fenced block in the reply, without its final newline.

    Raises:
        ValueError: If the reply has no fenced block, or the last one is blank.
    """
    blocks = FENCED_BLOCK.findall(reply)
    if not blocks:
        raise ValueError(f"No fenced code block in reflection reply: {reply[:80]!r}")
    instruction = blocks[-1].removesuffix("\n")
    if not instruction.strip():
        raise ValueError(f"Empty fenced code block in reflection reply: {reply[:80]!r}")
    return instruction


def shuffled_positions(n: int, rng: random.Random) -> Iterator[int]:
    """Every position once per epoch, reshuffled each epoch.

    A minibatch that straddles two epochs can hold the same position twice; accepted,
    since it is rare for a train split much larger than the minibatch.
    """
    while True:
        order = list(range(n))
        rng.shuffle(order)
        yield from order
