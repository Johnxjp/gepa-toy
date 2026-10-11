"""The ticket classifier as a DSPy program, for `dspy.GEPA`."""

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

# isort: off
# openai must be imported before dspy: dspy lazy-loads openai, and LiteLLM importing it
# through that lazy module fails with a circular import on the first model call.
import openai  # noqa: F401
import dspy
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError
# isort: on

from gepa_toy.config import LABELS
from gepa_toy.evaluate import gepa_score, labels_of
from gepa_toy.feedback import written_feedback
from gepa_toy.model import Model
from gepa_toy.zero_shot import SEED_INSTRUCTION

# No JSON-mode retry when a reply doesn't parse: it would be an extra call the budget
# doesn't count, its strict schema would give invalid labels a second chance that our
# own GEPA loop doesn't get, and a provider rejecting the schema would stop the run.
dspy.configure(adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))


class StopOnLMError(BaseCallback):
    """Stops the run on the first model call that fails after DSPy's retries.

    Inside `compile`, DSPy catches every error and scores the example 0 (or skips the
    proposal), so the run carries on with fake zeros. Raising from here doesn't help:
    DSPy swallows exceptions from callbacks too. Hence `on_stop` exits the process.
    GEPA checkpoints at the start of each cycle, so the saved state never holds the
    failed cycle and `--resume` reruns it cleanly.

    A reply the model got wrong (bad format, cut off part-way) is not an LM error: it
    surfaces later as an `AdapterParseError` and is scored 0 as intended. An empty reply
    is an API failure: OpenRouter sends one with status 200 when the provider fails, and
    a reflection model that spends its whole token limit thinking returns one too.
    """

    def __init__(self, on_stop: Callable[[str], None]):
        self.on_stop = on_stop

    def on_lm_end(self, call_id, outputs, exception=None):
        if exception is not None:
            self.on_stop(f"{type(exception).__name__}: {exception}")
        elif not outputs:
            self.on_stop("model reply had no choices")
        elif not all(_text(output) for output in outputs):
            self.on_stop("model reply was empty")


def _text(output: str | dict | None) -> str | None:
    # Reasoning models give {"text": ..., "reasoning_content": ...}; others a plain string.
    return output.get("text") if isinstance(output, dict) else output


class SharedLM(dspy.LM):
    """An LM that every copy of a program shares.

    GEPA deep-copies the program for each candidate, which would copy the LM and its
    call history too, leaving the original's history empty and the run's task-model
    usage unrecorded.
    """

    def __deepcopy__(self, memo):
        return self


def exit_run(stop_file: Path) -> Callable[[str], None]:
    def stop(reason: str) -> None:
        stop_file.write_text(reason + "\n")
        print(f"Stopping run: {reason}", file=sys.stderr, flush=True)
        # os._exit, not sys.exit: this runs in a worker thread and must not let GEPA
        # finish the cycle or save state.
        os._exit(1)

    return stop


class ProgressLog:
    """One line per GEPA cycle with calls used and an ETA, to stdout and `log_file`.

    DSPy's own progress bar has an ETA too, but it is buried among the bars printed
    for every batch it scores. The ETA covers the rest of the budget plus `eval_calls`
    for final scoring, at the call rate seen since the first cycle of this process.
    """

    def __init__(
        self, budget: int, eval_calls: int, log_file: Path, clock=time.monotonic
    ):
        self.budget, self.eval_calls, self.log_file, self.clock = (
            budget,
            eval_calls,
            log_file,
            clock,
        )
        self.first = None

    def on_iteration_end(self, event) -> None:
        state, now = event["state"], self.clock()
        calls = state.total_num_evals
        if self.first is None:
            self.first = (now, calls)
        started, first_calls = self.first
        rate = (calls - first_calls) / (now - started) if now > started else 0
        remaining = max(self.budget - calls, 0) + self.eval_calls
        eta = f"{remaining / rate / 60:.0f} min" if rate else "?"
        line = (
            f"cycle {event['iteration']} | calls {calls}/{self.budget} | "
            f"{'accepted' if event['proposal_accepted'] else 'rejected'} | "
            f"candidates {len(state.program_candidates)} | "
            f"best pareto {max(state.program_full_scores_val_set):.3f} | eta {eta}"
        )
        print(line, flush=True)
        with self.log_file.open("a") as f:
            f.write(line + "\n")


class ClassifyTicket(dspy.Signature):
    subject: str = dspy.InputField()
    body: str = dspy.InputField()
    queue: Literal[tuple(LABELS["queue"])] = dspy.OutputField()
    type: Literal[tuple(LABELS["type"])] = dspy.OutputField()
    priority: Literal[tuple(LABELS["priority"])] = dspy.OutputField()
    tags: list[Literal[tuple(LABELS["tags"])]] = dspy.OutputField()


def build_program(lm: dspy.LM) -> dspy.Predict:
    program = dspy.Predict(ClassifyTicket.with_instructions(SEED_INSTRUCTION))
    # Set on the program rather than globally, so worker threads all use it.
    program.set_lm(lm)
    return program


def make_lm(model: Model, callbacks: list[BaseCallback]) -> SharedLM:
    return SharedLM(
        f"openrouter/{model.model}",
        temperature=model.temperature,
        max_tokens=model.max_tokens,
        timeout=model.timeout,
        extra_body={"reasoning": {"enabled": model.reasoning}},
        callbacks=callbacks,
    )


def to_example(row: dict[str, str]) -> dspy.Example:
    return dspy.Example(
        id=row["id"],
        # Matches our template, which shows an empty subject as "(no subject)".
        subject=row["subject"] or "(no subject)",
        body=row["body"],
        # Not `labels`: that name is a method on dspy.Example.
        true_labels=labels_of(row),
    ).with_inputs("subject", "body")


def prediction_dict(pred: dspy.Prediction) -> dict:
    return {
        "queue": pred.queue,
        "type": pred.type,
        "priority": pred.priority,
        "tags": list(pred.tags),
    }


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
    # DSPy scores unparseable replies itself (failure_score), so pred always has every field.
    predicted = prediction_dict(pred)
    return dspy.Prediction(
        score=gepa_score(predicted, gold.true_labels),
        feedback=written_feedback(predicted, gold.true_labels, None),
    )


def predict_row(program: dspy.Module, row: dict[str, str]) -> dict:
    """One prediction record in the same shape as `evaluate.predict`."""
    example = to_example(row)
    predicted, error, raw = None, None, None
    try:
        predicted = prediction_dict(program(**example.inputs()))
    except AdapterParseError as e:
        error, raw = str(e), e.lm_response
    return {
        "id": row["id"],
        "predicted": predicted,
        "labels": example.true_labels,
        "gepa_score": gepa_score(predicted, example.true_labels),
        "error": error,
        # DSPy keeps the reply text in its LM history, not on the prediction.
        "raw": raw,
    }


def candidate_log(results, pareto_ids: list[str]) -> dict:
    """Lineage from `detailed_results`, in the shape our own GEPA log uses.

    DSPy keys pareto scores by position in the valset; `pareto_ids` maps those back to tickets.
    """
    return {
        "pareto_ids": pareto_ids,
        "best": results.best_idx,
        "total_metric_calls": results.total_metric_calls,
        "num_full_val_evals": results.num_full_val_evals,
        "candidates": [
            {
                "id": i,
                "instruction": candidate.predictors()[0].signature.instructions,
                "parents": results.parents[i],
                "mean_pareto_score": results.val_aggregate_scores[i],
                "pareto_scores": results.val_subscores[i],
                "examples_won": sorted(
                    key
                    for key, winners in results.per_val_instance_best_candidates.items()
                    if i in winners
                ),
                "calls_when_found": results.discovery_eval_counts[i],
            }
            for i, candidate in enumerate(results.candidates)
        ],
    }
