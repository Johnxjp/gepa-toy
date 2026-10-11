"""Stage 4: the ticket classifier as a DSPy program for `dspy.GEPA`.

See docs/gepa_design.md section 4.3 ("DSPy") and decision D2.
"""

import os
import sys
from collections.abc import Callable
from typing import Literal

# isort: off
# openai must be imported before dspy: dspy lazy-loads openai, and LiteLLM importing it
# through that lazy module fails with a circular import on the first model call.
import openai  # noqa: F401
import dspy
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError, LMError, is_retryable_lm_error
# isort: on

from gepa_toy.config import LABELS
from gepa_toy.evaluate import gepa_score, labels_of
from gepa_toy.model import ModelConfig
from gepa_toy.ticket_task import feedback
from gepa_toy.zero_shot import SEED_INSTRUCTION


class StopOnLMError(BaseCallback):
    """Stops the run on the first model call that fails after DSPy's retries.

    Inside `compile`, DSPy catches every error from the program, scores the example
    0 and carries on, so a bad key or an exhausted quota burns the whole budget and
    returns the seed as "best" with nothing in the log to show for it. Raising from a
    callback doesn't help: DSPy swallows those too. Hence `on_stop`, which in the
    script exits the process. A reply the model got wrong is not an LM error: it
    surfaces as an `AdapterParseError` and is scored 0 as intended.

    With `stop_on_transient=False`, errors DSPy classes as retryable (timeout, 5xx,
    429) pass through, so a failed proposal is skipped as in our loop (design 4.5).
    """

    def __init__(self, on_stop: Callable[[str], None], stop_on_transient: bool = True):
        self.on_stop = on_stop
        self.stop_on_transient = stop_on_transient

    def on_lm_end(self, call_id, outputs, exception=None):
        if exception is None:
            return
        if not self.stop_on_transient and is_retryable_lm_error(exception):
            return
        self.on_stop(f"{type(exception).__name__}: {exception}")


def exit_run(reason: str) -> None:
    print(f"Stopping run: {reason}", file=sys.stderr, flush=True)
    # os._exit, not sys.exit: this runs in a worker thread, and the main thread
    # would otherwise finish the iteration with zeros and save it as a result.
    os._exit(1)


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


def make_lm(
    model_config: ModelConfig, callbacks: list[BaseCallback] | None = None
) -> dspy.LM:
    return dspy.LM(
        f"openrouter/{model_config.model}",
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout=model_config.timeout,
        extra_body={"reasoning": {"enabled": model_config.reasoning}},
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
        feedback=feedback(predicted, gold.true_labels, None, "written"),
    )


def predict_row(program: dspy.Module, row: dict[str, str]) -> dict:
    """One prediction record in the same shape as `evaluate.predict`."""
    example = to_example(row)
    predicted, error, raw = None, None, None
    try:
        predicted = prediction_dict(program(**example.inputs()))
    except AdapterParseError as e:
        error, raw = str(e), e.lm_response
    except LMError as e:
        # Same policy as evaluate.predict: still failing after DSPy's retries is
        # recorded, a setup error (bad key, unknown model) must crash.
        if not is_retryable_lm_error(e):
            raise
        error = f"API error: {e}"
    return {
        "id": row["id"],
        "predicted": predicted,
        "labels": example.true_labels,
        "gepa_score": gepa_score(predicted, example.true_labels),
        "error": error,
        # DSPy keeps the reply text in its LM history, not on the prediction.
        "raw": raw,
    }


def candidate_log(results) -> dict:
    """Lineage from `detailed_results`, in the shape our own GEPA log uses.

    DSPy keys pareto scores and wins by position in the valset, as we do.
    """
    return {
        "candidates": [
            {
                "id": i,
                "text": candidate.predictors()[0].signature.instructions,
                # One parent per candidate with merge off (D3); DSPy stores a list.
                "parent": results.parents[i][0],
                "pareto_scores": [scores[j] for j in sorted(scores)],
                "mean_pareto_score": results.val_aggregate_scores[i],
                "pareto_wins": sorted(
                    j
                    for j, winners in results.per_val_instance_best_candidates.items()
                    if i in winners
                ),
                "calls_when_found": results.discovery_eval_counts[i],
            }
            for i, (candidate, scores) in enumerate(
                zip(results.candidates, results.val_subscores, strict=True)
            )
        ],
        "best": results.best_idx,
        "calls_used": results.total_metric_calls,
    }
