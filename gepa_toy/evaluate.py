import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import openai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from gepa_toy.config import PROJECT_ROOT
from gepa_toy.data_processing.tsv import read_tsv
from gepa_toy.model import ModelConfig, Result, classify
from gepa_toy.utils import git_version

SINGLE_FIELDS = ("queue", "type", "priority")
# Stands in for a failed prediction so it counts as wrong for every class.
NO_PREDICTION = "(none)"
# Still failing after the SDK's own retries: recorded so one flaky ticket doesn't sink a run.
# Anything else (bad key 401, unknown model 404, bad request 400) is a setup bug and must
# crash, or the run is saved with every ticket scored 0 and looks like a real result.
TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def labels_of(row: dict[str, str]) -> dict:
    """True labels for a split row, in the same shape as a `Prediction`.

    Tags come from `tag_groups`, the grouped tags the model chooses from, not the
    raw `tags` column. The TSV stores them comma-separated.
    """
    return {
        **{field: row[field] for field in SINGLE_FIELDS},
        "tags": row["tag_groups"].split(","),
    }


def tag_scores(predicted: list[str], labels: list[str]) -> dict:
    hits = len(set(predicted) & set(labels))
    precision = hits / len(set(predicted)) if predicted else 0.0
    recall = hits / len(set(labels)) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def gepa_score(predicted: dict | None, labels: dict) -> float:
    """Score one ticket from 0 to 1 for GEPA to optimise.

    GEPA compares candidate prompts ticket by ticket: these scores build the
    Pareto frontier, and their sum decides whether a child beats its parent
    on a minibatch.

    The score is the mean of four parts: exact match on queue, type and
    priority, plus F1 between the predicted and true tag sets. A failed
    prediction scores 0.
    """
    if predicted is None:
        return 0.0
    parts = [predicted[field] == labels[field] for field in SINGLE_FIELDS]
    parts.append(tag_scores(predicted["tags"], labels["tags"])["f1"])
    return sum(parts) / len(parts)


def predict(
    row: dict[str, str], build_prompt: Callable, model_config: ModelConfig
) -> dict:
    prompt = build_prompt(row["subject"], row["body"])
    try:
        result = classify(prompt, model_config)
    except TRANSIENT_ERRORS as e:
        result = Result(prediction=None, raw="", error=f"API error: {e}")
    predicted = result.prediction.model_dump() if result.prediction else None
    labels = labels_of(row)
    return {
        "id": row["id"],
        "predicted": predicted,
        "labels": labels,
        "gepa_score": gepa_score(predicted, labels),
        "error": result.error,
        "raw": result.raw,
    }


def field_scores(predictions: list[dict], field: str) -> dict:
    y_true = [p["labels"][field] for p in predictions]
    y_pred = [
        p["predicted"][field] if p["predicted"] else NO_PREDICTION for p in predictions
    ]
    # Macro over the classes present in the labels, so classes absent from a small split don't score 0.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def summarise(predictions: list[dict]) -> dict:
    """Aggregate scores for a run, for reporting and comparing prompts.

    Queue, type and priority get accuracy plus macro precision, recall and F1.
    Tags get set precision, recall and F1, averaged per ticket. `gepa_score` is
    the mean of the per-ticket scores, which GEPA uses to pick the final prompt.
    Failed predictions count as wrong on every field.
    """
    n = len(predictions)
    tags = [
        tag_scores(
            p["predicted"]["tags"] if p["predicted"] else [], p["labels"]["tags"]
        )
        for p in predictions
    ]
    return {
        "n": n,
        "errors": sum(p["error"] is not None for p in predictions),
        "gepa_score": sum(p["gepa_score"] for p in predictions) / n,
        **{field: field_scores(predictions, field) for field in SINGLE_FIELDS},
        "tags": {
            key: sum(t[key] for t in tags) / n for key in ("f1", "precision", "recall")
        },
    }


def evaluate(
    data_file: Path,
    build_prompt: Callable,
    model_config: ModelConfig,
    workers: int = 8,
) -> dict:
    # Gathered before any model calls, so a failure here doesn't waste a paid run.
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "git": git_version(),
        "data_file": str(data_file.relative_to(PROJECT_ROOT)),
        "model": model_config.model_dump(),
        "prompt": build_prompt("{subject}", "{body}"),
    }
    start = time.monotonic()
    rows = read_tsv(data_file)
    with ThreadPoolExecutor(workers) as pool:
        predictions = list(
            pool.map(lambda r: predict(r, build_prompt, model_config), rows)
        )
    metadata["runtime_seconds"] = round(time.monotonic() - start, 2)
    return {
        "metadata": metadata,
        "predictions": predictions,
        "results": summarise(predictions),
    }


def save_run(results_dir: Path, run: dict) -> Path:
    started_at = datetime.fromisoformat(run["metadata"]["started_at"])
    split = Path(run["metadata"]["data_file"]).stem
    # Microseconds so parallel runs (3 per stage) don't overwrite each other.
    path = results_dir / f"{started_at:%Y%m%d-%H%M%S-%f}_{split}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2) + "\n")
    return path
