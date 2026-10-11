import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from gepa_toy import model
from gepa_toy.config import LABELS, PROJECT_ROOT
from gepa_toy.data_processing.tsv import read_tsv
from gepa_toy.model import ModelConfig, parse_prediction
from gepa_toy.utils import git_version

SINGLE_FIELDS = ("queue", "type", "priority")
# Stands in for a failed prediction so it counts as wrong for every class.
NO_PREDICTION = "(none)"


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
    row: dict[str, str],
    build_prompt: Callable,
    model_config: ModelConfig,
    complete: Callable[[str, ModelConfig], str] | None = None,
) -> dict:
    # API errors propagate and stop the run. Only a
    # reply the model got wrong is recorded, as `error`.
    prompt = build_prompt(row["subject"], row["body"])
    result = parse_prediction((complete or model.complete)(prompt, model_config))
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


def evaluate_rows(
    rows: list[dict],
    build: Callable[[str, str], str],
    model_config: ModelConfig,
    workers: int,
    complete: Callable[[str, ModelConfig], str] | None = None,
) -> list[dict]:
    return map_until_error(
        lambda r: predict(r, build, model_config, complete), rows, workers
    )


def queue_confusion(predictions: list[dict]) -> dict:
    rows = LABELS["queue"]
    columns = rows + [NO_PREDICTION]
    counts = [[0] * len(columns) for _ in rows]
    for p in predictions:
        predicted = p["predicted"]["queue"] if p["predicted"] else NO_PREDICTION
        counts[rows.index(p["labels"]["queue"])][columns.index(predicted)] += 1
    return {"rows": rows, "columns": columns, "counts": counts}


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
        "queue_confusion": queue_confusion(predictions),
    }


def map_until_error(fn: Callable, items: list, workers: int) -> list:
    """`fn` over `items` in parallel; the first exception cancels every call not yet started."""
    pool = ThreadPoolExecutor(workers)
    try:
        return list(pool.map(fn, items))
    finally:
        pool.shutdown(cancel_futures=True)


def evaluate(
    data_file: Path,
    build_prompt: Callable,
    model_config: ModelConfig,
    workers: int = 8,
    complete: Callable[[str, ModelConfig], str] | None = None,
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
    predictions = evaluate_rows(
        read_tsv(data_file), build_prompt, model_config, workers, complete
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
    # Microseconds so parallel runs don't overwrite each other.
    path = results_dir / f"{started_at:%Y%m%d-%H%M%S-%f}_{split}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2) + "\n")
    return path
