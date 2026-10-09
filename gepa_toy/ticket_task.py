"""The ticket task behind GEPA's Task protocol. See docs/gepa_design.md section 4.4."""

from typing import Literal

from gepa_toy import model, zero_shot
from gepa_toy.cache import Cache
from gepa_toy.evaluate import SINGLE_FIELDS, evaluate_rows, gepa_score, tag_scores
from gepa_toy.gepa import Rollout
from gepa_toy.model import ModelConfig

FeedbackMode = Literal["written", "score"]


def labels_line(labels: dict) -> str:
    fields = ", ".join(f"{field} {labels[field]}" for field in SINGLE_FIELDS)
    return f"{fields}, tags [{', '.join(labels['tags'])}]"


def predicted_line(predicted: dict | None, error: str | None) -> str:
    if predicted is None:
        # Pydantic errors span several lines; collapsed so the reflector sees one per ticket.
        return f"Reply was not valid: {' '.join(error.split())}"
    return f"Predicted: {labels_line(predicted)}"


def feedback(
    predicted: dict | None, labels: dict, error: str | None, mode: FeedbackMode
) -> str:
    score = f"Score: {gepa_score(predicted, labels):.2f}"
    if mode == "score":
        return score
    if predicted is None:
        matches = [0] * len(SINGLE_FIELDS)
        f1 = 0.0
    else:
        matches = [int(predicted[f] == labels[f]) for f in SINGLE_FIELDS]
        f1 = tag_scores(predicted["tags"], labels["tags"])["f1"]
    parts = ", ".join(f"{f} {m}" for f, m in zip(SINGLE_FIELDS, matches, strict=True))
    return "\n".join(
        [
            f"{score} ({parts}, tags F1 {f1:.2f})",
            predicted_line(predicted, error),
            f"Correct:   {labels_line(labels)}",
        ]
    )


class TicketTask:
    def __init__(
        self,
        model_config: ModelConfig,
        feedback: FeedbackMode,
        cache: Cache,
        workers: int = 8,
    ):
        self.model_config = model_config
        self.mode = feedback
        self.cache = cache
        self.workers = workers

    def evaluate(self, candidate: str, rows: list[dict]) -> list[Rollout]:
        records = evaluate_rows(
            rows,
            lambda subject, body: zero_shot.build(candidate, subject, body),
            self.model_config,
            self.workers,
            # model.complete looked up per call, so tests can patch it.
            complete=lambda prompt, config: self.cache.complete(
                prompt, config, model.complete
            ),
        )
        return [
            Rollout(
                score=record["gepa_score"],
                # Same empty-subject placeholder as the prompt, so the reflector sees what the model saw.
                trace=(
                    f"Subject: {row['subject'] or '(no subject)'}\nBody:\n{row['body']}\n"
                    + predicted_line(record["predicted"], record["error"])
                ),
                feedback=feedback(
                    record["predicted"], record["labels"], record["error"], self.mode
                ),
                record=record,
            )
            for row, record in zip(rows, records, strict=True)
        ]

    def describe(self) -> dict:
        return {"model": self.model_config.model_dump(), "feedback": self.mode}
