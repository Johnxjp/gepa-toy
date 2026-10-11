"""The ticket task behind GEPA's Task protocol."""

from typing import Literal

from gepa_toy import zero_shot
from gepa_toy.cache import CachedModel
from gepa_toy.evaluate import evaluate_rows
from gepa_toy.feedback import prediction_line, score_feedback, written_feedback
from gepa_toy.gepa import Rollout
from gepa_toy.model import Model

FeedbackMode = Literal["written", "score"]


class TicketTask:
    def __init__(
        self,
        model: Model | CachedModel,
        feedback: FeedbackMode,
        workers: int = 8,
    ):
        self.model = model
        self.mode = feedback
        self.workers = workers

    def evaluate(self, candidate: str, rows: list[dict]) -> list[Rollout]:
        records = evaluate_rows(
            rows,
            lambda subject, body: zero_shot.build(candidate, subject, body),
            self.model,
            self.workers,
        )
        return [
            Rollout(
                score=record["gepa_score"],
                # Same empty-subject placeholder as the prompt, so the reflector sees what the model saw.
                trace=(
                    f"Subject: {row['subject'] or '(no subject)'}\nBody:\n{row['body']}\n"
                    + prediction_line(record["predicted"], record["error"])
                ),
                feedback=(
                    score_feedback(record["predicted"], record["labels"])
                    if self.mode == "score"
                    else written_feedback(
                        record["predicted"], record["labels"], record["error"]
                    )
                ),
                record=record,
            )
            for row, record in zip(rows, records, strict=True)
        ]

    def describe(self) -> dict:
        return {"model": self.model.config, "feedback": self.mode}
