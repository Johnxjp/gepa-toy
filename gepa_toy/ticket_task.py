"""The ticket task behind GEPA's Task protocol."""

from typing import Literal

from gepa_toy import model, zero_shot
from gepa_toy.cache import Cache
from gepa_toy.evaluate import evaluate_rows
from gepa_toy.feedback import prediction_line, score_feedback, written_feedback
from gepa_toy.gepa import Rollout
from gepa_toy.model import ModelConfig

FeedbackMode = Literal["written", "score"]


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
        return {"model": self.model_config.model_dump(), "feedback": self.mode}
