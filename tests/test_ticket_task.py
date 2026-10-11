import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import openai

from gepa_toy.cache import Cache, CachedModel
from gepa_toy.evaluate import labels_of
from gepa_toy.feedback import written_feedback
from gepa_toy.gepa import Rollout
from gepa_toy.model import Model
from gepa_toy.ticket_task import TicketTask
from gepa_toy.zero_shot import build

LABELS = {
    "queue": "Technical Support",
    "type": "Incident",
    "priority": "high",
    "tags": ["Security", "Outages & Recovery"],
}

ROWS = [
    {
        "id": "1",
        "subject": "Server down",
        "body": "The main server stopped responding.",
        "queue": "Technical Support",
        "type": "Incident",
        "priority": "high",
        "tag_groups": "Security,Outages & Recovery",
    },
    {
        "id": "2",
        "subject": "",
        "body": "Please refund my last invoice.",
        "queue": "Billing and Payments",
        "type": "Request",
        "priority": "low",
        "tag_groups": "Billing & Returns",
    },
    {
        "id": "3",
        "subject": "New laptop",
        "body": "I need a laptop for the new starter.",
        "queue": "IT Support",
        "type": "Request",
        "priority": "medium",
        "tag_groups": "Hardware & Infrastructure",
    },
]
REPLY = (
    '{"queue": "Technical Support", "type": "Incident", "priority": "high",'
    ' "tags": ["Security", "Outages & Recovery"]}'
)
CANDIDATE = "Classify carefully. Outages are always high priority."


class TicketTaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Cache(Path(self.tmp.name) / "cache.sqlite")
        self.model = CachedModel(Model(model="test/model", temperature=0.5), self.cache)
        self.task = TicketTask(self.model, "written", workers=2)

    def tearDown(self):
        self.tmp.cleanup()

    @patch.object(Model, "complete", return_value=REPLY)
    def test_one_rollout_per_row_in_order_from_the_candidate(self, complete):
        rollouts = self.task.evaluate(CANDIDATE, ROWS)
        self.assertEqual(len(rollouts), 3)
        # call_args_list is appended atomically; call_count is not, across worker threads.
        self.assertEqual(
            sorted(call.args for call in complete.call_args_list),
            sorted((build(CANDIDATE, r["subject"], r["body"]),) for r in ROWS),
        )
        for rollout, row in zip(rollouts, ROWS, strict=True):
            self.assertIsInstance(rollout, Rollout)
            self.assertEqual(rollout.record["id"], row["id"])
            self.assertEqual(rollout.record["labels"], labels_of(row))
            self.assertEqual(rollout.score, rollout.record["gepa_score"])
        self.assertEqual(
            set(rollouts[0].record),
            {"id", "predicted", "labels", "gepa_score", "error", "raw"},
        )
        self.assertEqual(rollouts[0].score, 1.0)
        self.assertEqual(rollouts[1].score, 0.0)
        self.assertEqual(
            rollouts[0].feedback,
            written_feedback(rollouts[0].record["predicted"], labels_of(ROWS[0]), None),
        )
        self.assertEqual(
            rollouts[0].trace,
            "Subject: Server down\n"
            "Body:\n"
            "The main server stopped responding.\n"
            "Predicted: queue Technical Support, type Incident, priority high,"
            " tags [Security, Outages & Recovery]",
        )

    @patch.object(Model, "complete", return_value=REPLY)
    def test_same_candidate_again_makes_no_model_calls(self, complete):
        first = self.task.evaluate(CANDIDATE, ROWS)
        second = self.task.evaluate(CANDIDATE, ROWS)
        self.assertEqual(len(complete.call_args_list), 3)
        self.assertEqual(first, second)
        self.task.evaluate("A different instruction.", ROWS)
        self.assertEqual(len(complete.call_args_list), 6)

    @patch.object(Model, "complete")
    def test_api_error_stops_and_is_not_cached(self, complete):
        complete.side_effect = openai.APIConnectionError(request=None)
        with self.assertRaises(openai.APIConnectionError):
            self.task.evaluate(CANDIDATE, ROWS[:1])
        complete.side_effect = None
        complete.return_value = REPLY
        [retried] = self.task.evaluate(CANDIDATE, ROWS[:1])
        self.assertEqual(len(complete.call_args_list), 2)
        self.assertEqual(retried.score, 1.0)

    @patch.object(Model, "complete", return_value="not json")
    def test_unparseable_reply_is_cached_and_scored_zero(self, complete):
        [rollout] = self.task.evaluate(CANDIDATE, ROWS[:1])
        [again] = self.task.evaluate(CANDIDATE, ROWS[:1])
        self.assertEqual(len(complete.call_args_list), 1)
        self.assertEqual(rollout.score, 0.0)
        self.assertEqual(rollout.record["raw"], "not json")
        self.assertEqual(rollout, again)

    @patch.object(Model, "complete", return_value=REPLY)
    def test_score_mode_feedback(self, _):
        task = TicketTask(CachedModel(Model(model="m"), self.cache), "score")
        rollouts = task.evaluate(CANDIDATE, ROWS[:2])
        self.assertEqual([r.feedback for r in rollouts], ["Score: 1.00", "Score: 0.00"])
        # The trace still carries the prediction, so the reflector has something to read.
        self.assertIn("Predicted: queue Technical Support", rollouts[0].trace)
        self.assertTrue(rollouts[1].trace.startswith("Subject: (no subject)\nBody:\n"))

    def test_describe(self):
        self.assertEqual(
            self.task.describe(),
            {"model": self.model.config, "feedback": "written"},
        )
        self.assertEqual(self.task.describe()["model"]["model"], "test/model")
        self.assertEqual(
            TicketTask(Model(model="m"), "score").describe()["feedback"],
            "score",
        )


if __name__ == "__main__":
    unittest.main()
