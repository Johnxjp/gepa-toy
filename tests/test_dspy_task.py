import copy
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from dspy.utils.exceptions import LMAuthError, LMRateLimitError

from gepa_toy.config import LABELS
from gepa_toy.dspy_task import (
    AdapterParseError,
    ClassifyTicket,
    StopOnLMError,
    build_program,
    candidate_log,
    dspy,
    make_lm,
    metric,
    predict_row,
    to_example,
)
from gepa_toy.evaluate import gepa_score, labels_of
from gepa_toy.feedback import written_feedback
from gepa_toy.model import ModelConfig
from gepa_toy.zero_shot import SEED_INSTRUCTION

ROW = {
    "id": "7",
    "subject": "VPN down",
    "body": "Nobody can connect.",
    "queue": "IT Support",
    "type": "Incident",
    "priority": "high",
    "tag_groups": "Network,Outage",
}
WRONG = {
    "queue": "Technical Support",
    "type": "Incident",
    "priority": "low",
    "tags": ["Network"],
}


def fake_program(result):
    """Stands in for a DSPy program: returns `result`, or raises it if it's an exception."""
    program = MagicMock()
    if isinstance(result, Exception):
        program.side_effect = result
    else:
        program.return_value = dspy.Prediction(**result)
    return program


class ExampleTest(unittest.TestCase):
    def test_inputs_are_subject_and_body_and_labels_match(self):
        example = to_example(ROW)
        self.assertEqual(
            example.inputs().toDict(),
            {"subject": "VPN down", "body": "Nobody can connect."},
        )
        self.assertEqual(example.true_labels, labels_of(ROW))
        self.assertEqual(example.id, "7")

    def test_empty_subject_matches_our_template(self):
        self.assertEqual(to_example({**ROW, "subject": ""}).subject, "(no subject)")


class ProgramTest(unittest.TestCase):
    def test_starts_from_the_seed_instruction(self):
        program = build_program(make_lm(ModelConfig(model="m"), []))
        self.assertEqual(program.signature.instructions, SEED_INSTRUCTION)

    def test_lm_reaches_the_predictor(self):
        lm = make_lm(ModelConfig(model="m"), [])
        self.assertIs(build_program(lm).lm, lm)

    def test_outputs_are_the_four_fields(self):
        self.assertEqual(
            list(ClassifyTicket.output_fields), ["queue", "type", "priority", "tags"]
        )

    def test_lm_sends_the_reasoning_flag(self):
        stop = StopOnLMError(print)
        lm = make_lm(ModelConfig(model="some/model", reasoning=False), [stop])
        self.assertEqual(lm.model, "openrouter/some/model")
        self.assertEqual(lm.kwargs["extra_body"], {"reasoning": {"enabled": False}})
        self.assertEqual(lm.callbacks, [stop])


class MetricTest(unittest.TestCase):
    def test_same_score_and_feedback_as_our_task(self):
        labels = labels_of(ROW)
        result = metric(to_example(ROW), dspy.Prediction(**WRONG))
        self.assertEqual(result.score, gepa_score(WRONG, labels))
        self.assertEqual(result.feedback, written_feedback(WRONG, labels, None))


class PredictRowTest(unittest.TestCase):
    def test_success_is_scored(self):
        record = predict_row(fake_program(WRONG), ROW)
        self.assertEqual(record["predicted"], WRONG)
        self.assertEqual(record["gepa_score"], gepa_score(WRONG, labels_of(ROW)))
        self.assertIsNone(record["error"])

    def test_parse_failure_is_recorded_with_the_reply(self):
        error = AdapterParseError("JSONAdapter", ClassifyTicket, "garbage", "bad reply")
        record = predict_row(fake_program(error), ROW)
        self.assertIsNone(record["predicted"])
        self.assertEqual(record["gepa_score"], 0.0)
        self.assertEqual(record["raw"], "garbage")
        self.assertIn("bad reply", record["error"])

    def test_api_errors_propagate(self):
        for error in (
            LMRateLimitError("slow down", status=429),
            LMAuthError("bad key", status=401),
        ):
            with self.assertRaises(type(error)):
                predict_row(fake_program(error), ROW)


class StopOnLMErrorTest(unittest.TestCase):
    def stops(self, outputs, exception=None):
        reasons = []
        StopOnLMError(reasons.append).on_lm_end("call", outputs, exception)
        return reasons

    def test_failed_call_stops(self):
        reasons = self.stops(None, LMRateLimitError("slow down", status=429))
        self.assertEqual(reasons, ["LMRateLimitError: slow down"])

    def test_reply_without_choices_stops(self):
        self.assertEqual(self.stops([]), ["model reply had no choices"])

    def test_empty_reply_stops(self):
        for outputs in (
            [""],
            [None],
            [{"text": None, "reasoning_content": "Thinking..."}],
        ):
            self.assertEqual(self.stops(outputs), ["model reply was empty"], outputs)

    def test_normal_reply_does_not_stop(self):
        for outputs in (
            ["[[ ## queue ## ]]..."],
            [{"text": "```new```", "reasoning_content": "x"}],
        ):
            self.assertEqual(self.stops(outputs), [], outputs)


class SharedLMTest(unittest.TestCase):
    def test_program_copies_share_one_lm_and_its_history(self):
        # GEPA deep-copies the program per candidate; usage must still land on our LM.
        lm = make_lm(ModelConfig(model="m"), [])
        self.assertIs(copy.deepcopy(build_program(lm)).lm, lm)

    def test_json_mode_retry_is_off(self):
        self.assertFalse(dspy.settings.adapter.use_json_adapter_fallback)


class CandidateLogTest(unittest.TestCase):
    def test_lineage_and_wins(self):
        def candidate(text):
            return SimpleNamespace(
                predictors=lambda: [
                    SimpleNamespace(signature=SimpleNamespace(instructions=text))
                ]
            )

        results = SimpleNamespace(
            candidates=[candidate("seed"), candidate("child")],
            parents=[[None], [0]],
            val_aggregate_scores=[0.5, 0.75],
            val_subscores=[{0: 0.5, 1: 0.5}, {0: 1.0, 1: 0.5}],
            per_val_instance_best_candidates={0: {1}, 1: {0, 1}},
            discovery_eval_counts=[2, 14],
            best_idx=1,
            total_metric_calls=16,
            num_full_val_evals=2,
        )
        log = candidate_log(results, ["a", "b"])
        self.assertEqual(log["best"], 1)
        self.assertEqual(log["pareto_ids"], ["a", "b"])
        seed, child = log["candidates"]
        self.assertEqual((seed["instruction"], child["instruction"]), ("seed", "child"))
        self.assertEqual(child["parents"], [0])
        self.assertEqual(seed["examples_won"], [1])
        self.assertEqual(child["examples_won"], [0, 1])


class SeedInstructionTest(unittest.TestCase):
    def test_has_every_queue_type_and_priority_once_and_no_tags(self):
        for field in ("queue", "type", "priority"):
            for value in LABELS[field]:
                self.assertEqual(SEED_INSTRUCTION.count(f"- {value}: "), 1, value)
        listed = {
            line[2:].split(":")[0]
            for line in SEED_INSTRUCTION.splitlines()
            if line.startswith("- ")
        }
        self.assertFalse(listed & set(LABELS["tags"]))


if __name__ == "__main__":
    unittest.main()
