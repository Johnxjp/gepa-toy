import unittest

from gepa_toy.dspy_task import (
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
from gepa_toy.model import ModelConfig
from gepa_toy.ticket_task import feedback
from gepa_toy.zero_shot import SEED_INSTRUCTION
from tests.test_ticket_task import CASES, LABELS, ROWS

# After gepa_toy.dspy_task, which imports openai before dspy (see its comment).
from dspy.teleprompt.gepa.gepa import DspyGEPAResult  # isort: skip
from dspy.utils.dummies import DummyLM  # isort: skip
from dspy.utils.exceptions import (  # isort: skip
    AdapterParseError,
    LMAuthError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
)


class MetricTest(unittest.TestCase):
    def test_same_score_and_text_as_our_feedback(self):
        gold = dspy.Example(true_labels=LABELS)
        # DSPy never calls the metric on a failed parse, so that case has no pred.
        for name, predicted, error, expected in CASES:
            if predicted is None:
                continue
            with self.subTest(name):
                out = metric(gold, dspy.Prediction(**predicted))
                self.assertEqual(out.score, gepa_score(predicted, LABELS))
                self.assertEqual(
                    out.feedback, feedback(predicted, LABELS, None, "written")
                )
                self.assertEqual(out.feedback, expected)
                # GEPA's wrapper reads these by item, not attribute.
                self.assertEqual(out["feedback"], expected)


class ConversionTest(unittest.TestCase):
    def test_row_round_trips_to_example(self):
        row = ROWS[0]
        example = to_example(row)
        self.assertEqual(set(example.inputs().keys()), {"subject", "body"})
        self.assertEqual(example.subject, row["subject"])
        self.assertEqual(example.body, row["body"])
        self.assertEqual(example.id, row["id"])
        self.assertEqual(example.true_labels, labels_of(row))

    def test_empty_subject_matches_our_template(self):
        self.assertEqual(to_example(ROWS[1]).subject, "(no subject)")


class ProgramTest(unittest.TestCase):
    def test_seed_instruction_word_for_word(self):
        lm = make_lm(ModelConfig())
        program = build_program(lm)
        self.assertEqual(program.signature.instructions, SEED_INSTRUCTION)
        self.assertIs(program.get_lm(), lm)
        self.assertEqual(list(ClassifyTicket.input_fields), ["subject", "body"])
        self.assertEqual(
            list(ClassifyTicket.output_fields), ["queue", "type", "priority", "tags"]
        )

    def test_lm_targets_openrouter_with_the_reasoning_flag(self):
        config = ModelConfig(model="some/model", reasoning=True, max_tokens=42)
        lm = make_lm(config)
        self.assertEqual(lm.model, "openrouter/some/model")
        self.assertEqual(lm.kwargs["extra_body"], {"reasoning": {"enabled": True}})
        self.assertEqual(lm.kwargs["max_tokens"], 42)
        self.assertEqual(lm.kwargs["temperature"], 0.0)
        self.assertEqual(lm.callbacks, [])

    def test_lm_error_reaches_the_stop_callback(self):
        class FailingLM(DummyLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                raise LMRateLimitError("quota")

        stops = []
        lm = FailingLM([])
        lm.callbacks = [StopOnLMError(stops.append)]
        program = build_program(lm)
        with self.assertRaises(LMRateLimitError):
            program(subject="a", body="b")
        self.assertEqual(stops, ["LMRateLimitError: quota"])

    def test_reflection_stop_ignores_transient_errors_but_not_setup_errors(self):
        # Design 4.5: a transient reflection error is logged and the loop goes on.
        stops = []
        stop = StopOnLMError(stops.append, stop_on_transient=False)
        stop.on_lm_end("id", None, LMServerError("upstream 500"))
        stop.on_lm_end("id", None, LMTimeoutError("timed out"))
        stop.on_lm_end("id", None, LMRateLimitError("quota"))
        self.assertEqual(stops, [])
        stop.on_lm_end("id", None, LMAuthError("bad key"))
        self.assertEqual(stops, ["LMAuthError: bad key"])

    def test_stop_callback_ignores_successful_calls(self):
        stops = []
        StopOnLMError(stops.append).on_lm_end("id", ["reply"], None)
        self.assertEqual(stops, [])


class PredictRowTest(unittest.TestCase):
    def test_record_has_the_evaluate_predict_shape(self):
        predicted = {**LABELS, "priority": "low"}

        def program(**inputs):
            self.assertEqual(set(inputs), {"subject", "body"})
            return dspy.Prediction(**predicted)

        record = predict_row(program, ROWS[0])
        self.assertEqual(
            record,
            {
                "id": "1",
                "predicted": predicted,
                "labels": labels_of(ROWS[0]),
                "gepa_score": 0.75,
                "error": None,
                "raw": None,
            },
        )

    def test_parse_failure_is_a_failed_prediction(self):
        def program(**inputs):
            raise AdapterParseError("ChatAdapter", ClassifyTicket, "garbage reply")

        record = predict_row(program, ROWS[0])
        self.assertIsNone(record["predicted"])
        self.assertEqual(record["gepa_score"], 0.0)
        self.assertIn("ChatAdapter", record["error"])
        self.assertEqual(record["raw"], "garbage reply")

    def test_transient_lm_error_is_a_failed_prediction(self):
        def program(**inputs):
            raise LMRateLimitError("quota")

        record = predict_row(program, ROWS[0])
        self.assertIsNone(record["predicted"])
        self.assertEqual(record["gepa_score"], 0.0)
        self.assertEqual(record["error"], "API error: quota")
        self.assertIsNone(record["raw"])

    def test_setup_error_crashes(self):
        def program(**inputs):
            raise LMAuthError("bad key")

        with self.assertRaises(LMAuthError):
            predict_row(program, ROWS[0])


class CandidateLogTest(unittest.TestCase):
    def test_converts_detailed_results(self):
        results = DspyGEPAResult(
            candidates=[
                dspy.Predict(ClassifyTicket.with_instructions(text))
                for text in ("seed", "child")
            ],
            parents=[[None], [0]],
            val_aggregate_scores=[0.5, 0.75],
            # Keys out of order on purpose: scores must come back in valset order.
            val_subscores=[{1: 0.0, 0: 1.0}, {1: 0.5, 0: 1.0}],
            per_val_instance_best_candidates={0: {0, 1}, 1: {1}},
            discovery_eval_counts=[2, 12],
            total_metric_calls=20,
            num_full_val_evals=2,
        )
        self.assertEqual(
            candidate_log(results),
            {
                "candidates": [
                    {
                        "id": 0,
                        "text": "seed",
                        "parent": None,
                        "pareto_scores": [1.0, 0.0],
                        "mean_pareto_score": 0.5,
                        "pareto_wins": [0],
                        "calls_when_found": 2,
                    },
                    {
                        "id": 1,
                        "text": "child",
                        "parent": 0,
                        "pareto_scores": [1.0, 0.5],
                        "mean_pareto_score": 0.75,
                        "pareto_wins": [0, 1],
                        "calls_when_found": 12,
                    },
                ],
                "best": 1,
                "calls_used": 20,
            },
        )


if __name__ == "__main__":
    unittest.main()
