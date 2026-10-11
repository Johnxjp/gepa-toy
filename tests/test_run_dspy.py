import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from gepa_toy import config
from gepa_toy.data_processing.tsv import write_tsv
from gepa_toy.dspy_task import ClassifyTicket, dspy
from gepa_toy.ticket_task import feedback
from gepa_toy.zero_shot import SEED_INSTRUCTION
from tests.test_run_gepa import FIELDS, ROWS

# After gepa_toy.dspy_task, which imports openai before dspy (see its comment).
from dspy.teleprompt.gepa.gepa import DspyGEPAResult  # isort: skip
from dspy.utils.dummies import DummyLM  # isort: skip

SCRIPT = config.PROJECT_ROOT / "scripts" / "run_dspy.py"
spec = importlib.util.spec_from_file_location("run_dspy", SCRIPT)
run_dspy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_dspy)

ANSWER = {"queue": "IT Support", "type": "Incident", "priority": "high", "tags": ["HR"]}
CHILD = SEED_INSTRUCTION + "\nAlways pick IT Support."


class FakeGEPA:
    """Stands in for dspy.GEPA: records what it was built with, returns one child."""

    built = None

    def __init__(self, **kwargs):
        FakeGEPA.built = kwargs

    def compile(self, program, trainset, valset):
        FakeGEPA.compiled = (program, trainset, valset)
        child = dspy.Predict(ClassifyTicket.with_instructions(CHILD))
        n = len(valset)
        child.detailed_results = DspyGEPAResult(
            candidates=[program, child],
            parents=[[None], [0]],
            val_aggregate_scores=[0.25, 0.5],
            val_subscores=[dict.fromkeys(range(n), 0.25), dict.fromkeys(range(n), 0.5)],
            per_val_instance_best_candidates={i: {1} for i in range(n)},
            discovery_eval_counts=[n, n + 12],
            total_metric_calls=n + 12 + n,
            num_full_val_evals=2,
        )
        return child


class MainTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        splits = self.root / "splits"
        splits.mkdir()
        for split in ("train", "pareto", "test"):
            write_tsv(splits / f"{split}.tsv", ROWS, FIELDS)
        self.results = self.root / "results"
        self.lms = []

        def fake_make_lm(model_config, callbacks=None):
            self.lms.append((model_config, callbacks))
            return DummyLM([ANSWER] * 10)

        patcher = patch.multiple(
            run_dspy,
            PROJECT_ROOT=self.root,
            SPLITS_DIR=splits,
            RESULTS_DIR=self.results,
            GEPA_RESULTS_DIR=self.results / "gepa",
            make_lm=fake_make_lm,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(run_dspy.dspy, "GEPA", FakeGEPA)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, *argv) -> dict:
        with (
            patch("sys.argv", ["run_dspy.py", *argv]),
            redirect_stdout(io.StringIO()) as out,
        ):
            run_dspy.main()
        self.stdout = out.getvalue()
        (path,) = self.results.glob("*.json")
        return json.loads(path.read_text())

    def test_saves_test_run_linked_to_the_log_with_run_gepa_metadata_keys(self):
        run = self.run_main("--seed", "2", "--budget", "40")
        meta = run["metadata"]
        self.assertEqual(
            set(meta),
            {
                "started_at",
                "git",
                "data_file",
                "model",
                "prompt",
                "runtime_seconds",
                "stage",
                "seed",
                "log_path",
                "best_candidate",
                "budget",
            },
        )
        self.assertEqual((meta["stage"], meta["seed"]), ("4", 2))
        self.assertEqual(meta["data_file"], "splits/test.tsv")
        self.assertEqual(meta["prompt"], CHILD)
        self.assertEqual(meta["best_candidate"], 1)
        self.assertEqual(run["results"]["n"], len(ROWS))
        self.assertEqual(run["results"]["queue"]["accuracy"], 0.0)

        log_path = self.root / meta["log_path"]
        self.assertEqual(log_path.parent, self.results / "gepa")
        self.assertEqual(log_path.name[22:], "_stage4_seed2.json")
        log = json.loads(log_path.read_text())
        self.assertEqual(set(log), {"metadata", "candidates", "best", "calls_used"})
        self.assertEqual((log["metadata"]["stage"], log["metadata"]["seed"]), ("4", 2))
        self.assertEqual(log["metadata"]["budget"], 40)
        self.assertEqual(log["metadata"]["task"]["feedback"], "written")
        self.assertEqual(
            log["metadata"]["reflection_model"]["model"], config.REFLECTION_MODEL
        )
        self.assertEqual(
            [c["text"] for c in log["candidates"]], [SEED_INSTRUCTION, CHILD]
        )
        self.assertEqual(log["best"], 1)
        self.assertIn(str(log_path), self.stdout)

    def test_gepa_settings_seed_instruction_and_metric_match_the_design(self):
        self.run_main(
            "--seed",
            "1",
            "--budget",
            "40",
            "--reflection-model",
            "x/y",
            "--model",
            "paid/model",
            "--max-tokens",
            "500",
        )
        built = FakeGEPA.built
        self.assertEqual(built["max_metric_calls"], 40)
        self.assertEqual(built["reflection_minibatch_size"], 5)
        self.assertEqual(built["candidate_selection_strategy"], "pareto")
        self.assertTrue(built["skip_perfect_score"])
        self.assertFalse(built["use_merge"])
        self.assertTrue(built["track_stats"])
        self.assertEqual(built["seed"], 1)
        program, trainset, valset = FakeGEPA.compiled
        self.assertEqual(program.signature.instructions, SEED_INSTRUCTION)
        self.assertEqual(len(trainset), len(ROWS))
        self.assertEqual(len(valset), len(ROWS))
        self.assertEqual(set(valset[0].inputs().keys()), {"subject", "body"})
        # The metric DSPy gets is our score and written feedback.
        out = built["metric"](valset[0], dspy.Prediction(**ANSWER))
        self.assertEqual(out.score, 0.5)
        self.assertEqual(
            out.feedback, feedback(ANSWER, valset[0].true_labels, None, "written")
        )
        # Task LM with the stop callback, reflection LM, then a plain task LM for test scoring.
        models = [(c.model, c.max_tokens, cb is not None) for c, cb in self.lms]
        self.assertEqual(
            models,
            [
                ("paid/model", 500, True),
                ("x/y", 16000, True),
                ("paid/model", 500, False),
            ],
        )
        # A transient reflection error skips a proposal; a task-model one stops the run.
        self.assertTrue(self.lms[0][1][0].stop_on_transient)
        self.assertFalse(self.lms[1][1][0].stop_on_transient)
        self.assertTrue(self.lms[1][0].reasoning)
        self.assertEqual(self.lms[1][0].temperature, 1.0)
        (run_file,) = self.results.glob("*.json")
        self.assertEqual(
            json.loads(run_file.read_text())["metadata"]["model"]["model"], "paid/model"
        )


if __name__ == "__main__":
    unittest.main()
