import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from gepa_toy import config
from gepa_toy.data_processing.tsv import write_tsv
from gepa_toy.reflect import DSPY_REFLECTION_PROMPT, REFLECTION_PROMPT
from gepa_toy.zero_shot import SEED_INSTRUCTION

SCRIPT = config.PROJECT_ROOT / "scripts" / "run_gepa.py"
spec = importlib.util.spec_from_file_location("run_gepa", SCRIPT)
run_gepa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_gepa)

FIELDS = ["id", "subject", "body", "queue", "type", "priority", "tag_groups"]
ROWS = [
    {
        "id": "1",
        "subject": "Server down",
        "body": "The main server stopped responding.",
        "queue": "Technical Support",
        "type": "Incident",
        "priority": "high",
        "tag_groups": "Security",
    },
    {
        "id": "2",
        "subject": "Still down",
        "body": "The server is still not responding.",
        "queue": "Technical Support",
        "type": "Incident",
        "priority": "high",
        "tag_groups": "Security",
    },
]
RIGHT = (
    '{"queue": "Technical Support", "type": "Incident", "priority": "high",'
    ' "tags": ["Security"]}'
)
WRONG = RIGHT.replace("Technical Support", "IT Support")
IMPROVED = "Improved instruction"


def fake_complete(prompt, model_config):
    """Reflection calls propose IMPROVED; the task model is right only under it."""
    if model_config.reasoning:
        return f"Here you go:\n```\n{IMPROVED}\n```"
    return RIGHT if IMPROVED in prompt else WRONG


class StageTableTest(unittest.TestCase):
    def test_matches_the_phases_table(self):
        self.assertEqual(
            run_gepa.STAGES,
            {
                "1": {
                    "feedback": "score",
                    "selection": "best",
                    "reflection_prompt": "ours",
                },
                "2": {
                    "feedback": "written",
                    "selection": "best",
                    "reflection_prompt": "ours",
                },
                "3": {
                    "feedback": "written",
                    "selection": "pareto",
                    "reflection_prompt": "ours",
                },
                "3c": {
                    "feedback": "written",
                    "selection": "pareto",
                    "reflection_prompt": "dspy",
                },
            },
        )

    def test_stages_1_to_3_differ_only_in_feedback_and_selection(self):
        for a, b in (("1", "2"), ("2", "3"), ("1", "3")):
            differing = {
                k
                for k in run_gepa.STAGES[a]
                if run_gepa.STAGES[a][k] != run_gepa.STAGES[b][k]
            }
            self.assertTrue(differing <= {"feedback", "selection"}, (a, b, differing))

    def test_3c_differs_from_3_only_in_the_reflection_prompt(self):
        differing = {
            k
            for k in run_gepa.STAGES["3"]
            if run_gepa.STAGES["3"][k] != run_gepa.STAGES["3c"][k]
        }
        self.assertEqual(differing, {"reflection_prompt"})
        self.assertEqual(run_gepa.REFLECTION_PROMPTS["ours"], REFLECTION_PROMPT)
        self.assertEqual(run_gepa.REFLECTION_PROMPTS["dspy"], DSPY_REFLECTION_PROMPT)


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
        self.gepa_results = self.results / "gepa"
        patcher = patch.multiple(
            run_gepa,
            PROJECT_ROOT=self.root,
            SPLITS_DIR=splits,
            RESULTS_DIR=self.results,
            GEPA_RESULTS_DIR=self.gepa_results,
            CACHE_PATH=self.results / "cache.sqlite",
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        # evaluate() records data_file relative to its own PROJECT_ROOT.
        patcher = patch("gepa_toy.evaluate.PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, *argv) -> dict:
        complete = Mock(side_effect=fake_complete)
        with (
            patch("gepa_toy.model.complete", complete),
            patch("sys.argv", ["run_gepa.py", *argv]),
            redirect_stdout(io.StringIO()) as out,
        ):
            run_gepa.main()
        self.complete = complete
        self.stdout = out.getvalue()
        (path,) = self.results.glob("*.json")
        return json.loads(path.read_text())

    def test_stage_run_saves_test_run_linked_to_the_log(self):
        # Seed on pareto (2) + one iteration (5 + 5) + child on pareto (2) = 14.
        run = self.run_main("--stage", "3", "--seed", "7", "--budget", "14")

        meta = run["metadata"]
        self.assertEqual(meta["stage"], "3")
        self.assertEqual(meta["seed"], 7)
        self.assertEqual(meta["best_candidate"], 1)
        self.assertEqual(meta["data_file"], "splits/test.tsv")
        log_path = self.root / meta["log_path"]
        self.assertTrue(log_path.is_file())
        self.assertEqual(log_path.parent, self.gepa_results)
        self.assertEqual(log_path.name[22:], "_stage3_seed7.json")

        log = json.loads(log_path.read_text())
        self.assertEqual(log["metadata"]["stage"], "3")
        self.assertEqual(log["metadata"]["seed"], 7)
        self.assertEqual(log["metadata"]["settings"]["seed"], 7)
        self.assertEqual(log["metadata"]["settings"]["budget"], 14)
        self.assertEqual(log["metadata"]["settings"]["minibatch_size"], 5)
        self.assertEqual(log["metadata"]["reflection_prompt"], "ours")
        self.assertEqual(
            log["metadata"]["reflection_model"],
            {
                "model": config.REFLECTION_MODEL,
                "temperature": 0.0,
                "max_tokens": 4000,
                "reasoning": True,
                "timeout": 120.0,
            },
        )
        self.assertEqual(log["candidates"][0]["text"], SEED_INSTRUCTION)
        self.assertEqual([i["outcome"] for i in log["iterations"]], ["accepted"])
        self.assertEqual(log["iterations"][0]["calls_used"], 14)
        self.assertEqual(log["best"], 1)

        # The test split is scored with the best candidate, not the seed.
        self.assertIn(IMPROVED, meta["prompt"])
        self.assertEqual(run["results"]["queue"]["accuracy"], 1.0)
        self.assertEqual(run["results"]["n"], 2)
        reflection_calls = [
            c for c in self.complete.call_args_list if c.args[1].reasoning
        ]
        self.assertEqual(len(reflection_calls), 1)
        self.assertEqual(reflection_calls[0].args[1].model, config.REFLECTION_MODEL)
        self.assertIn(str(log_path.relative_to(self.root)), self.stdout)

    def test_each_stage_records_its_switches(self):
        for stage, switches in run_gepa.STAGES.items():
            with self.subTest(stage):
                run = self.run_main("--stage", stage, "--seed", "0", "--budget", "14")
                log = json.loads((self.root / run["metadata"]["log_path"]).read_text())
                self.assertEqual(
                    log["metadata"]["task"]["feedback"], switches["feedback"]
                )
                self.assertEqual(
                    log["metadata"]["settings"]["selection"], switches["selection"]
                )
                self.assertEqual(
                    log["metadata"]["reflection_prompt"], switches["reflection_prompt"]
                )
                sent = log["iterations"][0]["reflection_prompt"]
                template = run_gepa.REFLECTION_PROMPTS[switches["reflection_prompt"]]
                self.assertTrue(sent.startswith(template.split("{instruction}")[0]))
                for path in self.results.glob("*.json"):
                    path.unlink()

    def test_reflection_model_option_is_used(self):
        run = self.run_main(
            "--stage", "1", "--seed", "0", "--budget", "14", "--reflection-model", "x/y"
        )
        log = json.loads((self.root / run["metadata"]["log_path"]).read_text())
        self.assertEqual(log["metadata"]["reflection_model"]["model"], "x/y")
        self.assertEqual(
            {
                c.args[1].model
                for c in self.complete.call_args_list
                if c.args[1].reasoning
            },
            {"x/y"},
        )

    def test_stage_0_scores_the_seed_without_optimising(self):
        with patch.object(run_gepa, "optimise") as optimise:
            run = self.run_main("--stage", "0", "--seed", "1")
        optimise.assert_not_called()
        meta = run["metadata"]
        self.assertEqual(meta["stage"], "0")
        self.assertEqual(meta["seed"], 1)
        self.assertIsNone(meta["log_path"])
        self.assertIsNone(meta["best_candidate"])
        self.assertEqual(
            meta["prompt"].split("\n\nQueue (choose one):")[0], SEED_INSTRUCTION
        )
        self.assertEqual(run["results"]["queue"]["accuracy"], 0.0)
        self.assertFalse(self.gepa_results.exists())
        self.assertFalse(any(c.args[1].reasoning for c in self.complete.call_args_list))

    def test_test_split_goes_through_the_cache(self):
        self.run_main("--stage", "0", "--seed", "0")
        first = self.complete.call_count
        for path in self.results.glob("*.json"):
            path.unlink()
        self.run_main("--stage", "0", "--seed", "0")
        self.assertEqual(first, len(ROWS))
        self.assertEqual(self.complete.call_count, 0)

    def test_reflector_is_not_cached(self):
        # Design 4.2: a cached reflector would make the three seeds propose the same prompts.
        for _ in range(2):
            self.run_main("--stage", "3", "--seed", "0", "--budget", "14")
            for path in self.results.glob("*.json"):
                path.unlink()
        calls = [c.args[1].reasoning for c in self.complete.call_args_list]
        self.assertEqual(calls, [True])

    def test_same_second_runs_get_different_logs(self):
        for _ in range(2):
            self.run_main("--stage", "1", "--seed", "0", "--budget", "14")
            for path in self.results.glob("*.json"):
                path.unlink()
        self.assertEqual(len(list(self.gepa_results.glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
