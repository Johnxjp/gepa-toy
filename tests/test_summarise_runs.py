import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from statistics import stdev
from unittest.mock import patch

from gepa_toy import config

SCRIPT = config.PROJECT_ROOT / "scripts" / "summarise_runs.py"
spec = importlib.util.spec_from_file_location("summarise_runs", SCRIPT)
summarise_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarise_runs)


def run(stage, accuracy, score):
    metadata = {} if stage is None else {"stage": stage}
    return {
        "metadata": metadata,
        "results": {"gepa_score": score, "queue": {"accuracy": accuracy}},
    }


class SummariseRunsTest(unittest.TestCase):
    def test_mean_sample_stdev_and_n_per_stage_in_stage_order(self):
        runs = [
            run("3", 0.5, 0.6),
            run("3c", 0.4, 0.5),
            run("3", 0.7, 0.9),
            run("0", 0.2, 0.3),
            run(4, 0.8, 0.85),
            run("3", 0.6, 0.6),
        ]
        rows = summarise_runs.summarise_runs(runs)
        self.assertEqual([r["stage"] for r in rows], ["0", "3", "3c", "4"])
        self.assertEqual([r["n"] for r in rows], [1, 3, 1, 1])
        stage3 = rows[1]
        self.assertAlmostEqual(stage3["queue_accuracy"], 0.6)
        self.assertAlmostEqual(stage3["queue_accuracy_sd"], stdev([0.5, 0.7, 0.6]))
        self.assertAlmostEqual(stage3["gepa_score"], 0.7)
        self.assertAlmostEqual(stage3["gepa_score_sd"], stdev([0.6, 0.9, 0.6]))

    def test_single_run_has_stdev_0(self):
        (row,) = summarise_runs.summarise_runs([run("1", 0.5, 0.6)])
        self.assertEqual(row["queue_accuracy"], 0.5)
        self.assertEqual(row["queue_accuracy_sd"], 0.0)
        self.assertEqual(row["gepa_score"], 0.6)
        self.assertEqual(row["gepa_score_sd"], 0.0)

    def test_no_runs_gives_no_rows(self):
        self.assertEqual(summarise_runs.summarise_runs([]), [])

    def test_unknown_stage_is_refused(self):
        with self.assertRaises(ValueError):
            summarise_runs.summarise_runs([run("9", 0.5, 0.6)])

    def test_main_reads_only_runs_with_a_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "gepa").mkdir()
            (results / "a_test.json").write_text(json.dumps(run("2", 0.5, 0.6)))
            (results / "b_test.json").write_text(json.dumps(run("2", 0.7, 0.8)))
            (results / "c_train.json").write_text(json.dumps(run(None, 0.1, 0.1)))
            (results / "gepa" / "log.json").write_text("{}")
            (results / "cache.sqlite").write_bytes(b"")
            with (
                patch.object(summarise_runs, "RESULTS_DIR", results),
                redirect_stdout(io.StringIO()) as out,
            ):
                summarise_runs.main()
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            lines[1].split(), ["2", "2", "0.600", "0.141", "0.700", "0.141"]
        )


if __name__ == "__main__":
    unittest.main()
