import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_dspy import REFLECTION_MODEL, TASK_MODEL, parse_args

SAVED = {
    "budget": 1000,
    "seed": 0,
    "eval_split": "label_key",
    "max_tokens": 1000,
    "workers": 8,
    "task_model": TASK_MODEL,
    "reflection_model": REFLECTION_MODEL,
}


class ResumeTest(unittest.TestCase):
    def resume(self, saved: dict, *extra: str):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "args.json").write_text(json.dumps(saved))
            return parse_args(["--resume", tmp, *extra])

    def test_uses_the_saved_arguments(self):
        args, _ = self.resume(SAVED)
        self.assertEqual(args.eval_split, "label_key")
        self.assertEqual(args.budget, 1000)

    def test_refuses_other_options(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.resume(SAVED, "--seed", "1")

    def test_refuses_when_config_models_changed(self):
        changed = {**SAVED, "task_model": "some/other-model"}
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.resume(changed)


if __name__ == "__main__":
    unittest.main()
