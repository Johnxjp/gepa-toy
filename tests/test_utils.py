import subprocess
import unittest
from unittest.mock import patch

from gepa_toy.utils import git_version

STATUS = " M AGENTS.md\n?? gepa_toy/new.py\n"


def fake_git(args, **_):
    stdout = STATUS if "status" in args else "abc123\n"
    return subprocess.CompletedProcess(args, 0, stdout=stdout)


class GitVersionTest(unittest.TestCase):
    @patch("gepa_toy.utils.subprocess.run", side_effect=fake_git)
    def test_untracked_files_make_it_dirty(self, _):
        version = git_version()
        self.assertEqual(version["commit"], "abc123")
        self.assertTrue(version["dirty"])
        # The leading space is part of git's status code and must survive.
        self.assertEqual(version["changes"], [" M AGENTS.md", "?? gepa_toy/new.py"])

    @patch("gepa_toy.utils.subprocess.run")
    def test_results_dir_is_excluded(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="")
        self.assertFalse(git_version()["dirty"])
        status_args = run.call_args_list[0].args[0]
        self.assertIn(":!results", status_args)
        self.assertNotIn("--untracked-files=no", status_args)


if __name__ == "__main__":
    unittest.main()
