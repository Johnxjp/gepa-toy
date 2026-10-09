import subprocess

from gepa_toy.config import PROJECT_ROOT


def git_version() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout

    # Untracked files count: new, uncommitted code would otherwise be recorded as a clean commit.
    changes = git("status", "--porcelain", "--", ".", ":!results").splitlines()
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(changes),
        "changes": changes,
    }
