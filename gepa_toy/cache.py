import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from gepa_toy.model import ModelConfig


class Cache:
    """SQLite cache of raw model replies keyed by prompt and model config."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, raw TEXT)"
            )

    @staticmethod
    def key(prompt: str, model_config: ModelConfig) -> str:
        # Timeout is left out: it changes whether a reply arrives, not what it says.
        fields = {
            "prompt": prompt,
            "model": model_config.model,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "reasoning": model_config.reasoning,
        }
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT raw FROM replies WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else row[0]

    def set(self, key: str, raw: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO replies (key, raw) VALUES (?, ?)", (key, raw)
            )

    def complete(
        self,
        prompt: str,
        model_config: ModelConfig,
        complete: Callable[[str, ModelConfig], str],
    ) -> str:
        key = self.key(prompt, model_config)
        raw = self.get(key)
        if raw is None:
            # The API call runs outside the lock so workers are not serialised.
            self.set(key, complete(prompt, model_config))
            # Another process may have stored its reply for this key while we waited. The
            # task model is not deterministic at temperature 0, so return the stored reply,
            # or parallel runs (and later hits in this one) see different answers.
            raw = self.get(key)
        return raw
