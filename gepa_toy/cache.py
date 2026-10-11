import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from gepa_toy.model import Model


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
    def key(prompt: str, model: Model) -> str:
        # Timeout is left out: it changes whether a reply arrives, not what it says.
        fields = {
            "prompt": prompt,
            "model": model.model,
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
            "reasoning": model.reasoning,
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


class CachedModel:
    """A model whose replies are read from and stored in a `Cache`.

    Never wrap the reflection model: at temperature 1 the seeds would get the same stored
    rewrite for the same prompt and stop exploring different ones.
    """

    def __init__(self, model: Model, cache: Cache):
        self.model = model
        self.cache = cache

    def complete(self, prompt: str) -> str:
        """The stored reply for `prompt` and this model's settings, calling the model on a miss.

        Errors from the model propagate and nothing is stored, so the next call retries.
        """
        key = Cache.key(prompt, self.model)
        raw = self.cache.get(key)
        if raw is None:
            # The API call runs outside the lock so workers are not serialised.
            self.cache.set(key, self.model.complete(prompt))
            # Another process may have stored its reply for this key while we waited. The
            # task model is not deterministic at temperature 0, so return the stored reply,
            # or parallel runs (and later hits in this one) see different answers.
            raw = self.cache.get(key)
        return raw

    @property
    def config(self) -> dict:
        return self.model.config
