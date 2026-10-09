import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gepa_toy.cache import Cache
from gepa_toy.model import ModelConfig

PROMPT = "Classify this ticket."
CONFIG = ModelConfig()


class CountingComplete:
    def __init__(self, reply="reply", error=None):
        self.reply = reply
        self.error = error
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, prompt, model_config):
        with self._lock:
            self.calls += 1
        if self.error is not None:
            raise self.error
        return self.reply


def row_count(path):
    return sqlite3.connect(path).execute("SELECT COUNT(*) FROM replies").fetchone()[0]


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nested" / "cache.sqlite"
        self.cache = Cache(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hit_after_one_call(self):
        fake = CountingComplete()
        first = self.cache.complete(PROMPT, CONFIG, fake)
        second = self.cache.complete(PROMPT, CONFIG, fake)
        self.assertEqual((first, second), ("reply", "reply"))
        self.assertEqual(fake.calls, 1)

    def test_config_fields_that_change_the_reply_are_misses(self):
        base = Cache.key(PROMPT, CONFIG)
        for changed in (
            CONFIG.model_copy(update={"model": "other/model"}),
            CONFIG.model_copy(update={"temperature": 0.5}),
            CONFIG.model_copy(update={"max_tokens": 301}),
            CONFIG.model_copy(update={"reasoning": True}),
        ):
            self.assertNotEqual(Cache.key(PROMPT, changed), base)
        self.assertNotEqual(Cache.key("other prompt", CONFIG), base)

    def test_timeout_is_a_hit(self):
        fake = CountingComplete()
        self.cache.complete(PROMPT, CONFIG, fake)
        self.cache.complete(PROMPT, CONFIG.model_copy(update={"timeout": 5.0}), fake)
        self.assertEqual(fake.calls, 1)

    def test_error_is_not_stored_and_next_call_retries(self):
        failing = CountingComplete(error=RuntimeError("transient"))
        with self.assertRaises(RuntimeError):
            self.cache.complete(PROMPT, CONFIG, failing)
        self.assertEqual(row_count(self.path), 0)
        working = CountingComplete()
        self.assertEqual(self.cache.complete(PROMPT, CONFIG, working), "reply")
        self.assertEqual(working.calls, 1)

    def test_empty_reply_is_a_hit(self):
        fake = CountingComplete(reply="")
        self.cache.complete(PROMPT, CONFIG, fake)
        self.assertEqual(self.cache.complete(PROMPT, CONFIG, fake), "")
        self.assertEqual(fake.calls, 1)

    def test_api_calls_run_in_parallel(self):
        def slow(prompt, model_config):
            time.sleep(0.2)
            return "reply"

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(lambda i: self.cache.complete(f"p{i}", CONFIG, slow), range(8))
            )
        # Serialised calls would take 1.6s.
        self.assertLess(time.monotonic() - start, 0.8)

    def test_unparseable_reply_is_stored(self):
        fake = CountingComplete(reply="not json {")
        self.cache.complete(PROMPT, CONFIG, fake)
        self.assertEqual(self.cache.get(Cache.key(PROMPT, CONFIG)), "not json {")

    def test_set_existing_key_is_not_an_error(self):
        key = Cache.key(PROMPT, CONFIG)
        self.cache.set(key, "first")
        self.cache.set(key, "first")
        self.assertEqual(self.cache.get(key), "first")
        self.assertEqual(row_count(self.path), 1)

    def test_concurrent_workers_on_same_key(self):
        fake = CountingComplete()
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(
                pool.map(lambda _: self.cache.complete(PROMPT, CONFIG, fake), range(64))
            )
        self.assertEqual(results, ["reply"] * 64)
        self.assertEqual(row_count(self.path), 1)

    def test_reopen_sees_stored_row(self):
        self.cache.complete(PROMPT, CONFIG, CountingComplete())
        fake = CountingComplete()
        self.assertEqual(Cache(self.path).complete(PROMPT, CONFIG, fake), "reply")
        self.assertEqual(fake.calls, 0)


if __name__ == "__main__":
    unittest.main()
