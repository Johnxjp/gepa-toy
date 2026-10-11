import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gepa_toy.cache import Cache, CachedModel
from gepa_toy.model import Model

PROMPT = "Classify this ticket."
MODEL = Model(model="m")


class CountingModel:
    """Stands in for a `Model`: same settings as MODEL, a fixed reply, and a call count."""

    def __init__(self, reply="reply", error=None, complete=None):
        self.model = MODEL.model
        self.temperature = MODEL.temperature
        self.max_tokens = MODEL.max_tokens
        self.reasoning = MODEL.reasoning
        self.reply = reply
        self.error = error
        self._complete = complete
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, prompt):
        with self._lock:
            self.calls += 1
        if self._complete is not None:
            return self._complete(prompt)
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

    def cached(self, model, cache=None):
        return CachedModel(model, cache or self.cache)

    def test_hit_after_one_call(self):
        fake = CountingModel()
        first = self.cached(fake).complete(PROMPT)
        second = self.cached(fake).complete(PROMPT)
        self.assertEqual((first, second), ("reply", "reply"))
        self.assertEqual(fake.calls, 1)

    def test_keys_match_replies_stored_by_earlier_runs(self):
        # Captured before the Model refactor. A different key would orphan every paid reply
        # already in the cache, silently.
        self.assertEqual(
            Cache.key(PROMPT, MODEL),
            "032214246ede81c1759a0afc3911aa2449f2ead98b375b5d0d16737104b72ea5",
        )
        self.assertEqual(
            Cache.key(
                PROMPT,
                Model(
                    model="some/model", temperature=0.7, max_tokens=123, reasoning=True
                ),
            ),
            "eb0c71fdf48ae2af371bb87abcde9dd6810845f44b67d755e1d73f9d4b6bc6ab",
        )

    def test_settings_that_change_the_reply_are_misses(self):
        base = Cache.key(PROMPT, MODEL)
        for changed in (
            Model(model="other/model"),
            Model(model="m", temperature=0.5),
            Model(model="m", max_tokens=301),
            Model(model="m", reasoning=True),
        ):
            self.assertNotEqual(Cache.key(PROMPT, changed), base)
        self.assertNotEqual(Cache.key("other prompt", MODEL), base)

    def test_timeout_is_a_hit(self):
        self.assertEqual(
            Cache.key(PROMPT, Model(model="m", timeout=5.0)),
            Cache.key(PROMPT, MODEL),
        )

    def test_error_is_not_stored_and_next_call_retries(self):
        failing = CountingModel(error=RuntimeError("transient"))
        with self.assertRaises(RuntimeError):
            self.cached(failing).complete(PROMPT)
        self.assertEqual(row_count(self.path), 0)
        working = CountingModel()
        self.assertEqual(self.cached(working).complete(PROMPT), "reply")
        self.assertEqual(working.calls, 1)

    def test_empty_reply_is_a_hit(self):
        fake = CountingModel(reply="")
        self.cached(fake).complete(PROMPT)
        self.assertEqual(self.cached(fake).complete(PROMPT), "")
        self.assertEqual(fake.calls, 1)

    def test_api_calls_run_in_parallel(self):
        def slow(prompt):
            time.sleep(0.2)
            return "reply"

        cached = self.cached(CountingModel(complete=slow))
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: cached.complete(f"p{i}"), range(8)))
        # Serialised calls would take 1.6s.
        self.assertLess(time.monotonic() - start, 0.8)

    def test_unparseable_reply_is_stored(self):
        self.cached(CountingModel(reply="not json {")).complete(PROMPT)
        self.assertEqual(self.cache.get(Cache.key(PROMPT, MODEL)), "not json {")

    def test_set_existing_key_is_not_an_error(self):
        key = Cache.key(PROMPT, MODEL)
        self.cache.set(key, "first")
        self.cache.set(key, "first")
        self.assertEqual(self.cache.get(key), "first")
        self.assertEqual(row_count(self.path), 1)

    def test_concurrent_workers_on_same_key(self):
        cached = self.cached(CountingModel())
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda _: cached.complete(PROMPT), range(64)))
        self.assertEqual(results, ["reply"] * 64)
        self.assertEqual(row_count(self.path), 1)

    def test_racing_miss_returns_the_stored_reply(self):
        # Another process stores its reply while this call waits on the API. The task
        # model is not deterministic at temperature 0, so the two replies can differ;
        # every caller must get the stored one or parallel stage 0 runs disagree.
        other = self.cached(CountingModel(reply="stored first"), Cache(self.path))

        def slow_complete(prompt):
            other.complete(prompt)
            return "arrived second"

        cached = self.cached(CountingModel(complete=slow_complete))
        self.assertEqual(cached.complete(PROMPT), "stored first")

    def test_reopen_sees_stored_row(self):
        self.cached(CountingModel()).complete(PROMPT)
        fake = CountingModel()
        self.assertEqual(self.cached(fake, Cache(self.path)).complete(PROMPT), "reply")
        self.assertEqual(fake.calls, 0)

    def test_config_is_the_inner_models(self):
        # Logged as the task model's settings, so cached and uncached runs log the same shape.
        self.assertEqual(self.cached(MODEL).config, MODEL.config)


if __name__ == "__main__":
    unittest.main()
