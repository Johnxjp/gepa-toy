import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gepa_toy.gepa import Rollout, Settings, optimise

GIT = {"commit": "test", "dirty": False, "changes": []}


class FakeTask:
    """Scores each example with score(candidate, example) and records every evaluate call."""

    def __init__(self, score):
        self.score = score
        self.calls = []  # (candidate, examples)

    def evaluate(self, candidate, examples):
        self.calls.append((candidate, list(examples)))
        return [
            Rollout(
                score=self.score(candidate, e),
                trace=str(e),
                feedback=f"missing {e}",
                record={"example": e},
            )
            for e in examples
        ]

    def describe(self):
        return {"task": "fake"}

    @property
    def calls_used(self):
        return sum(len(examples) for _, examples in self.calls)


def keyword_score(candidate, example):
    """Planted answer: an example scores 1 when its keyword is in the candidate.

    A "!word" example scores 1 when the word is absent, so a child that helps on the
    minibatch can hurt on pareto.
    """
    words = candidate.split()
    if example.startswith("!"):
        return 0.0 if example[1:] in words else 1.0
    return 1.0 if example in words else 0.0


def add_keyword(parent, rollouts):
    """Fake reflector: append the first keyword the feedback says is missing."""
    missing = next(r.feedback for r in rollouts if r.score < 1)
    return f"{parent} {missing.split()[1]}".strip()


def append_x(parent, rollouts):
    return parent + "x"


def longer_is_better(candidate, example):
    return min(len(candidate), 20) / 20


class CountingPropose:
    def __init__(self, propose):
        self.propose = propose
        self.calls = 0

    def __call__(self, parent, rollouts):
        self.calls += 1
        return self.propose(parent, rollouts)


class RecordingPropose:
    """Like make_reflector's propose: keeps the last prompt and reply, even when it raises."""

    def __init__(self, fail_first):
        self.fail_first = fail_first
        self.last_prompt = self.last_reply = None

    def __call__(self, parent, rollouts):
        self.last_prompt = self.last_reply = None
        self.last_prompt = f"prompt for {parent!r}"
        if self.fail_first:
            self.fail_first = False
            raise ValueError("no fenced block")
        self.last_reply = f"reply for {parent!r}"
        return add_keyword(parent, rollouts)


class GepaTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "run.json"
        patcher = patch("gepa_toy.gepa.git_version", return_value=GIT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def optimise(self, task, propose, train, pareto, **settings):
        settings = Settings(selection="best", **settings)
        return optimise("", train, pareto, task, propose, settings, self.log_path)

    def test_budget_never_exceeded_wherever_it_lands(self):
        pareto = ["p1", "p2", "p3"]
        train = ["t1", "t2", "t3", "t4", "t5"]
        outcomes = set()
        for budget in range(len(pareto), 40):
            with self.subTest(budget=budget):
                task = FakeTask(longer_is_better)  # every child is accepted
                log = self.optimise(
                    task, append_x, train, pareto, budget=budget, minibatch_size=2
                )
                self.assertLessEqual(task.calls_used, budget)
                if log["iterations"]:
                    self.assertEqual(
                        log["iterations"][-1]["calls_used"], task.calls_used
                    )
                outcomes.update(i["outcome"] for i in log["iterations"])
        # Some budgets land after an accepted child with too little left to score it on pareto.
        self.assertIn("out of budget", outcomes)

    def test_out_of_budget_child_is_not_added(self):
        # Seed 3, parent 2, child 2 = 7 used, 2 left: the child is accepted but cannot be scored.
        task = FakeTask(longer_is_better)
        log = self.optimise(
            task, append_x, ["t1", "t2"], ["p1", "p2", "p3"], budget=9, minibatch_size=2
        )
        self.assertEqual([i["outcome"] for i in log["iterations"]], ["out of budget"])
        self.assertEqual(task.calls_used, 7)
        self.assertEqual(len(log["candidates"]), 1)
        self.assertIsNone(log["iterations"][0]["child"])

    def test_seed_pareto_scoring_is_counted(self):
        task = FakeTask(longer_is_better)
        log = self.optimise(
            task, append_x, ["t1", "t2"], ["p1", "p2", "p3"], budget=6, minibatch_size=2
        )
        self.assertEqual(task.calls_used, 3)
        self.assertEqual(log["iterations"], [])  # 3 left is under 2 * minibatch_size
        self.assertEqual(log["best"], 0)

    def test_budget_below_pareto_size_is_refused(self):
        with self.assertRaises(ValueError):
            self.optimise(
                FakeTask(longer_is_better), append_x, ["t1"], ["p1", "p2"], budget=1
            )

    def test_child_is_scored_when_exactly_len_pareto_is_left(self):
        # Seed 3, parent 2, child 2 = 7 of 10: exactly 3 left, enough to score the child.
        task = FakeTask(longer_is_better)
        log = self.optimise(
            task,
            append_x,
            ["t1", "t2"],
            ["p1", "p2", "p3"],
            budget=10,
            minibatch_size=2,
        )
        self.assertEqual([i["outcome"] for i in log["iterations"]], ["accepted"])
        self.assertEqual(task.calls_used, 10)
        self.assertEqual(len(log["candidates"]), 2)

    def test_run_stops_at_out_of_budget_even_with_minibatch_budget_left(self):
        # Seed 5, parent 1, child 1 = 7 of 11: 4 left is under len(pareto) but over 2 * minibatch_size.
        task = FakeTask(longer_is_better)
        log = self.optimise(
            task, append_x, ["t1", "t2"], ["p1"] * 5, budget=11, minibatch_size=1
        )
        self.assertEqual([i["outcome"] for i in log["iterations"]], ["out of budget"])
        self.assertEqual(task.calls_used, 7)

    def test_wrong_rollout_count_is_refused(self):
        class Extra(FakeTask):
            def evaluate(self, candidate, examples):
                return super().evaluate(candidate, list(examples) * 2)

        with self.assertRaises(ValueError):
            self.optimise(Extra(longer_is_better), append_x, ["t1"], ["p1"], budget=9)

    def test_no_rollouts_is_refused_not_treated_as_perfect(self):
        class Empty(FakeTask):
            def evaluate(self, candidate, examples):
                return (
                    [] if len(examples) == 2 else super().evaluate(candidate, examples)
                )

        with self.assertRaises(ValueError):
            self.optimise(
                Empty(longer_is_better),
                append_x,
                ["t1", "t2"],
                ["p1"],
                budget=9,
                minibatch_size=2,
            )

    def test_nan_score_is_refused(self):
        task = FakeTask(lambda c, e: math.nan if c else 0.5)
        with self.assertRaises(ValueError):
            self.optimise(
                task, append_x, ["t1", "t2"], ["p1", "p2"], budget=11, minibatch_size=2
            )

    def test_pareto_selection_is_a_hook(self):
        task = FakeTask(longer_is_better)
        settings = Settings(budget=20, minibatch_size=2, selection="pareto")
        with self.assertRaises(NotImplementedError):
            optimise("", ["t1", "t2"], ["p1"], task, append_x, settings, self.log_path)

    def test_equal_child_is_rejected_and_never_scored_on_pareto(self):
        train = ["t1", "t2", "t3", "t4"]
        pareto = ["p1", "p2"]
        task = FakeTask(lambda candidate, example: 0.5)
        log = self.optimise(task, append_x, train, pareto, budget=30, minibatch_size=2)
        self.assertTrue(log["iterations"])
        self.assertEqual({i["outcome"] for i in log["iterations"]}, {"rejected"})
        self.assertEqual([c["text"] for c in log["candidates"]], [""])
        for candidate, examples in task.calls:
            if candidate != "":
                self.assertTrue(set(examples) <= set(train), examples)

    def test_equal_sums_are_rejected_despite_float_rounding(self):
        # Real gepa_scores: 0.6 = (1+1+0+0.4)/4 and 0.2 = (0+0+0+0.8)/4. The sums are equal,
        # but in floats 0.2 + 0.2 + 0.2 is 0.6000000000000001.
        by_candidate = {"": [0.0, 0.6, 0.0], "x": [0.2, 0.2, 0.2]}

        def score(candidate, example):
            return (
                by_candidate[candidate][int(example[1])] if example[0] == "t" else 0.0
            )

        log = self.optimise(
            FakeTask(score),
            append_x,
            ["t0", "t1", "t2"],
            ["p1"],
            budget=8,
            minibatch_size=3,
        )
        self.assertEqual(log["iterations"][0]["outcome"], "rejected")

    def test_reflector_sees_every_minibatch_rollout(self):
        seen = []

        def propose(parent, rollouts):
            seen.append([r.record["example"] for r in rollouts])
            return parent + "x"

        train = ["t1", "t2", "t3"]
        log = self.optimise(
            FakeTask(longer_is_better),
            propose,
            train,
            ["p1"],
            budget=20,
            minibatch_size=3,
        )
        self.assertTrue(seen)
        for iteration, examples in zip(log["iterations"], seen, strict=True):
            self.assertEqual(examples, [train[i] for i in iteration["minibatch"]])

    def test_greedy_always_picks_the_top_mean(self):
        train = ["alpha", "beta", "gamma", "delta"]
        pareto = ["alpha", "gamma", "delta", "!beta", "!beta", "!beta"]
        task = FakeTask(keyword_score)
        log = self.optimise(
            task, add_keyword, train, pareto, budget=120, minibatch_size=2
        )
        candidates = log["candidates"]
        went_back = False
        for n, iteration in enumerate(log["iterations"]):
            pool = [
                c for c in candidates if c["iteration"] is None or c["iteration"] < n
            ]
            expected = max(pool, key=lambda c: c["mean_pareto_score"])
            self.assertEqual(iteration["parent"], expected["id"], f"iteration {n}")
            went_back |= iteration["parent"] != pool[-1]["id"]
        # Adding beta wins the minibatch but loses on pareto, so greedy must leave the newest candidate.
        self.assertTrue(went_back)

    def test_final_pick_is_earliest_on_tie(self):
        def score(candidate, example):
            return (
                1.0 if example.startswith("p") else longer_is_better(candidate, example)
            )

        task = FakeTask(score)
        log = self.optimise(
            task, append_x, ["t1", "t2"], ["p1", "p2"], budget=30, minibatch_size=2
        )
        self.assertGreater(len(log["candidates"]), 1)
        self.assertTrue(all(c["mean_pareto_score"] == 1 for c in log["candidates"]))
        self.assertEqual(log["best"], 0)

    def test_perfect_parent_skips_reflection(self):
        task = FakeTask(lambda candidate, example: 1.0)
        propose = CountingPropose(append_x)
        log = self.optimise(
            task, propose, ["t1", "t2"], ["p1"], budget=9, minibatch_size=2
        )
        self.assertEqual(propose.calls, 0)
        self.assertEqual([i["outcome"] for i in log["iterations"]], ["skipped"] * 3)
        self.assertEqual(
            [i["calls_used"] for i in log["iterations"]], [3, 5, 7]
        )  # 2 left is under 4

    def test_proposal_failure_is_logged_and_loop_continues(self):
        def flaky(parent, rollouts):
            if not log_calls:
                log_calls.append(1)
                raise ValueError("boom")
            return add_keyword(parent, rollouts)

        log_calls = []
        task = FakeTask(keyword_score)
        train = ["alpha", "beta", "gamma", "delta"]
        log = self.optimise(
            task, flaky, train, ["alpha", "beta"], budget=20, minibatch_size=2
        )
        first = log["iterations"][0]
        self.assertEqual(first["outcome"], "proposal failed")
        self.assertEqual(first["error"], "ValueError: boom")
        self.assertIsNone(first["child_text"])
        self.assertEqual(first["calls_used"], 2 + 2)
        self.assertGreater(len(log["iterations"]), 1)
        self.assertEqual(log["iterations"][1]["outcome"], "accepted")

    def test_reflection_prompt_and_reply_are_logged(self):
        train = ["alpha", "beta", "gamma", "delta"]
        task = FakeTask(keyword_score)
        log = self.optimise(
            task,
            RecordingPropose(fail_first=True),
            train,
            ["alpha"],
            budget=12,
            minibatch_size=2,
        )
        failed, succeeded = log["iterations"][:2]
        self.assertEqual(failed["outcome"], "proposal failed")
        self.assertEqual(failed["reflection_prompt"], "prompt for ''")
        self.assertIsNone(failed["reflection_reply"])
        self.assertEqual(succeeded["outcome"], "accepted")
        self.assertEqual(succeeded["reflection_prompt"], "prompt for ''")
        self.assertEqual(succeeded["reflection_reply"], "reply for ''")

    def test_propose_without_the_attributes_logs_none(self):
        log = self.optimise(
            FakeTask(keyword_score),
            add_keyword,
            ["alpha", "beta"],
            ["alpha"],
            budget=8,
            minibatch_size=2,
        )
        first = log["iterations"][0]
        self.assertEqual(first["outcome"], "accepted")
        self.assertIsNone(first["reflection_prompt"])
        self.assertIsNone(first["reflection_reply"])

    def test_same_seed_gives_identical_log(self):
        train = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        pareto = ["alpha", "beta", "gamma"]
        logs = [
            self.optimise(
                FakeTask(keyword_score),
                add_keyword,
                train,
                pareto,
                budget=60,
                minibatch_size=2,
                seed=1,
            )
            for _ in range(2)
        ]
        for log in logs:
            del log["metadata"]["started"], log["metadata"]["runtime_seconds"]
        self.assertEqual(logs[0], logs[1])

    def test_different_seeds_give_different_minibatch_orders(self):
        train = [f"t{i}" for i in range(10)]
        orders = []
        for seed in (0, 1):
            log = self.optimise(
                FakeTask(lambda c, e: 0.5),
                append_x,
                train,
                ["p1"],
                budget=41,
                minibatch_size=2,
                seed=seed,
            )
            orders.append([i["minibatch"] for i in log["iterations"]])
        self.assertNotEqual(orders[0], orders[1])
        # Every train position appears once per epoch (10 positions, 5 minibatches of 2).
        first_epoch = [p for batch in orders[0][:5] for p in batch]
        self.assertEqual(sorted(first_epoch), list(range(10)))

    def test_minibatch_order_is_reshuffled_each_epoch(self):
        train = [f"t{i}" for i in range(6)]
        log = self.optimise(
            FakeTask(lambda c, e: 0.5),
            append_x,
            train,
            ["p1"],
            budget=100,
            minibatch_size=2,
            seed=3,
        )
        batches = [i["minibatch"] for i in log["iterations"]]
        epochs = [batches[i : i + 3] for i in range(0, 12, 3)]
        for epoch in epochs:
            self.assertEqual(sorted(p for b in epoch for p in b), list(range(6)))
        self.assertGreater(len({tuple(map(tuple, e)) for e in epochs}), 1)

    def test_log_on_disk_has_every_finished_iteration_when_the_task_crashes(self):
        class Crashing(FakeTask):
            def evaluate(self, candidate, examples):
                if len(self.calls) == 6:
                    raise RuntimeError("API down")
                return super().evaluate(candidate, examples)

        with self.assertRaises(RuntimeError):
            self.optimise(
                Crashing(longer_is_better),
                append_x,
                ["t1", "t2"],
                ["p1"],
                budget=50,
                minibatch_size=1,
            )
        on_disk = json.loads(self.log_path.read_text())
        self.assertEqual([i["outcome"] for i in on_disk["iterations"]], ["accepted"])
        self.assertEqual(on_disk["best"], 1)

    def test_log_file_survives_a_crash_mid_write(self):
        real = Path.write_text
        writes = []

        def dies_on_fourth_write(path, data, *args, **kwargs):
            writes.append(1)
            if len(writes) == 4:
                real(path, data[: len(data) // 2])
                raise OSError("killed mid-write")
            return real(path, data, *args, **kwargs)

        with (
            patch.object(Path, "write_text", dies_on_fourth_write),
            self.assertRaises(OSError),
        ):
            self.optimise(
                FakeTask(longer_is_better),
                append_x,
                ["t1", "t2"],
                ["p1"],
                budget=50,
                minibatch_size=1,
            )
        # Writes 1 to 3 are the seed and two iterations; the half-written fourth never replaced it.
        self.assertEqual(len(json.loads(self.log_path.read_text())["iterations"]), 2)

    def test_log_is_consistent_and_on_disk(self):
        train = ["alpha", "beta", "gamma", "delta"]
        pareto = ["alpha", "beta", "gamma", "!delta"]
        log = self.optimise(
            FakeTask(keyword_score),
            add_keyword,
            train,
            pareto,
            budget=80,
            minibatch_size=2,
        )
        self.assertEqual(json.loads(self.log_path.read_text()), log)
        candidates = log["candidates"]
        ids = {c["id"] for c in candidates}
        for c in candidates[1:]:
            self.assertIn(c["parent"], ids)
            self.assertEqual(log["iterations"][c["iteration"]]["child"], c["id"])
        for j in range(len(pareto)):
            top = max(c["pareto_scores"][j] for c in candidates)
            winners = [c["id"] for c in candidates if c["pareto_scores"][j] == top]
            self.assertEqual(
                winners, [c["id"] for c in candidates if j in c["pareto_wins"]]
            )
        best = max(candidates, key=lambda c: c["mean_pareto_score"])
        self.assertEqual(log["best"], best["id"])
        self.assertEqual(log["metadata"]["settings"]["selection"], "best")
        self.assertEqual(log["metadata"]["task"], {"task": "fake"})
        self.assertEqual(
            (log["metadata"]["train_size"], log["metadata"]["pareto_size"]), (4, 4)
        )

    def test_planted_answer_is_found_with_best_selection(self):
        keywords = ["alpha", "beta", "gamma", "delta"]
        log = self.optimise(
            FakeTask(keyword_score),
            add_keyword,
            keywords,
            keywords,
            budget=100,
            minibatch_size=2,
        )
        best = log["candidates"][log["best"]]
        self.assertEqual(best["mean_pareto_score"], 1)
        # Each epoch (2 minibatches) shows every keyword, so at least one is added per epoch:
        # 4 keywords need at most 4 epochs of 2 iterations.
        self.assertLess(best["iteration"], 8)


if __name__ == "__main__":
    unittest.main()
