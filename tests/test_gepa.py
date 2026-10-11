import json
import math
import random
import re
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from unittest.mock import patch

from gepa_toy.gepa import GEPA, Candidate, CandidatePool, Rollout, shuffled_positions
from gepa_toy.model import ModelConfig

GIT = {"commit": "test", "dirty": False, "changes": []}
TEMPLATE = "{instruction}\n---\n{rollouts}"
REFLECTION_MODEL = ModelConfig(model="fake/reflector", temperature=1.0, reasoning=True)


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
                feedback=f"missing {e}" if self.score(candidate, e) < 1 else f"ok {e}",
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
    minibatch can hurt on pareto. A "beta|gamma" example scores 1 when either word is
    present, so two lines can win the same example with different words.
    """
    words = candidate.split()
    if example.startswith("!"):
        return 0.0 if example[1:] in words else 1.0
    return 1.0 if any(word in words for word in example.split("|")) else 0.0


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


def fake_reflection(propose):
    """A stand-in for model.complete: parse the prompt back into the parent text and the
    rollouts, call propose(parent, rollouts) and put the result in a fenced block.

    Rollout scores are not in the prompt, so they are read from the feedback ("missing
    ..." means a score below 1). A result starting "NOFENCE:" is returned as the bare reply.
    """

    def complete(prompt, model_config):
        parent, examples = prompt.split("\n---\n", 1)
        rollouts = []
        for block in re.split(r"\n\n(?=Example \d+\n)", examples):
            _, trace, feedback = block.split("\n")
            score = 0.0 if feedback.startswith("missing") else 1.0
            rollouts.append(Rollout(score, trace, feedback, {}))
        result = propose(parent, rollouts)
        if result.startswith("NOFENCE:"):
            return result.removeprefix("NOFENCE:")
        return f"Thinking...\n```\n{result}\n```\n"

    return complete


class GepaTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log_path = Path(tmp.name) / "run.json"
        patcher = patch("gepa_toy.gepa.git_version", return_value=GIT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_gepa(self, task, **settings):
        settings.setdefault("selection", "best")
        return GEPA(task, REFLECTION_MODEL, TEMPLATE, self.log_path, **settings)

    def optimise(self, task, propose, train, pareto, **settings):
        """Run GEPA with the fake reflection model; return the log. Keeps the GEPA as self.gepa."""
        self.gepa = self.make_gepa(task, **settings)
        with patch("gepa_toy.model.complete", fake_reflection(propose)):
            self.best = self.gepa.run("", train, pareto)
        return self.gepa.log

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

    def test_planted_answer_pareto_keeps_two_lines_and_best_follows_one(self):
        # "gamma" helps on train and the second pareto example but loses "!gamma", so a
        # gamma line and a gamma-free line win different examples. Only the gamma-free
        # line can reach a mean of 1. Pareto sampling makes the iteration count
        # seed-dependent, so only greedy gets a bound (at most 4 over 40 seeds).
        train = ["alpha", "beta", "gamma"]
        pareto = ["alpha", "beta|gamma", "!gamma"]
        gamma_parents = {"pareto": set(), "best": set()}
        for selection, seeds_with_gamma_parent in gamma_parents.items():
            for seed in range(16):
                with self.subTest(selection=selection, seed=seed):
                    log = self.optimise(
                        FakeTask(keyword_score),
                        add_keyword,
                        train,
                        pareto,
                        budget=150,
                        minibatch_size=2,
                        selection=selection,
                        seed=seed,
                    )
                    candidates = log["candidates"]
                    best = candidates[log["best"]]
                    self.assertEqual(best["mean_pareto_score"], 1)
                    if selection == "best":
                        self.assertLess(best["iteration"], 8)
                    for n, iteration in enumerate(log["iterations"]):
                        parent = candidates[iteration["parent"]]
                        if "gamma" in parent["text"].split():
                            seeds_with_gamma_parent.add(seed)
                        if selection == "best":
                            pool = [
                                c
                                for c in candidates
                                if c["iteration"] is None or c["iteration"] < n
                            ]
                            expected = max(pool, key=lambda c: c["mean_pareto_score"])
                            self.assertEqual(parent["id"], expected["id"])
                    self.assertTrue(
                        any("gamma" in c["text"].split() for c in candidates)
                    )
        # A gamma candidate never beats the gamma-free line's mean and, being created
        # later, loses ties, so greedy never comes back to it.
        self.assertEqual(gamma_parents["best"], set())
        self.assertGreater(len(gamma_parents["pareto"]), 1)

    def test_same_seed_gives_identical_log_with_pareto_selection(self):
        train = ["alpha", "beta", "gamma"]
        pareto = ["alpha", "beta|gamma", "!gamma"]
        logs = []
        for _ in range(2):
            log = self.optimise(
                FakeTask(keyword_score),
                add_keyword,
                train,
                pareto,
                budget=100,
                minibatch_size=2,
                selection="pareto",
                seed=4,
            )
            del log["metadata"]["started"], log["metadata"]["runtime_seconds"]
            logs.append(log)
        self.assertEqual(logs[0], logs[1])
        self.assertGreater(len({i["parent"] for i in logs[0]["iterations"]}), 1)

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
            seen.append([r.trace for r in rollouts])
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
        self.assertEqual(
            first["reflection_prompt"],
            "\n---\nExample 1\nalpha\nmissing alpha\n\nExample 2\nbeta\nmissing beta",
        )
        self.assertEqual(first["reflection_reply"], "Thinking...\n```\nalpha\n```\n")

    def test_skipped_iteration_logs_no_reflection(self):
        log = self.optimise(
            FakeTask(lambda candidate, example: 1.0),
            append_x,
            ["t1", "t2"],
            ["p1"],
            budget=5,
            minibatch_size=2,
        )
        first = log["iterations"][0]
        self.assertEqual(first["outcome"], "skipped")
        self.assertIsNone(first["reflection_prompt"])
        self.assertIsNone(first["reflection_reply"])

    def test_failed_reflection_call_logs_the_prompt_and_no_reply(self):
        def failing(parent, rollouts):
            raise RuntimeError("api down")

        log = self.optimise(
            FakeTask(keyword_score),
            failing,
            ["alpha", "beta", "gamma"],
            ["alpha"],
            budget=12,
            minibatch_size=2,
        )
        self.assertGreater(len(log["iterations"]), 1)
        for iteration in log["iterations"]:
            self.assertEqual(iteration["outcome"], "proposal failed")
            self.assertEqual(iteration["error"], "RuntimeError: api down")
            self.assertTrue(
                iteration["reflection_prompt"].startswith("\n---\nExample 1")
            )
            self.assertIsNone(iteration["reflection_reply"])

    def test_reply_without_a_fenced_block_logs_prompt_and_reply(self):
        log = self.optimise(
            FakeTask(keyword_score),
            lambda parent, rollouts: "NOFENCE:just say alpha",
            ["alpha", "beta"],
            ["alpha"],
            budget=8,
            minibatch_size=2,
        )
        first = log["iterations"][0]
        self.assertEqual(first["outcome"], "proposal failed")
        self.assertTrue(first["error"].startswith("ValueError: No fenced code block"))
        self.assertIn("Example 1", first["reflection_prompt"])
        self.assertEqual(first["reflection_reply"], "just say alpha")

    def test_reflection_model_gets_the_configured_model(self):
        complete = fake_reflection(add_keyword)
        calls = []

        def recording(prompt, model_config):
            calls.append(model_config)
            return complete(prompt, model_config)

        gepa = self.make_gepa(FakeTask(keyword_score), budget=8, minibatch_size=2)
        with patch("gepa_toy.model.complete", recording):
            gepa.run("", ["alpha", "beta"], ["alpha"])
        self.assertEqual(calls, [REFLECTION_MODEL] * len(calls))
        self.assertTrue(calls)

    def test_run_returns_the_best_candidate(self):
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
        top = max(c["mean_pareto_score"] for c in log["candidates"])
        self.assertEqual(self.best.mean_pareto_score, top)
        self.assertEqual(self.best.id, log["best"])
        self.assertIs(self.best, self.gepa.pool.best())
        self.assertGreater(len(log["candidates"]), 1)

    def test_run_twice_gives_two_runs_and_the_second_overwrites_the_file(self):
        train = ["alpha", "beta", "gamma", "delta"]
        pareto = ["alpha", "beta|gamma", "!gamma"]
        task = FakeTask(keyword_score)
        gepa = self.make_gepa(task, budget=100, minibatch_size=2, seed=4)
        logs, pools = [], []
        with patch("gepa_toy.model.complete", fake_reflection(add_keyword)):
            for _ in range(2):
                gepa.run("", train, pareto)
                logs.append(gepa.log)
                pools.append(gepa.pool)
        self.assertIsNot(logs[0], logs[1])
        self.assertIsNot(pools[0], pools[1])
        # The budget restarts with each run.
        self.assertEqual(task.calls_used, 2 * logs[0]["iterations"][-1]["calls_used"])
        for log in logs:
            del log["metadata"]["started"], log["metadata"]["runtime_seconds"]
        self.assertEqual(logs[0], logs[1])
        on_disk = json.loads(self.log_path.read_text())
        del on_disk["metadata"]["started"], on_disk["metadata"]["runtime_seconds"]
        self.assertEqual(on_disk, logs[1])

    def test_log_metadata_has_the_reflection_model_and_full_template(self):
        log = self.optimise(
            FakeTask(keyword_score),
            add_keyword,
            ["alpha", "beta"],
            ["alpha"],
            budget=8,
            minibatch_size=2,
            seed=5,
        )
        metadata = log["metadata"]
        self.assertEqual(metadata["reflection_model"], REFLECTION_MODEL.model_dump())
        self.assertEqual(metadata["reflection_prompt"], TEMPLATE)
        self.assertEqual(
            metadata["settings"],
            {"budget": 8, "minibatch_size": 2, "selection": "best", "seed": 5},
        )
        self.assertNotIn("seed", metadata)
        self.assertNotIn("stage", metadata)

    def test_log_file_equals_the_log_on_the_instance(self):
        log = self.optimise(
            FakeTask(keyword_score),
            add_keyword,
            ["alpha", "beta"],
            ["alpha"],
            budget=8,
            minibatch_size=2,
        )
        self.assertEqual(json.loads(self.log_path.read_text()), log)

    def test_candidates_are_logged_with_the_same_keys_in_order(self):
        log = self.optimise(
            FakeTask(keyword_score),
            add_keyword,
            ["alpha", "beta"],
            ["alpha"],
            budget=8,
            minibatch_size=2,
        )
        self.assertEqual(
            list(log["candidates"][0]),
            [
                "id",
                "text",
                "parent",
                "iteration",
                "pareto_scores",
                "mean_pareto_score",
            ],
        )

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

    def test_parent_is_drawn_before_the_minibatch(self):
        # rng.choices draws even when the front has one candidate and shuffled_positions is
        # lazy, so swapping the two calls changes every minibatch of every run.
        rng = random.Random(3)
        rng.choices([0], weights=[1])
        expected = list(islice(shuffled_positions(6, rng), 2))
        log = self.optimise(
            FakeTask(lambda c, e: 0.5),
            append_x,
            [f"t{i}" for i in range(6)],
            ["p1"],
            budget=10,
            minibatch_size=2,
            selection="pareto",
            seed=3,
        )
        self.assertEqual(log["iterations"][0]["minibatch"], expected)

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


def candidates_from(rows):
    """A pool with one candidate per row of pareto scores, ids in row order."""
    pool = CandidatePool()
    for row in rows:
        pool.add(str(len(pool.candidates)), None, None, row)
    return pool


def draws(pool, n=3000, seed=0):
    """How often select_parent picks each id over n pareto selections."""
    rng = random.Random(seed)
    return Counter(pool.select_parent("pareto", rng).id for _ in range(n))


class ParetoSelectionTest(unittest.TestCase):
    def test_add_assigns_ids_and_stores_the_mean(self):
        pool = CandidatePool()
        seed = pool.add("seed", None, None, [1, 0, 0.5])
        child = pool.add("child", 0, 3, [0, 0, 0])
        self.assertEqual((seed.id, child.id), (0, 1))
        self.assertEqual((child.parent, child.iteration), (0, 3))
        self.assertEqual(seed.mean_pareto_score, 0.5)
        self.assertEqual(list(asdict(seed)), list(asdict(child)))

    def test_wins_follow_later_candidates(self):
        pool = candidates_from([[1, 0, 0.5]])
        self.assertEqual(pool.wins(), {0: {0, 1, 2}})
        # The child ties the seed on example 1 and beats it on example 2.
        pool.add("child", 0, 0, [0, 0, 1])
        self.assertEqual(pool.wins(), {0: {0, 1}, 1: {1, 2}})

    def test_mean_is_derived_from_the_scores_and_cannot_be_passed(self):
        candidate = Candidate(0, "x", None, None, [1, 0, 0.5])
        self.assertEqual(candidate.mean_pareto_score, 0.5)
        with self.assertRaises(TypeError):
            Candidate(0, "x", None, None, [1], mean_pareto_score=1.0)

    def test_best_selection_takes_the_top_mean_and_leaves_the_rng_alone(self):
        pool = candidates_from([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
        rng = random.Random(0)
        self.assertIs(pool.select_parent("best", rng), pool.candidates[1])
        self.assertEqual(rng.random(), random.Random(0).random())

    def test_winners_per_example_include_ties(self):
        pool = candidates_from([[1, 0, 0.5], [1, 1, 0], [0, 0, 1]])
        self.assertEqual(pool.wins(), {0: {0}, 1: {0, 1}, 2: {2}})

    def test_weights_equal_win_counts_and_sampling_matches(self):
        pool = candidates_from([[1, 0, 0.5], [1, 1, 0], [0, 0, 1]])
        front = pool.pareto_front()
        self.assertEqual([c.id for c in front], [1, 2])
        self.assertEqual([len(pool.wins()[c.id]) for c in front], [2, 1])
        counts = draws(pool, n=3000)
        self.assertEqual(set(counts), {1, 2})
        self.assertAlmostEqual(counts[1] / 3000, 2 / 3, delta=0.05)
        self.assertAlmostEqual(counts[2] / 3000, 1 / 3, delta=0.05)

    def test_candidate_that_wins_nothing_is_never_picked(self):
        pool = candidates_from([[1, 0.5], [0.5, 0]])
        self.assertEqual(pool.wins()[1], set())
        self.assertEqual(set(draws(pool)), {0})

    def test_subset_win_set_is_never_picked(self):
        pool = candidates_from([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        self.assertEqual(pool.wins()[0], {0})
        self.assertNotIn(0, draws(pool))

    def test_same_win_set_only_the_higher_mean_can_be_picked(self):
        for rows, weaker in (
            ([[1, 1, 0], [1, 1, 0.5], [0, 0, 1]], 0),
            ([[1, 1, 0.5], [1, 1, 0], [0, 0, 1]], 1),
        ):
            with self.subTest(weaker=weaker):
                pool = candidates_from(rows)
                self.assertEqual(pool.wins()[0], pool.wins()[1])
                counts = draws(pool)
                self.assertNotIn(weaker, counts)
                self.assertEqual(set(counts), {1 - weaker, 2})

    def test_wins_include_ties_that_differ_only_by_float_rounding(self):
        # Real gepa_scores for the same result can come out one ulp apart.
        tie = [0.7, math.nextafter(0.7, 1)]
        pool = candidates_from([[tie[0], 1], [tie[1], 0]])
        self.assertEqual(pool.wins(), {0: {0, 1}, 1: {0}})

    def test_equal_mean_and_identical_wins_keeps_the_later_candidate(self):
        pool = candidates_from([[1, 1, 0], [1, 1, 0]])
        self.assertEqual([c.id for c in pool.pareto_front()], [1])
        self.assertIs(pool.best(), pool.candidates[0])

    def test_unique_win_survives_whatever_its_mean(self):
        pool = candidates_from([[1, 1, 1, 0], [0, 0, 0, 1]])
        counts = draws(pool)
        self.assertEqual(set(counts), {0, 1})
        self.assertAlmostEqual(counts[1] / 3000, 1 / 4, delta=0.05)

    def test_wins_covered_by_several_candidates_together_are_dominated(self):
        # Candidate 2 has the lowest mean and no single candidate covers its wins {0, 1},
        # but 0 and 1 together do, so the reference rule drops it.
        pool = candidates_from([[1, 0, 1, 0.2], [0, 1, 0.2, 1], [1, 1, 0, 0]])
        self.assertEqual(pool.wins()[2], {0, 1})
        self.assertEqual([c.id for c in pool.pareto_front()], [0, 1])

    def test_a_dropped_candidate_no_longer_covers_later_ones(self):
        # Ascending by mean: 0 is dropped first (covered by 1 and 2), after which 1 is the
        # only winner of example 0 left and must stay. Dropping both would leave
        # example 0 with no winner in the front.
        pool = candidates_from([[1, 0.5, 0], [1, 0, 1], [0, 1, 1]])
        self.assertEqual([c.id for c in pool.pareto_front()], [1, 2])

    def test_dominated_candidate_with_highest_mean_is_still_the_final_pick(self):
        pool = candidates_from([[1, 0.9, 0.9], [1, 1, 0], [0, 0, 1]])
        self.assertIs(pool.best(), pool.candidates[0])
        self.assertNotIn(0, draws(pool))

    def test_same_seed_gives_the_same_picks(self):
        pool = candidates_from([[1, 0, 0.5], [1, 1, 0], [0, 0, 1]])
        picks = []
        for _ in range(2):
            rng = random.Random(7)
            picks.append([pool.select_parent("pareto", rng).id for _ in range(50)])
        self.assertEqual(picks[0], picks[1])
        self.assertEqual(set(picks[0]), {1, 2})


if __name__ == "__main__":
    unittest.main()
