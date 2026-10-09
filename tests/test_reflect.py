import unittest

from gepa_toy.gepa import Rollout
from gepa_toy.reflect import (
    DSPY_REFLECTION_PROMPT,
    REFLECTION_PROMPT,
    extract_instruction,
    make_reflector,
)


def rollout(trace: str, feedback: str) -> Rollout:
    return Rollout(score=0.0, trace=trace, feedback=feedback, record={})


class FakeComplete:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


ROLLOUTS = [
    rollout("TRACE-ONE", "FEEDBACK-ONE"),
    rollout("TRACE-TWO", "FEEDBACK-TWO"),
    rollout("TRACE-THREE", "FEEDBACK-THREE"),
]


class PromptTest(unittest.TestCase):
    def test_prompt_has_instruction_then_every_trace_and_feedback_in_order(self):
        fake = FakeComplete("```\nnew\n```")
        make_reflector(fake)("THE-INSTRUCTION", ROLLOUTS)
        prompt = fake.prompts[0]
        expected = ["THE-INSTRUCTION"]
        for r in ROLLOUTS:
            expected += [r.trace, r.feedback]
        positions = [prompt.index(piece) for piece in expected]
        self.assertEqual(positions, sorted(positions))
        for piece in expected:
            self.assertEqual(prompt.count(piece), 1, piece)

    def test_braces_and_placeholders_in_inputs_appear_verbatim(self):
        instruction = 'Reply as {"queue": "x"} or {rollouts} and never }{'
        trace = "Body mentions {instruction} and {rollouts}"
        fake = FakeComplete("```\nnew\n```")
        make_reflector(fake)(instruction, [rollout(trace, "Score: 0")])
        prompt = fake.prompts[0]
        self.assertEqual(prompt.count(instruction), 1)
        self.assertEqual(prompt.count(trace), 1)

    def test_backslashes_in_inputs_appear_verbatim(self):
        instruction = r"Write \n for newlines, never \1 or \g<0>"
        fake = FakeComplete("```\nnew\n```")
        make_reflector(fake)(instruction, [rollout(r"C:\\temp \2", "Score: 0")])
        self.assertIn(instruction, fake.prompts[0])
        self.assertIn(r"C:\\temp \2", fake.prompts[0])

    def test_custom_template_is_used(self):
        fake = FakeComplete("```\nnew\n```")
        make_reflector(fake, "A {instruction} B {rollouts} C")("INS", ROLLOUTS)
        self.assertTrue(fake.prompts[0].startswith("A INS B Example 1\n"))
        self.assertTrue(fake.prompts[0].endswith("FEEDBACK-THREE C"))

    def test_both_templates_contain_both_placeholders(self):
        for template in (REFLECTION_PROMPT, DSPY_REFLECTION_PROMPT):
            self.assertIn("{instruction}", template)
            self.assertIn("{rollouts}", template)


class ExtractionTest(unittest.TestCase):
    def propose(self, reply: str) -> str:
        return make_reflector(FakeComplete(reply))("parent", ROLLOUTS)

    def test_returns_last_of_several_blocks_without_tag(self):
        reply = "Old:\n```\nfirst\n```\nNew:\n```\nsecond\n```\nDone."
        self.assertEqual(self.propose(reply), "second")

    def test_returns_last_of_several_blocks_with_tag(self):
        reply = "```text\nfirst\n```\n\n```markdown\nsecond\n```"
        self.assertEqual(self.propose(reply), "second")

    def test_tag_with_spaces_is_allowed(self):
        self.assertEqual(self.propose("``` python \nnew\n```"), "new")

    def test_fence_mentioned_mid_sentence_is_not_a_fence(self):
        reply = "I'll put it in a ``` block:\n```\nreal\n```"
        self.assertEqual(self.propose(reply), "real")

    def test_indented_fences_are_accepted(self):
        self.assertEqual(self.propose("  ```\n  body\n  ```"), "  body")

    def test_closing_fence_must_start_a_line(self):
        with self.assertRaises(ValueError):
            self.propose("```\nbody```")

    def test_leading_blank_line_inside_block_is_kept(self):
        self.assertEqual(self.propose("```\n\nbody\n```"), "\nbody")

    def test_inner_newlines_and_blank_lines_are_preserved(self):
        reply = "```\nline one\n\n  indented two\nline three\n```"
        self.assertEqual(self.propose(reply), "line one\n\n  indented two\nline three")

    def test_only_one_trailing_newline_is_removed(self):
        self.assertEqual(self.propose("```\nbody\n\n```"), "body\n")

    def test_no_fenced_block_raises_with_start_of_reply(self):
        with self.assertRaises(ValueError) as cm:
            self.propose("Here is the instruction without a fence.")
        self.assertIn("Here is the instruction", str(cm.exception))

    def test_unclosed_fence_raises(self):
        with self.assertRaises(ValueError):
            self.propose("```\nnever closed")

    def test_exception_from_complete_propagates(self):
        def failing(prompt: str) -> str:
            raise RuntimeError("api down")

        with self.assertRaises(RuntimeError):
            make_reflector(failing)("parent", ROLLOUTS)


class RecordingTest(unittest.TestCase):
    def test_last_prompt_and_reply_are_recorded(self):
        fake = FakeComplete("```\nnew\n```")
        propose = make_reflector(fake)
        self.assertIsNone(propose.last_prompt)
        self.assertIsNone(propose.last_reply)
        propose("parent", ROLLOUTS)
        self.assertEqual(propose.last_prompt, fake.prompts[0])
        self.assertEqual(propose.last_reply, "```\nnew\n```")

    def test_failed_extraction_still_records_prompt_and_reply(self):
        propose = make_reflector(FakeComplete("no fence"))
        with self.assertRaises(ValueError):
            propose("parent", ROLLOUTS)
        self.assertIn("parent", propose.last_prompt)
        self.assertEqual(propose.last_reply, "no fence")

    def test_failed_call_clears_stale_reply(self):
        calls = iter(["```\nfirst\n```", RuntimeError("api down")])

        def complete(prompt: str) -> str:
            result = next(calls)
            if isinstance(result, Exception):
                raise result
            return result

        propose = make_reflector(complete)
        propose("parent", ROLLOUTS)
        with self.assertRaises(RuntimeError):
            propose("second-parent", ROLLOUTS)
        self.assertIn("second-parent", propose.last_prompt)
        self.assertIsNone(propose.last_reply)

    def test_failed_prompt_build_clears_both(self):
        propose = make_reflector(FakeComplete("```\nnew\n```"))
        propose("parent", ROLLOUTS)
        with self.assertRaises(AttributeError):
            propose("parent", [None])
        self.assertIsNone(propose.last_prompt)
        self.assertIsNone(propose.last_reply)


if __name__ == "__main__":
    unittest.main()


class EmptyBlockTest(unittest.TestCase):
    def test_empty_fenced_block_raises(self):
        with self.assertRaises(ValueError):
            extract_instruction("Here:\n```\n\n```\n")
