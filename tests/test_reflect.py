import unittest

from gepa_toy.gepa import Rollout, build_reflection_prompt, extract_instruction
from gepa_toy.reflect import DSPY_REFLECTION_PROMPT, REFLECTION_PROMPT


def rollout(trace: str, feedback: str) -> Rollout:
    return Rollout(score=0.0, trace=trace, feedback=feedback, record={})


ROLLOUTS = [
    rollout("TRACE-ONE", "FEEDBACK-ONE"),
    rollout("TRACE-TWO", "FEEDBACK-TWO"),
    rollout("TRACE-THREE", "FEEDBACK-THREE"),
]


class PromptTest(unittest.TestCase):
    def test_prompt_has_instruction_then_every_trace_and_feedback_in_order(self):
        prompt = build_reflection_prompt(REFLECTION_PROMPT, "THE-INSTRUCTION", ROLLOUTS)
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
        prompt = build_reflection_prompt(
            REFLECTION_PROMPT, instruction, [rollout(trace, "Score: 0")]
        )
        self.assertEqual(prompt.count(instruction), 1)
        self.assertEqual(prompt.count(trace), 1)

    def test_backslashes_in_inputs_appear_verbatim(self):
        instruction = r"Write \n for newlines, never \1 or \g<0>"
        prompt = build_reflection_prompt(
            REFLECTION_PROMPT, instruction, [rollout(r"C:\\temp \2", "Score: 0")]
        )
        self.assertIn(instruction, prompt)
        self.assertIn(r"C:\\temp \2", prompt)

    def test_custom_template_is_used(self):
        prompt = build_reflection_prompt(
            "A {instruction} B {rollouts} C", "INS", ROLLOUTS
        )
        self.assertTrue(prompt.startswith("A INS B Example 1\n"))
        self.assertTrue(prompt.endswith("FEEDBACK-THREE C"))

    def test_both_templates_contain_both_placeholders(self):
        for template in (REFLECTION_PROMPT, DSPY_REFLECTION_PROMPT):
            self.assertIn("{instruction}", template)
            self.assertIn("{rollouts}", template)


class ExtractionTest(unittest.TestCase):
    def test_returns_last_of_several_blocks_without_tag(self):
        reply = "Old:\n```\nfirst\n```\nNew:\n```\nsecond\n```\nDone."
        self.assertEqual(extract_instruction(reply), "second")

    def test_returns_last_of_several_blocks_with_tag(self):
        reply = "```text\nfirst\n```\n\n```markdown\nsecond\n```"
        self.assertEqual(extract_instruction(reply), "second")

    def test_tag_with_spaces_is_allowed(self):
        self.assertEqual(extract_instruction("``` python \nnew\n```"), "new")

    def test_fence_mentioned_mid_sentence_is_not_a_fence(self):
        reply = "I'll put it in a ``` block:\n```\nreal\n```"
        self.assertEqual(extract_instruction(reply), "real")

    def test_indented_fences_are_accepted(self):
        self.assertEqual(extract_instruction("  ```\n  body\n  ```"), "  body")

    def test_closing_fence_must_start_a_line(self):
        with self.assertRaises(ValueError):
            extract_instruction("```\nbody```")

    def test_leading_blank_line_inside_block_is_kept(self):
        self.assertEqual(extract_instruction("```\n\nbody\n```"), "\nbody")

    def test_inner_newlines_and_blank_lines_are_preserved(self):
        reply = "```\nline one\n\n  indented two\nline three\n```"
        self.assertEqual(
            extract_instruction(reply), "line one\n\n  indented two\nline three"
        )

    def test_only_one_trailing_newline_is_removed(self):
        self.assertEqual(extract_instruction("```\nbody\n\n```"), "body\n")

    def test_no_fenced_block_raises_with_start_of_reply(self):
        with self.assertRaises(ValueError) as cm:
            extract_instruction("Here is the instruction without a fence.")
        self.assertIn("Here is the instruction", str(cm.exception))

    def test_unclosed_fence_raises(self):
        with self.assertRaises(ValueError):
            extract_instruction("```\nnever closed")

    def test_empty_fenced_block_raises(self):
        with self.assertRaises(ValueError):
            extract_instruction("Here:\n```\n\n```\n")


if __name__ == "__main__":
    unittest.main()
