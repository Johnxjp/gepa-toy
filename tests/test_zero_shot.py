import unittest
from pathlib import Path

from gepa_toy.config import LABELS
from gepa_toy.zero_shot import (
    DESCRIPTIONS,
    PLAIN_INSTRUCTION,
    SEED_INSTRUCTION,
    build,
    build_expanded_prompt,
    build_prompt,
)

# Captured from build_prompt before the instruction was split out of the template.
GOLDEN = Path(__file__).with_name("golden_zero_shot_prompt.txt").read_text()


class BuildPromptTest(unittest.TestCase):
    def test_matches_golden_byte_for_byte(self):
        self.assertEqual(
            build_prompt("Cannot log in", "My password is rejected."), GOLDEN
        )

    def test_empty_subject_is_marked(self):
        self.assertIn("Ticket subject: (no subject)", build_prompt("", "Body"))


class BuildTest(unittest.TestCase):
    def test_braces_in_instruction_stay_verbatim_and_ticket_fills_once(self):
        instruction = ' Reply like {"queue": "x"} or {body} and never }{\n'
        prompt = build(instruction, "SUBJ-1", "BODY-1")
        self.assertTrue(prompt.startswith(instruction + "\n\n"))
        self.assertEqual(prompt.count("{body}"), 1)
        self.assertEqual(prompt.count("Ticket subject: SUBJ-1\n"), 1)
        self.assertEqual(prompt.count("Ticket body:\nBODY-1\n"), 1)


class SeedPromptTest(unittest.TestCase):
    def test_descriptions_cover_exactly_the_label_values(self):
        for field, descriptions in DESCRIPTIONS.items():
            self.assertEqual(set(descriptions), set(LABELS[field]))

    def test_each_description_and_label_line_appears_exactly_once(self):
        # Whole-line comparison: names such as "Request" also occur inside
        # description text, so a substring count would not mean "listed once".
        prompt = build_expanded_prompt("Subject", "Body")
        items = [line for line in prompt.splitlines() if line.startswith("- ")]
        described = [
            f"- {value}: {DESCRIPTIONS[field][value]}"
            for field in DESCRIPTIONS
            for value in LABELS[field]
        ]
        listed = [f"- {value}" for values in LABELS.values() for value in values]
        self.assertCountEqual(items, described + listed)
        for descriptions in DESCRIPTIONS.values():
            for text in descriptions.values():
                self.assertEqual(prompt.count(text), 1, text)

    def test_seed_is_opening_paragraph_plus_descriptions(self):
        self.assertTrue(SEED_INSTRUCTION.startswith(PLAIN_INSTRUCTION + "\n\n"))
        prompt = build_expanded_prompt("Subject", "Body")
        fixed_part = prompt[len(SEED_INSTRUCTION) :]
        for descriptions in DESCRIPTIONS.values():
            for text in descriptions.values():
                self.assertIn(text, SEED_INSTRUCTION)
                self.assertNotIn(text, fixed_part)

    def test_ticket_is_included(self):
        prompt = build_expanded_prompt("Cannot log in", "Body text")
        self.assertIn("Ticket subject: Cannot log in", prompt)
        self.assertIn("Ticket body:\nBody text", prompt)


if __name__ == "__main__":
    unittest.main()
