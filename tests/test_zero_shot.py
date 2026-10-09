import unittest

from gepa_toy.config import LABELS
from gepa_toy.zero_shot import DESCRIPTIONS, build_expanded_prompt, build_prompt


class BuildPromptTest(unittest.TestCase):
    def test_lists_every_label_value(self):
        prompt = build_prompt("Subject", "Body")
        for values in LABELS.values():
            for value in values:
                self.assertIn(f"- {value}\n", prompt)

    def test_includes_subject_and_body(self):
        prompt = build_prompt("Cannot log in", "My password is rejected.")
        self.assertIn("Ticket subject: Cannot log in", prompt)
        self.assertIn("Ticket body:\nMy password is rejected.", prompt)

    def test_empty_subject_is_marked(self):
        self.assertIn("Ticket subject: (no subject)", build_prompt("", "Body"))


class BuildExpandedPromptTest(unittest.TestCase):
    def test_descriptions_cover_exactly_the_label_values(self):
        for field, descriptions in DESCRIPTIONS.items():
            self.assertEqual(set(descriptions), set(LABELS[field]))

    def test_every_description_appears_next_to_its_label(self):
        prompt = build_expanded_prompt("Subject", "Body")
        for descriptions in DESCRIPTIONS.values():
            for value, text in descriptions.items():
                self.assertIn(f"- {value}: {text}\n", prompt)

    def test_tags_stay_plain_and_ticket_is_included(self):
        prompt = build_expanded_prompt("Cannot log in", "Body text")
        for tag in LABELS["tags"]:
            self.assertIn(f"- {tag}\n", prompt)
        self.assertIn("Ticket subject: Cannot log in", prompt)
        self.assertIn("Ticket body:\nBody text", prompt)


if __name__ == "__main__":
    unittest.main()
