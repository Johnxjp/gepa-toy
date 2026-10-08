import unittest

from gepa_toy.zero_shot import LABELS, build_prompt


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


if __name__ == "__main__":
    unittest.main()
