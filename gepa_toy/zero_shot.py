from gepa_toy.config import LABELS


def _options(values):
    return "\n".join(f"- {v}" for v in values)


def build_prompt(subject: str, body: str) -> str:
    return f"""You are classifying a customer support ticket.

Pick exactly one queue, one type and one priority. Pick one or more tags. Use only the values listed below, spelled exactly as written.

Queue (choose one):
{_options(LABELS["queue"])}

Type (choose one):
{_options(LABELS["type"])}

Priority (choose one):
{_options(LABELS["priority"])}

Tags (choose one or more):
{_options(LABELS["tags"])}

Ticket subject: {subject or "(no subject)"}
Ticket body:
{body}

Respond with only a JSON object in this shape, with no other text:
{{"queue": "<queue>", "type": "<type>", "priority": "<priority>", "tags": ["<tag>"]}}
"""
