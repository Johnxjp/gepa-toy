from gepa_toy.config import LABELS

# Wording from data/customer_support_tickets/README.md.
DESCRIPTIONS = {
    "queue": {
        "Technical Support": "Technical issues and support requests.",
        "Customer Service": "Customer inquiries and service requests.",
        "Billing and Payments": "Billing issues and payment processing.",
        "Product Support": "Support for product-related issues.",
        "IT Support": "Internal IT support and infrastructure issues.",
        "Returns and Exchanges": "Product returns and exchanges.",
        "Sales and Pre-Sales": "Sales inquiries and pre-sales questions.",
        "Human Resources": "Employee inquiries and HR-related issues.",
        "Service Outages and Maintenance": "Service interruptions and maintenance.",
        "General Inquiry": "General inquiries and information requests.",
    },
    "type": {
        "Incident": "Unexpected issue requiring immediate attention.",
        "Request": "Routine inquiry or service request.",
        "Problem": "Underlying issue causing multiple incidents.",
        "Change": "Planned change or update.",
    },
    "priority": {
        "low": "Non-urgent issues that do not require immediate attention. Examples: general inquiries, minor inconveniences, routine updates, and feature requests.",
        "medium": "Moderately urgent issues that need timely resolution but are not critical. Examples: performance issues, intermittent errors, and detailed user questions.",
        "high": "Urgent issues that require immediate attention and quick resolution. Examples: system outages, security breaches, data loss, and major malfunctions.",
    },
}


def _options(values, descriptions=None):
    if descriptions is None:
        return "\n".join(f"- {v}" for v in values)
    return "\n".join(
        f"- {v}: {descriptions[v]}" if v in descriptions else f"- {v}" for v in values
    )


def _build(subject: str, body: str, descriptions: dict) -> str:
    return f"""You are classifying a customer support ticket.

Pick exactly one queue, one type and one priority. Pick one or more tags. Use only the values listed below, spelled exactly as written.

Queue (choose one):
{_options(LABELS["queue"], descriptions.get("queue"))}

Type (choose one):
{_options(LABELS["type"], descriptions.get("type"))}

Priority (choose one):
{_options(LABELS["priority"], descriptions.get("priority"))}

Tags (choose one or more):
{_options(LABELS["tags"])}

Ticket subject: {subject or "(no subject)"}
Ticket body:
{body}

Respond with only a JSON object in this shape, with no other text:
{{"queue": "<queue>", "type": "<type>", "priority": "<priority>", "tags": ["<tag>"]}}
"""


def build_prompt(subject: str, body: str) -> str:
    return _build(subject, body, {})


def build_expanded_prompt(subject: str, body: str) -> str:
    """Zero-shot prompt plus the README's description of each queue, type and priority."""
    return _build(subject, body, DESCRIPTIONS)
