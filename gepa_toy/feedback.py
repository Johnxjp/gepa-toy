"""Per-ticket feedback text for the reflector.

Shared by our GEPA loop and the DSPy metric, so both read the same text.
"""

from gepa_toy.evaluate import SINGLE_FIELDS, gepa_score, tag_scores


def _describe(labels: dict) -> str:
    fields = ", ".join(f"{field} {labels[field]}" for field in SINGLE_FIELDS)
    return f"{fields}, tags [{', '.join(labels['tags'])}]"


def _one_line(error: str) -> str:
    # Pydantic errors span several lines and end each one with a docs link the reflector can't use.
    lines = [line.strip() for line in error.splitlines()]
    return " ".join(
        line
        for line in lines
        if line and not line.startswith("For further information")
    )


def prediction_line(predicted: dict | None, error: str | None) -> str:
    """What the model answered, or why its reply couldn't be read."""
    if predicted is None:
        return f"Reply was not valid: {_one_line(error or '')}"
    return f"Predicted: {_describe(predicted)}"


def score_feedback(predicted: dict | None, labels: dict) -> str:
    """The score alone, with no hint of which label was wrong."""
    return f"Score: {gepa_score(predicted, labels):.2f}"


def written_feedback(predicted: dict | None, labels: dict, error: str | None) -> str:
    """Score breakdown, then what was predicted and what was correct."""
    correct = f"Correct:   {_describe(labels)}"
    if predicted is None:
        return (
            f"Score: 0.00 (no valid reply)\n{prediction_line(None, error)}\n{correct}"
        )
    parts = [
        f"{field} {int(predicted[field] == labels[field])}" for field in SINGLE_FIELDS
    ]
    parts.append(f"tags F1 {tag_scores(predicted['tags'], labels['tags'])['f1']:.2f}")
    return (
        f"Score: {gepa_score(predicted, labels):.2f} ({', '.join(parts)})\n"
        f"{prediction_line(predicted, error)}\n"
        f"{correct}"
    )
