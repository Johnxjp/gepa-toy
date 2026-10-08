from gepa_toy.config import SPLITS_DIR
from gepa_toy.data_processing.tsv import read_tsv


def ticket_text(row: dict[str, str]) -> str:
    # Bodies contain literal "\n" sequences, not real newlines.
    return f"{row['subject']} {row['body']}".replace("\\n", " ")


def load_split(name: str) -> list[dict[str, str]]:
    return read_tsv(SPLITS_DIR / f"{name}.tsv")
