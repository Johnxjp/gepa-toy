import json
from pathlib import Path

# Anchored to this file so scripts work from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "customer_support_tickets"
RAW_CSV = DATA_DIR / "data.csv"
ENGLISH_TSV = DATA_DIR / "english.tsv"
TAG_SPELLING_MAP = DATA_DIR / "tag_spelling_map.tsv"
TAG_GROUPS = DATA_DIR / "tag_groups.tsv"
LABELS_JSON = DATA_DIR / "labels.json"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
GEPA_RESULTS_DIR = RESULTS_DIR / "gepa"

LABELS = json.loads(LABELS_JSON.read_text())

# The only place models are named, so every run records exactly what it used. A free
# model past its daily quota fails the run on its first call.
TASK_MODEL = "nvidia/nemotron-3.5-lightning"
REFLECTION_MODEL = "~deepseek/deepseek-flash-latest"

# Chosen in docs/dataset_preparation.md; lower values chain clusters together.
DUPLICATE_CUTOFF = 0.45
