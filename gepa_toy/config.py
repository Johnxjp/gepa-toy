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
CACHE_PATH = RESULTS_DIR / "cache.sqlite"

LABELS = json.loads(LABELS_JSON.read_text())

# Chosen in docs/dataset_preparation.md; lower values chain clusters together.
DUPLICATE_CUTOFF = 0.45

GEPA_RESULTS_DIR = RESULTS_DIR / "gepa"
# The leading ~ is OpenRouter's alias for the newest model in the family. The resolved
# model is not logged: if OpenRouter repoints the alias, runs before and after differ.
REFLECTION_MODEL = "~deepseek/deepseek-flash-latest"
