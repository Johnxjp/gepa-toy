"""Filter the support tickets to English, clean up tags, and print label breakdowns."""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data/customer_support_tickets")
SOURCE = DATA_DIR / "data.csv"
OUTPUT = DATA_DIR / "english.tsv"
SPELLING_MAP = DATA_DIR / "tag_spelling_map.tsv"
TAG_GROUPS = DATA_DIR / "tag_groups.tsv"
TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]
FIELDS = [
    "subject",
    "body",
    "answer",
    "type",
    "queue",
    "priority",
    "version",
    "tags",
    "tag_groups",
]


def merge_tags(row: dict[str, str]) -> list[str]:
    # Some source cells already hold several comma-joined tags, so split them.
    tags = []
    for column in TAG_COLUMNS:
        for tag in row[column].split(","):
            tag = tag.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tags


def build_spelling_map(counts: Counter) -> dict[str, str]:
    """Map each tag variant to the most common spelling among its case, hyphen and plural variants."""
    variants = defaultdict(list)
    for tag in counts:
        variants[re.sub(r"[-_\s]+", " ", tag.lower()).strip()].append(tag)
    # Only fold a plural when the singular also exists, so "Analytics" or "Sales" stay intact.
    for key in list(variants):
        if key.endswith("s") and key[:-1] in variants:
            variants[key[:-1]] += variants.pop(key)
    spelling = {}
    for tags in variants.values():
        canonical = max(tags, key=lambda t: counts[t])
        for tag in tags:
            spelling[tag] = canonical
    return spelling


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def print_breakdown(name: str, counter: Counter, total: int) -> None:
    print(f"\n## {name} ({len(counter)} values)")
    for value, count in counter.most_common():
        print(f"{value or '(blank)'}\t{count}\t{count / total:.1%}")


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["language"] == "en"]
    with TAG_GROUPS.open(newline="", encoding="utf-8") as f:
        groups = {r["tag"]: r["group"] for r in csv.DictReader(f, delimiter="\t")}

    raw_tags = [merge_tags(r) for r in rows]
    raw_counts = Counter(t for tags in raw_tags for t in tags)
    spelling = build_spelling_map(raw_counts)
    clean_tags = [dedupe([spelling[t] for t in tags]) for tags in raw_tags]
    tag_groups = [
        dedupe([groups[t] for t in tags if t in groups]) for tags in clean_tags
    ]

    with SPELLING_MAP.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["variant", "canonical", "count"])
        for variant, canonical in sorted(spelling.items(), key=lambda kv: kv[1]):
            if variant != canonical:
                writer.writerow([variant, canonical, raw_counts[variant]])

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row, tags, grouped in zip(rows, clean_tags, tag_groups):
            writer.writerow(
                {**row, "tags": ",".join(tags), "tag_groups": ",".join(grouped)}
            )

    total = len(rows)
    print(f"Wrote {total} English rows to {OUTPUT}")
    for column in ["type", "queue", "priority"]:
        print_breakdown(column, Counter(r[column] for r in rows), total)
    # Tag percentages are share of tickets carrying the tag, so they sum past 100%.
    print_breakdown("tags", Counter(t for tags in clean_tags for t in tags), total)
    print_breakdown("tag_groups", Counter(g for gs in tag_groups for g in gs), total)
    print(f"\nTickets with no tag group: {sum(not gs for gs in tag_groups)}")


if __name__ == "__main__":
    main()
