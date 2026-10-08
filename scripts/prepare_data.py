"""Filter the support tickets to English, clean up tags, and print label breakdowns."""

import csv
import json
import re
from collections import Counter, defaultdict

from gepa_toy.config import (
    ENGLISH_TSV,
    LABELS_JSON,
    RAW_CSV,
    TAG_GROUPS,
    TAG_SPELLING_MAP,
)
from gepa_toy.data_processing.tsv import read_tsv, write_tsv

TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]
PRIORITY_ORDER = ["low", "medium", "high"]
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
    with RAW_CSV.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["language"] == "en"]
    groups = {r["tag"]: r["group"] for r in read_tsv(TAG_GROUPS)}

    raw_tags = [merge_tags(r) for r in rows]
    raw_counts = Counter(t for tags in raw_tags for t in tags)
    spelling = build_spelling_map(raw_counts)
    clean_tags = [dedupe([spelling[t] for t in tags]) for tags in raw_tags]
    tag_groups = [
        dedupe([groups[t] for t in tags if t in groups]) for tags in clean_tags
    ]

    merged = [
        {"variant": variant, "canonical": canonical, "count": raw_counts[variant]}
        for variant, canonical in sorted(spelling.items(), key=lambda kv: kv[1])
        if variant != canonical
    ]
    write_tsv(TAG_SPELLING_MAP, merged, ["variant", "canonical", "count"])
    write_tsv(
        ENGLISH_TSV,
        [
            {**row, "tags": ",".join(tags), "tag_groups": ",".join(grouped)}
            for row, tags, grouped in zip(rows, clean_tags, tag_groups)
        ],
        FIELDS,
    )
    labels = {
        "queue": sorted({r["queue"] for r in rows}),
        "type": sorted({r["type"] for r in rows}),
        "priority": sorted({r["priority"] for r in rows}, key=PRIORITY_ORDER.index),
        "tags": sorted(set(groups.values())),
    }
    LABELS_JSON.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    total = len(rows)
    print(f"Wrote {total} English rows to {ENGLISH_TSV}")
    for column in ["type", "queue", "priority"]:
        print_breakdown(column, Counter(r[column] for r in rows), total)
    # Tag percentages are share of tickets carrying the tag, so they sum past 100%.
    print_breakdown("tags", Counter(t for tags in clean_tags for t in tags), total)
    print_breakdown("tag_groups", Counter(g for gs in tag_groups for g in gs), total)
    print(f"\nTickets with no tag group: {sum(not gs for gs in tag_groups)}")


if __name__ == "__main__":
    main()
