#!/usr/bin/env bash
set -euo pipefail

URL="https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv?download=true"
DEST="data/customer_support_tickets/data.csv"

mkdir -p "$(dirname "$DEST")"
curl -fL --progress-bar -o "$DEST" "$URL"
