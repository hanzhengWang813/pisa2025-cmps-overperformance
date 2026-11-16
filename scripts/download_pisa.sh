#!/usr/bin/env bash
# Download and unpack the three PISA 2025 public-use files used in this repository.
# Usage: bash scripts/download_pisa.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
EXT="$RAW/extracted"
mkdir -p "$RAW" "$EXT"

BASE_URL="https://webfs.oecd.org/pisa2022/2025"

# Student and school questionnaires, then the item-level LDW file (largest, last)
FILES=(CY09_MS_STU_PUF CY09_MS_SCH_PUF CY09_MS_LDW_PUF)

for f in "${FILES[@]}"; do
  echo "[$(date '+%H:%M:%S')] downloading ${f}"
  curl -L -C - --retry 10 --retry-delay 3 --retry-all-errors \
       --connect-timeout 30 \
       -o "$RAW/${f}.zip" "${BASE_URL}/${f}.zip"
  unzip -tq "$RAW/${f}.zip" >/dev/null
  unzip -oq "$RAW/${f}.zip" -d "$EXT"
  echo "[$(date '+%H:%M:%S')] ok ${f}"
done

ls -lh "$EXT"
