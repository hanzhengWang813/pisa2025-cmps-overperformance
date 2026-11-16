#!/usr/bin/env bash
# End-to-end pipeline. Run from anywhere; paths resolve from the repository root.
# Exploration scripts 01-03 and 05 are omitted; they only document how variables were identified.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

run() {
  echo "===== $1"
  python3 "scripts/$1"
}

# Optional: metadata survey. Needs the .zip archives in data/raw/.
[ -f data/raw/CY09_MS_STU_PUF.zip ] && run 00_inspect.py

run 04_school_index.py     # school-level resources and location
run 06_ldw_process.py      # item-level LDW -> process indicators
run 10_build.py            # merge into the analysis file
run 11_residual.py         # relative overperformance
run 12_descriptive.py      # descriptives and gradients
run 13_models.py           # multilevel models M1-M6
run 14_robust.py           # robustness R1-R6
run 45_system_fe.py        # system fixed effects
run 23_brr.py              # BRR standard errors
run 24_brr_china.py        # BRR for the QCI subsample
run 25_brr_country.py      # per-system slopes with BRR
run 52_figures_v2.py       # figures 1-5

echo "===== done: see output/"
