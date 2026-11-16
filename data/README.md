# Data

PISA 2025 public-use files are **not** included in this repository. They are distributed by the
OECD under its terms of use; download them yourself with `bash scripts/download_pisa.sh`, or
place them here manually.

Expected layout after `scripts/download_pisa.sh`:

```
data/
├── raw/
│   ├── CY09_MS_STU_PUF.zip
│   ├── CY09_MS_SCH_PUF.zip
│   ├── CY09_MS_LDW_PUF.zip
│   └── extracted/
│       ├── CY09_MS_STU_PUF.sav
│       ├── CY09_MS_SCH_PUF.sav
│       └── CY09_MS_LDW_20260806.sav
└── clean/            # written by the pipeline
    ├── school_index.csv.gz
    ├── ldw_process.csv.gz
    ├── analysis.csv.gz
    └── resid.csv.gz
```

`00_inspect.py` reads the `.zip` archives directly (metadata only), so keep them in `data/raw/`.
Everything under `data/raw/` and `data/clean/` is git-ignored.

Only one column of the LDW file is derived per item: the scoring, total time, number of actions
and time before the first action, aggregated per student by `06_ldw_process.py`. Event-level logs
remain on disk and are never loaded into memory.
