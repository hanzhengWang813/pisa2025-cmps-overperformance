# School Digital Resources, Family Background and Computational Problem Solving across 84 Systems

Analysis code for the manuscript *School Digital Resources, Family Background and Computational
Problem Solving across 84 Systems* (under review).

The central construct is **relative overperformance**: the part of computational problem solving
(CMPS) that mathematics, reading and science do not explain. The question is whether school
digital resources change the socioeconomic gradient in what remains.

The repository reproduces every number, table and figure in the paper from the OECD PISA 2025
public-use files: data preparation, the construction of *relative overperformance*, two-level
models with Rubin-pooled plausible values, design-based standard errors with 80 BRR replicate
weights, robustness checks, and the final 300 dpi figures.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_all.sh                 # end-to-end pipeline
├── scripts/                   # analysis code (see the pipeline table below)
├── docs/
│   └── analysis_notes_zh.md   # working notes: definitions, key results, coding decisions
├── data/
│   ├── raw/                   # PISA 2025 .zip / .sav go here (not tracked)
│   └── clean/                 # intermediate files written by the pipeline (not tracked)
└── output/                    # tables, reports, figures (not tracked)
```

Raw data, intermediate files and outputs are deliberately excluded from version control
(`.gitignore`). Only code is versioned.

## Data

The analysis uses three PISA 2025 public-use files, available from the OECD:

| File | Content | Used for |
|---|---|---|
| `CY09_MS_STU_PUF` | student questionnaire: CMPS and domain plausible values, ESCS, weights, 80 replicate weights | dependent variable, covariates, weights |
| `CY09_MS_SCH_PUF` | school questionnaire: digital resources, school location | school-level predictors |
| `CY09_MS_LDW_PUF` | item-level log data for the Learning in the Digital World (LDW) units | process indicators (time on task, actions, disengagement) |

Download and unpack:

```bash
bash scripts/download_pisa.sh
```

The script writes the archives to `data/raw/` and extracts them to `data/raw/extracted/`.
`00_inspect.py` reads the `.zip` archives directly, so keep them in place.

Two notes on file names: the LDW file carries an OECD release date in its name
(`CY09_MS_LDW_20260806.sav`); `06_ldw_process.py` matches it by prefix, so a different release
date is fine. The student and school files are expected as `CY09_MS_STU_PUF.sav` and
`CY09_MS_SCH_PUF.sav`.

PISA data are distributed under the OECD's terms of use and are not redistributed here.

## Environment

Python 3.11 or newer (developed and run on 3.13.12).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Peak memory is dominated by the student file: `10_build.py` needs roughly 3–4 GB. The pipeline
reads only the columns it needs, but `pyreadstat` loads the full `.sav` record layout.

## Pipeline

Run from the repository root. Each script resolves paths relative to its own location, so the
working directory does not matter, but the numbered order does.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `00_inspect.py` | Reads SPSS metadata only (no data rows): system coverage, naming of the LDW plausible values, granularity of the process file | `output/inspect_report.txt`, `output/columns_*.csv` |
| 2 | `01_country_probe.py` | Maps `CNT` / `CNTRYID` values in the school file; locates the four Chinese provinces (QCI) | `output/country_coverage_school.csv` |
| 3 | `02_school_vars.py` | Groups school-questionnaire items by theme (infrastructure, curriculum offer, shortage, organisation) | `output/school_vars_thematic.csv` |
| 4 | `03_find_vars.py` | Locates design, weight, plausible-value and background variables by regex instead of assumed names | `output/var_discovery.csv` |
| 5 | `04_school_index.py` | Builds school-level indices: digital shortage, quality, clubs, soft-opportunity index, location | `data/clean/school_index.csv.gz` |
| 6 | `05_stu_probe.py` | Probes student-file variables and missingness | `output/stu_probe.txt` |
| 7 | `06_ldw_process.py` | Aggregates item-level LDW records to per-student process indicators; applies the missing codes (9999997+) | `data/clean/ldw_process.csv.gz` |
| 8 | `10_build.py` | Merges student, school and process data into the analysis file | `data/clean/analysis.csv.gz` |
| 9 | `11_residual.py` | Constructs relative overperformance (pooled cubic regression, within-system standardisation, averaged over ten plausible values) | `data/clean/resid.csv.gz` |
| 10 | `27_system_ranks.py` | Recomputes the pooled cubic residuals **in score points** (step 9 standardises them away) and ranks systems: absolute rank on CMPS, relative rank on the residual | `output/tab_system_resid.csv` |
| 11 | `12_descriptive.py` | System-level descriptives, ESCS gradients across outcomes, urban/rural resource patterns | `output/tab1_country.csv`, `tab2_escs_gradient.csv`, `tab3_urban.csv`, `descriptive_report.txt` |
| 12 | `13_models.py` | Models M1–M6: two-level models with school random intercepts, Rubin-pooled over ten plausible values; design-weighted OLS with school-clustered SE as a design-based check | `output/models_results.json`, `models_report.txt` |
| 13 | `14_robust.py` | Robustness R1–R6: alternative residual definition, rapid responders removed, QCI dropped, alternative overperformer threshold, random ESCS slopes, Johnson–Neyman regions | `output/robust_results.json`, `robust_report.txt` |
| 14 | `45_system_fe.py` | System fixed-effects check: whether pooling dilutes the slope through between-system variance | `output/system_fe_report.txt` |
| 15 | `23_brr.py` | BRR standard errors (Fay = 0.5, 80 replicate weights) for the key estimates | `output/brr_results.json`, `brr_report.txt` |
| 16 | `24_brr_china.py` | BRR for the QCI subsample estimates | `output/brr_china.txt` |
| 17 | `25_brr_country.py` | Per-system ESCS slopes with BRR standard errors; significance classification | `output/brr_country_slopes.csv`, `brr_country_summary.txt` |
| 18 | `52_figures_v2.py` | The five published figures, 300 dpi | `output/figures/fig1…fig5.png` |

`run_all.sh` runs the scripts in this order, skipping the exploration scripts (`01`–`03` and `05`);
step 1 runs only when the `.zip` archives are present. The exploration scripts are kept because
they document how the variables were identified, not because the pipeline needs them.

Runtime: steps 7–8 are I/O bound (minutes), step 11 is the slowest (tens of minutes to a few
hours, depending on cores), step 14–16 are the second slowest because each refits a model 80 × 10
times.

## Definitions and estimation

**Relative overperformance.** For each of the ten plausible values, CMPS is regressed on the
mathematics, reading and science plausible values with cubic terms, pooled internationally and
without interactions. Residuals are standardised within system and averaged across the ten
plausible values. The mean is the point estimate; the between-plausible-value variance enters the
standard errors through Rubin's rules. This follows the definition of *relative performance* used
in OECD (2024), PISA 2022 Volume III, transplanted to computational problem solving.

The resulting variable is a definition, not an estimator: once fixed, it is a deterministic
transformation of the data. `23_brr.py` therefore fixes the transformation and re-estimates only
the sampling variance of the analysis stage.

**Overperformers.** `over_q75` marks students at or above the 75th percentile of the within-system
residual; `over_z1` (z ≥ 1) is used as a sensitivity check.

**Standard errors.** Two routes are reported. Design-based: weighted OLS with school-clustered
standard errors. Replicate-based: 80 BRR replicate weights with Fay = 0.5, which gives standard
errors roughly 1.3–1.4 times the clustered ones. Multilevel models are fitted by maximum
likelihood with school random intercepts; `statsmodels` does not accept sampling weights in
`MixedLM`, so the design-based route carries the weights while the multilevel route carries the
nesting.

## Mapping to the manuscript

| Manuscript element | Source |
|---|---|
| Table 1, columns CMPS / mathematics / absolute rank | `output/tab_system_resid.csv` (step 10), 10-PV averages |
| Table 1, columns relative rank / residual in points | `output/tab_system_resid.csv` (step 10) |
| Table 2 (multilevel models) | `output/models_results.json` (step 12) |
| Table 3 (robustness) | `output/robust_results.json` (step 13) |
| Figure 1 (CMPS against mathematics) | step 18, panel from `tab1_country.csv` |
| Figure 2 (conditional slopes) | step 18, refits the full model |
| Figure 3 (overperformers by ESCS quintile) | step 18, from `tab2_escs_gradient.csv` |
| Figure 4 (process differences) | step 18, from `ldw_process.csv.gz` |
| Figure 5 (per-system slopes) | step 18, from `brr_country_slopes.csv` |

Two conventions coexist and explain the small differences between `tab1_country.csv` and the
published Table 1. `12_descriptive.py` computes descriptive means from the first plausible value
only, the convention used in OECD reports for descriptive tables; `27_system_ranks.py` averages
the ten plausible values, which is what Table 1 reports. For the four Chinese provinces the
difference is 560.71 against 560.31 on CMPS, and the residual is −27.6 points under the
first-value convention against −28.0 as the ten-value average the paper reports.

## Known limitations that the code makes visible

- LDW is a rotated design: each student answers two of six units, so process indicators exist for
  about a third of the sample. `14_robust.py` reports the exact screening scope.
- CMPS and the domain scales are not equated; differences between them are descriptive only and
  are not interpreted as score-point gains or losses.
- ESCS is missing for about 10% of students, and more than 20% in five systems. Schools and
  systems with heavy missingness are retained; the missingness pattern itself is reported.
- Several systems do not administer the gender item; models that include gender drop them, which
  is why the sample sizes in Tables 2 and 3 differ.

## Notes

`docs/analysis_notes_zh.md` is the working log kept during the analysis (Chinese): definitions,
coding decisions, and the numbers each script produced. It is included because it records why
particular choices were made, not as documentation of the final text.

## Chinese quick start

```bash
bash scripts/download_pisa.sh          # 下载并解压三个 PISA 2025 数据文件
pip install -r requirements.txt
python3 scripts/04_school_index.py     # 学校层指标
python3 scripts/06_ldw_process.py      # LDW 过程指标
python3 scripts/10_build.py            # 合并主数据集
python3 scripts/11_residual.py         # 构造相对超额表现
python3 scripts/13_models.py           # 多层模型 M1–M6
python3 scripts/52_figures_v2.py       # 五张图
```

数据文件放在 `data/raw/extracted/`，所有路径均相对仓库根目录解析，与当前工作目录无关。

## Citation

If you use this code, please cite the manuscript:

> School Digital Resources, Family Background and Computational Problem Solving across 84
> Systems (under review).

A `CITATION.cff` will be added with the full author list and DOI once the paper is accepted.

## Licence

The code in this repository is released under the MIT Licence (see `LICENSE`). This licence
covers the code only. The PISA 2025 public-use files are not redistributed here and remain
subject to the OECD's terms of use.
