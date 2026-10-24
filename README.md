# Predicting Future Purchase Activity from E-commerce Transaction Data

*A comparison of machine-learning models and an assessment of temporal stability* (coursework project).

**Question.** Can transaction history available before a cutoff predict whether a respondent records at least one
eligible Amazon purchase in the following 90 days?

**Data.** Open e-commerce 1.0 (Berke et al., 2024, *Scientific Data* 11, 491, doi:10.1038/s41597-024-03329-6;
Harvard Dataverse doi:10.7910/DVN/YGLYDY, v1.0).

- **Where the data stay.** `amazon-purchases.csv` and `survey.csv` remain in the parent folder and are read in
  place. Nothing is copied into this project, and the MD5 checksum is verified on every run.
- **Unit.** `Survey ResponseID` identifies a survey participant, not a verified Amazon customer, and accounts
  may be shared.
- **Target.** y = 0 means no recorded purchase in 90 days. It is not churn.

## Status

| Stage | State |
|---|---|
| A Audit and feasibility | Done: `reports/audit_and_feasibility.md` |
| B Data and target validation | Done: `reports/stage_b_data_target_validation.md` |
| C Features · D Modelling · E Report · F Reproducibility | Not started (awaiting approval) |

## Layout

| Path | Contents |
|---|---|
| `src/apf/config.py` | Paths, checksums, observation period, 15 cutoffs, 90-day horizon, gift-card rule |
| `src/apf/data.py` | Chunked read-only loader, raw-text quality counts, duplicate and gift-card flags, data versions |
| `src/apf/target.py` | Eligibility, target window, completeness guard, respondent-cutoff panel |
| `scripts/stage_b_validate.py` | Stage B run: checks, panel, sensitivity, repeated-respondent analysis, independent check |
| `scripts/independent_check.py` | Standard-library recomputation of counts and every label (shares no code with `src/`) |
| `scripts/mutation_check.py` | Plants known bugs in a temporary copy of `src/` and confirms the tests catch them |
| `scripts/build_stage_b_report.py` | Writes the Stage B report from `outputs/` |
| `tests/` | Unit tests on synthetic data, plus tests on the real CSV (marker `realdata`, skipped if the file is absent) |
| `outputs/` | `stage_b_validation_summary.json` (machine-readable summary), per-cutoff counts, monthly rows, repeated-respondent table, mutation results. `*.csv.gz` (row-level labels) are git-ignored |

Code reused from the RFM thesis (`../rfm_v3/rfm_study_v3.ipynb`, cells 4 and 6) is marked at its point of use.
The old projects are only read.

## Reproduce Stage B

```bash
cd "/Users/quynhtien/Documents/Đồ án dự kiến/amazon_purchase_forecasting"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pytest                          # all tests (about 10 s)
.venv/bin/python scripts/stage_b_validate.py        # about 20 s; writes outputs/
.venv/bin/python scripts/mutation_check.py          # about 10 s
.venv/bin/python scripts/build_stage_b_report.py    # writes reports/stage_b_data_target_validation.md
```

Set `APF_DATA_DIR` if the CSVs are not in the project's parent folder. Run `pytest -m "not realdata"` to skip
the tests that need the CSV. The environment is Python 3.14.7 (macOS arm64); `requirements.in` lists the
direct pins.
