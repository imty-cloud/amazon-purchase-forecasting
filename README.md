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
| B Data and target validation | Done: `reports/stage_b_data_target_validation.md` (git tag `stage-b`) |
| C Features, baselines, validation design | Done: `reports/stage_c_features_validation.md` |
| D Modelling | Protocol v1.1 `reports/stage_d_protocol.md`; Phase 1 (selection) done: `reports/stage_d_phase1_results.md`; lock committed (tag `stage-d-lock`); Phase 2 (test) not run |
| E Report · F Reproducibility | Not started |

## Layout

| Path | Contents |
|---|---|
| `src/apf/config.py` | Paths, checksums, observation period, 15 cutoffs, 90-day horizon, gift-card rule |
| `src/apf/data.py` | Chunked read-only loader, raw-text quality counts, duplicate and gift-card flags, data versions |
| `src/apf/target.py` | Eligibility, target window, completeness guard, respondent-cutoff panel |
| `src/apf/features.py` | 19 features at a cutoff from history ≤ cutoff; definitions in `FEATURES` |
| `src/apf/evaluation.py` | Metrics for the event y = 0 (AP, ROC-AUC, Brier, calibration, top-k %), respondent-cluster bootstrap, baselines |
| `src/apf/splits.py` | Temporal roles, 20 % respondent hold-out (seed 20261024), embargo and isolation checks |
| `src/apf/models.py` | Stage D model specifications: feature sets, logistic-regression pipeline, LightGBM grid, early stopping, refit, selection rule |
| `src/apf/lock.py` | Test-set guard: selection reads only pre-test rows; test rows only under a committed, unchanged lock |
| `src/apf/phase2.py` | Stage D Phase 2: lock check first, refit with locked configurations, test metrics, P1–P3 comparisons, per-cutoff temporal-stability table, paired bootstrap; runs once |
| `src/apf/plots.py` | Temporal-stability and reliability figures, drawn from saved result tables |
| `scripts/stage_b_validate.py` | Stage B run: checks, panel, sensitivity, repeated-respondent analysis, independent check |
| `scripts/independent_check.py` | Standard-library recomputation of counts and every label (shares no code with `src/`) |
| `scripts/stage_c_features.py` | Stage C run: features, leakage checks, split checks, training-period baseline smoke check |
| `scripts/independent_feature_check.py` | Standard-library recomputation of 13 features for 300 sampled respondents |
| `scripts/mutation_check.py` | Plants known bugs (Stages B and C) in a temporary copy of `src/` and confirms the tests catch them |
| `scripts/build_stage_b_report.py`, `build_stage_c_report.py` | Write the stage reports from `outputs/` |
| `tests/` | Unit tests on synthetic data, plus tests on the real CSV (marker `realdata`, skipped if the file is absent) |
| `outputs/` | `stage_b_validation_summary.json` (machine-readable summary), per-cutoff counts, monthly rows, repeated-respondent table, mutation results. `*.csv.gz` (row-level labels) are git-ignored |

Code reused from the RFM thesis (`../rfm_v3/rfm_study_v3.ipynb`, cells 4 and 6) is marked at its point of use.
The old projects are only read.

## Reproduce Stages B and C

```bash
cd "/Users/quynhtien/Documents/Đồ án dự kiến/amazon_purchase_forecasting"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pytest                          # all tests (about 10 s)
.venv/bin/python scripts/stage_b_validate.py        # about 20 s; writes outputs/stage_b_*
.venv/bin/python scripts/stage_c_features.py        # about 30 s; writes outputs/stage_c_*, model_table_*.csv.gz
.venv/bin/python scripts/mutation_check.py          # about 1 min (42 mutants, 4 in parallel); outputs/mutation_check.json
.venv/bin/python scripts/build_stage_c_report.py    # writes reports/stage_c_features_validation.md
.venv/bin/python scripts/stage_d_phase1.py          # about 20 s; selection on Train -> Tune only (no test labels)
.venv/bin/python scripts/stage_d_write_lock.py      # refuses if a lock exists; --supersede "reason" [--dry-run] to replace it
.venv/bin/python scripts/stage_d_phase2.py          # NOT YET RUN: test evaluation, once, only under a valid lock
```

`reports/stage_b_data_target_validation.md` is frozen at tag `stage-b`; `build_stage_b_report.py` regenerates it
(its run line and test count then reflect the current suite).

Set `APF_DATA_DIR` if the CSVs are not in the project's parent folder. Run `pytest -m "not realdata"` to skip
the tests that need the CSV. The environment is Python 3.14.7 (macOS arm64); `requirements.in` lists the
direct pins.
