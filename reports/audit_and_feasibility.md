# Stage A — Read-only audit and feasibility

**Project:** Predicting Future Purchase Activity from E-commerce Transaction Data: A Comparison of Machine Learning
Models and Temporal Stability Assessment
**Date:** 2026-10-24. **Stage:** A (read-only). No old file was modified; the raw CSV was only read.

---

## 1. Paths and access

| Resource | Path | Access | Notes |
|---|---|---|---|
| Parent folder | `/Users/quynhtien/Documents/Đồ án dự kiến/` | read | |
| Raw purchases | `…/amazon-purchases.csv` (313,070,173 B) | read | MD5 `3549d72735b1211ab009ff0177b94bd9` = official Dataverse v1.0 |
| Survey | `…/survey.csv` (5,027 rows) | read | Demographics, collected 2022–23 |
| Data dictionary | `…/fields.csv` | read | Survey question texts only; no purchase-column dictionary |
| Exploratory notebook | `…/dukien.ipynb` (65 cells) | read | Hidden-state cells; not reused (§3) |
| Old fold results | `…/rolling_foldwise_cluster_results.csv` | read | Clustering R²/AUC; not relevant |
| RFM thesis v3 | `…/rfm_v3/` (git, 4 commits) | read | Protected; one pre-existing uncommitted change (`stage0/RESUME_TOMORROW.md`), not ours |
| Audited baseline | `…/rfm_audited/` | read | 72 files carry `uchg` flag and SHA-256 records |
| Python | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` 3.14.7 | run | numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, joblib 1.6.0, **lightgbm 4.7.0** (libomp present). Missing: pytest, pyarrow, statsmodels, xgboost |
| Hardware | 16 GB RAM, 10 cores | | |
| **New project** | `…/amazon_purchase_forecasting/` | read/write | Created 2026-10-24 after approval (§10); own `.venv` and git repository |

`rfm_v3/.venv` was not executed: running it can write bytecode caches into the protected project.

## 2. Data verification (memory-efficient, chunked)

The CSV was read in 250k-row chunks as strings, then reduced to 5 typed columns (peak RSS 0.70 GB, 17 s).

| Check | Result |
|---|---|
| Columns | Order Date, Purchase Price Per Unit, Quantity, Shipping Address State, Title, ASIN/ISBN (Product Code), Category, Survey ResponseID |
| Rows / respondents | 1,850,717 / 5,027 (all 5,027 in `survey.csv`, and vice versa) |
| Date range | 2018-01-01 – 2024-08-15; 0 unparseable dates |
| Price, Quantity | 0 missing, 0 non-numeric, 0 zero, 0 negative |
| Missing | Title 89,740; Category 89,458; State 87,812; ASIN 973 |
| Exact duplicate rows (extra copies) | 11,624 (may be genuine repeat purchases) |
| Gift-card rows (rfm_v3 rule) | 41,483 (descriptor reports 40,368; rule unpublished) |
| Rows after 2023-03-20 (end of collection) | 29 rows, 9 IDs (incl. one 2024-08-15 row) |
| Monthly rows | ~35–42k/month to Nov 2022, then 32k, 17k, 16k, 7k (Dec 2022–Mar 2023): exports end in the collection window |

All counts agree with `rfm_audited/results_v2/00_audit.csv`. Re-implementing the old 61-day sample flow from scratch
reproduces `01_sample_flow.csv` exactly (e.g. 2020-10-31: n = 4,861, 14.75 % zero).

## 3. Reusable components

All in `rfm_v3/rfm_study_v3.ipynb` (byte-identical code to `rfm_audited/rfm_study_v2_allinone_audited.ipynb`).
The audit (`rfm_audited/AUDIT.md` §2) verified that features are unchanged when all rows after the cutoff are deleted.

| Component | Source (cell) | Decision |
|---|---|---|
| `OFFICIAL_MD5`, column names, `START`, `COLLECTION_END`, gift regexes | 4 | **Unchanged** |
| `md5sum`, `sha256sum`, `verify_files` | 6 | **Unchanged** |
| `load_purchases` (Spend, `is_dup_copy`, `is_gift`, `is_post_collection`) | 6 | **Adapt**: explicit dtypes, `category` IDs, drop unused columns after flagging (memory) |
| `data_version` (raw / dedup / nogift / dedup_nogift) | 6 | **Unchanged** |
| `make_store` (purchase-day table, first date per category) | 6 | **Adapt**: add quantity, item count, gift share per day |
| `features_at` (R, F, M, Tenure, spend per day, unique categories, median interval, `assert Last <= c`) | 6 | **Adapt**: keep logic and assertion; add windowed features (§6); Tenure = c − first purchase (old: last − first) |
| `target_after` (count in (c, c+H]) | 6 | **Adapt**: binary, H = 90, refuse windows ending after 2022-10-31 |
| `stage0_verify` audit dictionary | 26 | **Adapt** to a validation report |
| `make_joint_weights`, `ci` (respondent-cluster bootstrap shared across cutoffs) | 14 | **Unchanged** (used with `sample_weight` AP/AUC/Brier) |
| `environment_info` | 18 | **Adapt**: add lightgbm |
| `Prep` (fold-internal imputation/log/scale) | 8 | **Replace** with an sklearn `Pipeline` (same idea, cleaner) |
| `cv_oof`, `oot_customer_disjoint` | 12 | **Not reused as is** (regression, random KFold); out-of-respondent idea re-implemented |
| R² metrics, K-Means specs, ratios, descriptive clustering, report builder | 10, 14, 16, 22, 68 | **Not reused** (clustering/regression-specific) |
| `dukien.ipynb` cells 45, 49, 53 (XGBoost / LR classifiers) | — | **Not reused**: random `train_test_split`, full-sample scaling, end date 2022-12-31 (censored) |

Old code is coupled to the notebook and clustering; I recommend a clean `src/` package with the above functions
copied in, each marked with its source cell.

## 4. Filtering rules (verified against code and reports)

| Rule | Old practice | Proposal |
|---|---|---|
| Start date | rows < 2018-01-01 dropped (none exist) | same |
| Duplicates | kept (primary); `dedup` sensitivity | same. Dedup cannot change day-level targets (identical rows share a date); it changes only spend features |
| Gift cards | kept (primary); `nogift` sensitivity | **Decision needed (Q2)**. Effect on 90-day negative rate: +0.1 to +0.9 pp |
| Post-collection rows | flagged; never enter a window | same; all windows end ≤ 2022-10-31 |
| Missing Category/Title | kept; ignored by category counts | same; add "missing-category share" feature |
| Spend | Price × Quantity, no cleaning needed | same |

## 5. 90-day target feasibility

**Target:** `y = 1` if the respondent has ≥ 1 eligible purchase day in (c, c + 90 d], among respondents with ≥ 1
purchase day in [2018-01-01, c]. No population filter uses post-cutoff data.

**Completeness.** No per-respondent export date exists. Per the data descriptor (Berke et al., 2024, as quoted in
rfm_v3; to be re-verified in Stage E) collection started November 2022, so every window ending ≤ 2022-10-31 is
fully observed for every respondent. Latest complete cutoff: **2022-08-02**. The 2022-09-30 cutoff (window to
2022-12-29) is **excluded**: it is not labelled at all, rather than labelled negative.

Quarter-end cutoffs, raw data (gift cards and duplicates kept):

| Cutoff | Window end | n | y=1 | y=0 | % y=0 |
|---|---|---|---|---|---|
| 2018-06-30 | 2018-09-28 | 3,774 | 3,196 | 578 | 15.3 |
| 2018-09-30 | 2018-12-29 | 4,040 | 3,530 | 510 | 12.6 |
| 2018-12-31 | 2019-03-31 | 4,281 | 3,558 | 723 | 16.9 |
| 2019-03-31 | 2019-06-29 | 4,378 | 3,563 | 815 | 18.6 |
| 2019-06-30 | 2019-09-28 | 4,469 | 3,675 | 794 | 17.8 |
| 2019-09-30 | 2019-12-29 | 4,550 | 3,837 | 713 | 15.7 |
| 2019-12-31 | 2020-03-30 | 4,669 | 3,852 | 817 | 17.5 |
| 2020-03-31 | 2020-06-29 | 4,728 | 4,056 | 672 | 14.2 |
| 2020-06-30 | 2020-09-28 | 4,796 | 4,089 | 707 | 14.7 |
| 2020-09-30 | 2020-12-29 | 4,842 | 4,279 | 563 | 11.6 |
| 2020-12-31 | 2021-03-31 | 4,882 | 4,279 | 603 | 12.4 |
| 2021-03-31 | 2021-06-29 | 4,915 | 4,340 | 575 | 11.7 |
| 2021-06-30 | 2021-09-28 | 4,932 | 4,287 | 645 | 13.1 |
| 2021-09-30 | 2021-12-29 | 4,956 | 4,432 | 524 | 10.6 |
| 2021-12-31 | 2022-03-31 | 4,991 | 4,364 | 627 | 12.6 |
| 2022-03-31 | 2022-06-29 | 5,001 | 4,359 | 642 | 12.8 |
| 2022-06-30 | 2022-09-28 | 5,008 | 4,384 | 624 | 12.5 |
| *2022-09-30* | *2022-12-29* | *excluded: incomplete window* | | | |

The `nogift` version gives 0.1–0.9 pp more zeros; `dedup` is identical at day level.

**Recommended cutoffs:** 2018-12-31 … 2022-06-30 (15 quarterly cutoffs; ≥ 12 months of history at each).
Total 71,398 respondent-cutoff observations: 61,354 positive and 10,044 negative.

**Consequences:**
1. **Class balance is inverted.** 85.9 % of observations are positive (81.4–89.4 % by cutoff; *corrected in
   §10: the first version said "about 87 %"*), so AP on y = 1 has a floor near 0.86 and says little. Recommendation: report AP for the minority outcome "no purchase" (score 1 − p) as the primary
   ranking metric, plus ROC-AUC, Brier and Brier skill score. **Decision needed (Q3).**
2. **Seasonality.** % y=0 is lowest for September cutoffs (holiday window) and highest for March/June cutoffs.
   Cutoff quarter is known at the cutoff, so it is a legitimate feature. Comparisons across cutoffs must allow for
   season.
3. **COVID-19.** 2020 cutoffs show a level shift (14–18 % → 11–13 % y=0). This is the natural test of temporal
   stability.

## 6. Repeated observations and validation design

3,774 of 5,027 respondents appear at all 17 complete quarterly cutoffs (79,212 observations from 2018-06-30). The
observations are repeated measures, not independent, so random row-level CV would put the same person in train
and test and overstate performance.

Proposed design (an *embargo* rule: a training label window must end on or before the evaluation cutoff):

| Role | Cutoffs | Last label used |
|---|---|---|
| Training | 2018-12-31 – 2021-03-31 (10) | 2021-06-29 |
| Validation (tuning, threshold, calibration) | 2021-06-30, 2021-09-30 | — |
| Refit | 2018-12-31 – 2021-12-31 (13) | 2022-03-31 |
| **Test (touched once)** | **2022-03-31, 2022-06-30** | — |

- Hyperparameters chosen by rolling-origin validation inside the training period (never by random KFold on pooled
  cutoffs).
- **Out-of-respondent sensitivity:** a fixed 20 % of respondents (GroupShuffleSplit by `Survey ResponseID`) is
  never used for training; test metrics are reported separately for seen and unseen respondents.
- **Temporal stability:** rolling-origin evaluation at every cutoff from 2020-03-31 with frozen hyperparameters.
  Track AP, AUC, Brier, calibration slope/intercept and feature drift (PSI) against the training period.
  Points overlapping the validation period are labelled as such.
- **Uncertainty:** respondent-cluster bootstrap (reused `make_joint_weights`), paired between models.

## 7. Leakage risks and controls

| # | Risk | Control |
|---|---|---|
| 1 | Feature uses post-cutoff rows | Filter `date <= c` at one point; assert; test that deleting rows after c leaves features unchanged |
| 2 | Incomplete window labelled 0 | `target_after` raises if c + 90 > 2022-10-31; test |
| 3 | Imputation/scaling fit on all data | sklearn `Pipeline` fitted on training rows only |
| 4 | Tuning or threshold on test | Validation cutoffs only; test evaluated once, logged |
| 5 | Same respondent in train and test | Temporal split + out-of-respondent sensitivity |
| 6 | Training label window overlaps test period | Embargo assertion: window end ≤ test cutoff |
| 7 | **Survey variables** are answered in 2022–23, after every cutoff; `Q-amazon-use-how-oft` and `Q-life-changes` (2021) describe the outcome period | **Exclude from primary models.** At most a labelled sensitivity with stable demographics (age, state, household size) |
| 8 | Panel selection: respondents were recruited in 2022–23, so early cutoffs condition on later panel membership | Cannot be removed; disclose. Population at c is never filtered on later behaviour |
| 9 | Old project choices (cutoff dates, horizons) were made after outcome exploration on these data | Freeze cutoffs, horizon, features, model grid and metrics in `decision_rules.md` before Stage D; label later changes exploratory |
| 10 | Left truncation (3,351 IDs first seen in 2018Q1) | Tenure called "observed history length"; cutoffs start 2018-12-31 |

## 8. Proposed features (all at cutoff c, purchase-day level)

Recency; frequency (purchase days) overall and in the last 30/90/180/365 days; monetary overall and windowed;
AOV proxy (spend per purchase day; no order IDs exist); items per day; category diversity (all-time and last
365 d); observed tenure (c − first); inter-purchase interval median/mean/CV and the latest gap; recency ÷ median
interval; trend (last 90 d vs previous 90 d); active months in the last 12; activity in the same 90-day window
one year earlier; gift-card and missing-category shares; cutoff quarter.

**Baselines:** prevalence; "purchased in the last 90 days"; same window last year.
**Models:** L2 Logistic Regression (log1p + standardization + missing indicators); LightGBM 4.7.0 (available,
native NaNs, `n_jobs` and seed fixed). Calibration is checked; isotonic or Platt only if fitted on validation.
**Threshold:** chosen on validation. Recommended: a capacity rule that flags the top 10 % / 20 % predicted
"no purchase" risk, plus the F1-optimal threshold for reference.

## 9. Recommended structure and plan

```
amazon_purchase_forecasting/
  README.md  pyproject.toml / requirements.in / requirements.lock.txt  decision_rules.md
  config/config.yaml            # data path (outside project), cutoffs, horizon, seeds
  src/apf/
    data.py      # verify, load, data versions        (from rfm_v3 cell 4/6)
    store.py     # purchase-day aggregates            (from cell 6)
    features.py  # features_at                        (adapted from cell 6)
    target.py    # target_after + completeness guard  (adapted from cell 6)
    splits.py    # temporal/embargo/group splits      (new)
    models.py    # LR / LightGBM pipelines            (new)
    evaluate.py  # metrics, bootstrap, calibration    (new + cell 14)
    report.py    # tables/figures from saved results  (new)
  scripts/01_validate.py … 05_report.py
  tests/                        # synthetic-data unit tests (leakage, eligibility, embargo)
  results/  figures/  reports/  # git-ignored large outputs; raw CSV never copied
```

Own `.venv` (Python 3.14.7) with the rfm_v3 analysis pins + lightgbm 4.7.0 + pytest; git initialised in the new
project only.

| Stage | Output | Gate |
|---|---|---|
| B | `data.py`, `store.py`, `target.py`; validation report; target table above reproduced by tests | counts match §5 |
| C | `features.py`; leakage tests (row deletion, embargo, completeness); feature table | all tests pass |
| D | Frozen decision rules; baselines, LR, LightGBM; validation then single test run; stability run | your approval |
| E | Tables, figures, pipeline diagram, report (APA 7, 2021–2026, no MDPI/Frontiers/Hindawi, DOIs verified) | |
| F | Clean re-run from empty results, hash manifest, test run, report-vs-CSV consistency check | |

Estimated runtime for Stage D: minutes (71k rows × ~30 features).

---

## 10. Finalisation and reassessment (2026-10-24, before Stage B)

**Decisions received.**
- The project was created at `/Users/quynhtien/Documents/Đồ án dự kiến/amazon_purchase_forecasting/`.
- Gift cards are **kept in the primary analysis**, with a sensitivity check that excludes them.
- Stage B was approved. Stages C–F are not.

**Corrections to this report.**
1. Overall prevalence of y = 1 is 85.9 %, not "about 87 %" (§5; the per-cutoff table was correct).
2. 3,774 respondents are present at all 17 complete cutoffs counted from 2018-06-30. With the 15 recommended
   cutoffs (from 2018-12-31) the figure is 4,281. These are different cutoff sets, not an error.

**Reassessment of the design.**

| Item | Assessment |
|---|---|
| 90-day target | Kept. Stage B shows that most y = 0 respondents buy again later (89 %), so the target measures short-term purchase activity, not exit. "Churn" stays excluded from the wording |
| Observation coverage | Now verified against the descriptor text (Berke et al., 2024, PMC11091104): collection ran November 2022 – 20 March 2023, and each export runs from 2018-01-01 to the survey date. Coverage through 2022-10-31 follows from that. The exact start day in November is not given; this does not change the bound |
| Cutoffs | 15 quarter-ends from 2018-12-31 to 2022-06-30 kept. Windows never overlap; 17 single days between windows belong to no window (Stage B §4) |
| Label prevalence | y = 0 is the minority (14.1 %), so the primary ranking metric should target y = 0. **Still awaiting approval (Q3)** |
| Temporal validation | The embargo rule holds for the proposed roles. Only 93 test respondents are absent from training, so an unseen-respondent evaluation needs the deliberate hold-out. **Design still awaiting approval (Q4)** |
| Repeated respondents | Strong dependence: ICC 0.35; P(y=0 \| previous y=0) = 0.53 vs 0.07. Cluster bootstrap by respondent and a previous-label/recency baseline are required |
| Leakage | Unchanged list (§7). Survey variables stay excluded. Stage B adds tests, real-data perturbation checks and an independent recomputation for the target and eligibility |

**Unresolved limitations** (details in `reports/stage_b_data_target_validation.md` §11):
- `Survey ResponseID` is a random survey identifier, not a verified Amazon customer ID, and accounts may be
  shared.
- y = 0 is not churn.
- Per-respondent export dates are unavailable.
- The panel was recruited in 2022–23 (selection).
- Histories are left-truncated at 2018-01-01.
- The meaning of `Order Date` is undefined in the descriptor.
- The gift-card rule differs from the descriptor's.
- The data were explored earlier in the rfm_v3 thesis.
