# Stage D protocol — model selection, refitting and evaluation

**Version 1.1 (2026-10-25; review changes in §12). Status: design approved; Phase 1 cleared to run;
not locked.** No test metric has been computed.

**Purpose.** The protocol fixes every analysis choice before any tune or test result exists.
- Once approved, the file is committed and its SHA-256 is written into `outputs/stage_d_locked_config.json`
  together with the selected configurations.
- The test-phase code (`apf.lock.release_test_rows`) refuses to release test rows unless that lock file, this
  protocol and all code are committed and unchanged.
- Any later change creates `stage_d_protocol_v2.md` with a dated change log. Every result affected by the
  change is labelled *exploratory*, and the version 1 results are still reported.

Sources for all numbers quoted here: `outputs/stage_c_summary.json` and `outputs/stage_c_split_summary.csv`.

---

## 1. Question, unit, target

- **Question:** can transaction history up to a cutoff predict whether a respondent records ≥ 1 purchase day
  in the following 90 days?
- **Unit:** one respondent-cutoff pair. `Survey ResponseID` is a survey identifier, not a verified Amazon
  customer ID.
- **Event of interest:** **y = 0**, no purchase day in days c+1 … c+90. This is not churn: 89 % of such
  observations buy again later (Stage B §8).
- **Data version (primary):** all rows, with gift cards and exact duplicates kept. No other data version is
  analysed in Stage D unless approved separately (§11, D-Q4).

## 2. Data sets

Hold-out: 1,002 of 5,008 respondents (20 %, seed 20261024, `apf.splits.holdout_respondents`).

| Set | Respondents | Cutoffs | Rows | y = 0 | Used for |
|---|---|---|---|---|---|
| **Train** | non-hold-out | 2018-12-31 … 2021-03-31 (10) | 37,239 | 5,606 | fitting during selection |
| **Tune** | non-hold-out | 2021-06-30, 2021-09-30 | 7,915 | 941 | early stopping, choice of C and of the LightGBM configuration |
| **Refit** | non-hold-out | 2018-12-31 … 2021-12-31 (13) | 49,147 | 7,057 | final fit of every model |
| **Temporal test** (primary) | non-hold-out | 2022-03-31, 2022-06-30 | 8,005 | 1,017 | evaluation, once |
| **Hold-out test** | hold-out | 2022-03-31, 2022-06-30 | 2,004 | 249 | evaluation, once |
| Hold-out, training period (diagnostic) | hold-out | 2018-12-31 … 2021-03-31 | 9,271 | 1,376 | evaluation, after locking |

**Phases and data access.**
- **Phase 1 (selection)** builds labels and features only for the 13 cutoffs before 2022-03-31 and keeps only
  non-hold-out rows (`apf.lock.rows_before_test`). No test-cutoff label is constructed.
- **Hold-out membership** is derived from respondent IDs alone: those with a purchase on or before 2022-06-30,
  read from purchase history without any label. Hold-out rows are dropped before any fitting or scoring, and
  the train and tune counts are asserted equal to Stage C.
- **Phase 2 (test)** starts only after the lock (§7).

## 3. Models

All models output `p_no_purchase` = P(y = 0). Logistic regression and LightGBM are fitted on y and converted
with `apf.evaluation.p_no_purchase`, which selects the class by label.

| ID | Model | Features | Hyperparameters |
|---|---|---|---|
| B0 | Prevalence baseline | — | none (training rate of y = 0) |
| B1 | Previous-quarter persistence | `prev_90d_active` (0 / 1 / no prior) | none (cell rates) |
| B2 | Recency rule | `recency_days` | none (rank = recency; probabilities from an isotonic map) |
| M1 | Logistic regression, RFM only | `recency_days`, `frequency_days`, `monetary_total` | C ∈ {0.001, 0.01, 0.1, 1, 10} |
| M2 | Logistic regression, full | the 19 predefined features | C as M1 |
| M3 | LightGBM, full | the 19 predefined features | grid §4.2 |
| S1 | Logistic regression, without `prev_90d_active` | 18 features | C as M1 |
| S2 | LightGBM, without `prev_90d_active` | 18 features | grid §4.2 |

S1 and S2 form the sensitivity analysis: they show how much the results depend on the previous outcome. They
are tuned with the same procedure as M2 and M3.

**Preprocessing** (`apf.models`; all fitted only on the rows the model is fitted on):
- **Logistic regression:**
  - `log1p`, then standardisation, for counts, spend, recency, tenure and gaps;
  - `median_gap_days` and `recency_to_median_gap`: median imputation plus a missing indicator, then `log1p`
    and standardisation;
  - `prev_90d_active`: median imputation plus a missing indicator, then standardisation;
  - `cutoff_quarter`: one-hot with fixed categories 1–4 (first dropped);
  - L2 penalty, `max_iter` 5000, no class weights (probabilities stay on the observed scale).
- **LightGBM:** raw features with native NaN handling and no class weights.
- **Class imbalance** (y = 0 is about 13–15 %) is handled by the choice of metric (AP for the minority class,
  plus calibration), not by weighting or resampling. Either would distort the probabilities that the Brier
  score and calibration evaluate.

## 4. Phase 1 — model selection (non-hold-out rows only; fit on Train, score on Tune)

**Selection criterion:** Average Precision for y = 0 on the **pooled** tune rows (both tune cutoffs
together). Other tune metrics are recorded but never used for selection.

### 4.1 Logistic regression (M1, M2, S1)
For each C in the grid: fit on Train, score Tune, compute tune AP. Select the highest; exact ties go to the
smaller C (stronger regularisation). A `ConvergenceWarning` makes that C ineligible and is recorded.

### 4.2 LightGBM (M3, S2)
- **Fixed settings:** objective `binary`, learning rate 0.03, `n_estimators` cap 3000, `subsample` 1.0,
  `colsample_bytree` 1.0, `reg_lambda` 1.0, `deterministic=True`, `force_row_wise=True`, `n_jobs` 4,
  `random_state` 20261024.
- **Grid (6 configurations):** `num_leaves` ∈ {7, 15, 31} × `min_child_samples` ∈ {50, 200}.
- **Early stopping:**
  - on the Tune rows, with the custom metric AP for y = 0 (`ap_no_purchase_eval`) as the only metric;
  - patience 200 rounds;
  - `best_iteration_` is recorded.
  - A test checks that LightGBM's best score equals our `metrics()` AP.
- **Selection:** the configuration with the highest tune AP at its best iteration. Exact ties go to fewer
  `num_leaves`, then more `min_child_samples`.
- **Recorded checks:**
  - LightGBM's reported best score must equal our `metrics()` AP for every configuration;
  - a best iteration at the 3000 cap is flagged and reported, with no change to the grid.
- **Caveat:** early stopping and configuration choice share the Tune set, so tune AP is optimistic for LightGBM.
  Tune results are reported as *selection-set* results, never as performance estimates.

### 4.3 Baselines
No hyperparameters. They are fitted on Train for the tune report and on Refit for testing.

### 4.4 Reproducibility check of Phase 1
Phase 1 (`scripts/stage_d_phase1.py`) is run twice, in separate processes. The two
`stage_d_phase1_selection.json` files must be identical except for `run_started` and `runtime_seconds`, and
the two `stage_d_tune_results.csv` files must be identical except for the per-fit `seconds` column. Any other
difference stops Stage D.

## 5. Refitting (after selection, before testing)

- Every model is refitted **once** on the Refit set: non-hold-out rows at cutoffs 2018-12-31 … 2021-12-31.
- The tune rows join the fitting data only after selection.
- **LightGBM keeps the selected configuration and exactly the selected `best_iteration_`** as `n_estimators`
  (`apf.models.make_lgbm_refit`, tested). The count is not rescaled for the larger refit set (49,147 vs
  37,239 rows). This may slightly under-fit, which errs on the conservative side.
- **Logistic regression** keeps the selected C. Imputation and scaling are re-estimated on the refit rows.
- **Baselines** re-estimate their rates (B0, B1) and isotonic map (B2) on the refit rows.
- **One model for both test cutoffs.** The 2022-03-31 labels end 2022-06-29, before the second test cutoff,
  but they are **not** used for a rolling refit. Tuning also forecasts 1–2 quarters ahead of its last fitted
  cutoff, so selection and testing match in forecast distance.

## 6. Metrics

Computed for every model on every evaluation set, pooled over the two test cutoffs and per cutoff.

| Metric | Definition |
|---|---|
| **AP for y = 0 (primary)** | `average_precision_score(y == 0, score0)`; reported with its no-skill value (prevalence of y = 0 in that set) |
| ROC-AUC | for y = 0 vs score0 |
| Brier score | mean (1[y = 0] − p0)²; Brier skill against the prevalence of y = 0 in the model's fitting set |
| Calibration | slope (logistic recalibration on logit p0) and calibration-in-the-large intercept; reliability table and plot in deciles of p0 |
| Precision and recall for y = 0 at **top 10 %** and **top 20 %** | within each cutoff, flag the ⌈k · n_c⌉ highest score0; ties at the threshold shared in expectation; counts pooled over cutoffs (`top_fraction_by_group`) |

- **Top-k estimand** (`top_fraction_by_group`):
  - within each cutoff c, independently, flag k_c = ⌈f · n_c⌉ rows with the highest score0 (f = 0.10, 0.20);
  - rows tied with the k_c-th score are flagged in expectation (as if ties were broken at random), giving
    the expected count tp_c of flagged y = 0 rows;
  - across cutoffs, counts are pooled: precision = Σ tp_c / Σ k_c and recall = Σ tp_c / Σ events_c, not
    averages of per-cutoff rates.
- **score0:** for B2 it is `recency_days`; for every other model it is p0. Brier and calibration use p0
  throughout, including B2's isotonic probabilities.
- **No test-based tuning, threshold selection or recalibration.** The operating points are fixed fractions
  defined here. No model is recalibrated in the primary analysis.

## 7. Locking and Phase 2 (test)

1. **Phase 1** writes the *candidate* selection `outputs/stage_d_phase1_selection.json` and
   `outputs/stage_d_tune_results.csv`. These contain:
   - the selected C values, LightGBM configurations and best iterations;
   - all tune metrics and the full grid results;
   - every warning, the protocol SHA-256 at run time, data MD5, package versions and seeds.

   **The lock is not written.**
2. **Report** the Phase 1 results to the project owner. Only after approval: write
   `outputs/stage_d_locked_config.json` from the candidate selection (`apf.lock.write_lock`) and commit it
   with the protocol and code.
3. **Phase 2** loads test rows through `release_test_rows`. That call fails unless:
   - the lock exists;
   - the protocol hash matches;
   - `src/`, `scripts/`, this protocol and the lock file have no uncommitted changes.
4. **Phase 2 runs once.** It refits (§5) and evaluates:
   - (a) the temporal test (primary);
   - (b) the hold-out test;
   - (c) the hold-out training-period diagnostic, using the Phase-1 models fitted on Train.

   It writes predictions, metrics and bootstrap draws to `outputs/stage_d_*`.
5. **No re-runs with changed settings.** A re-run is permitted only to reproduce identical numbers.

## 8. Uncertainty

- **Paired respondent-cluster bootstrap:** B = 1,000 resamples, seed 20261024 (`paired_cluster_bootstrap`).
  - Each evaluation set is resampled separately.
  - Respondents are drawn with replacement, and all of a respondent's rows in that set (both test cutoffs)
    travel together.
  - Every model is scored on the same draws, so model differences are paired.
- **Intervals:** percentile 95 % intervals for each metric of each model, and for paired differences in AP for
  y = 0 (also ROC-AUC and Brier).
- **Predefined comparisons** (estimate = first minus second; temporal test first, then hold-out test):

| ID | Comparison | Role |
|---|---|---|
| C1 | M3 LightGBM full − M2 LR full | **primary model comparison** |
| C2 | M2 LR full − M1 LR RFM | value of the added features (linear) |
| C3 | M3 − B2 recency rule | gain over the simplest strong rule |
| C4 | M2 − B2 recency rule | gain over the simplest strong rule (linear) |
| C5 | M3 − B1 persistence | gain over persistence |
| S-a | M3 − S2 (with − without `prev_90d_active`) | sensitivity |
| S-b | M2 − S1 | sensitivity |

- **Temporal vs hold-out test (per model).** These sets have disjoint respondents, so the comparison is
  unpaired. Their AP difference is estimated from independent bootstraps of the two sets. It is descriptive
  only, and the hold-out has just 249 events.
- **Reading the intervals.**
  - A difference is described as supported when its 95 % interval excludes 0; otherwise "no clear
    difference".
  - No equivalence claims are made, and there is no multiplicity adjustment. C1 is the only comparison
    labelled primary; all others are secondary.
- **Not covered by these intervals:** uncertainty from the particular two test quarters (period-level shocks).
  This is stated as a limitation.

## 9. Does the split design support every comparison?

| Evaluation or comparison | Respondent overlap | Temporal leakage | Verdict |
|---|---|---|---|
| Selection: Train → Tune | same people by design (temporal); hold-out excluded | train labels end 2021-06-29 ≤ 2021-06-30 (asserted) | OK |
| Early stopping on Tune | — | uses tune labels only; no test or hold-out rows | OK (tune AP optimistic, §4.2) |
| Refit → temporal test | same people by design; 13 test respondents absent from Refit (new entrants), kept | refit labels end 2022-03-31 ≤ 2022-03-31 (asserted) | OK |
| Refit → hold-out test | **none** (asserted by `check_design`) | as above | OK |
| Phase-1 models → hold-out training period | none | same calendar quarters as Train; no future labels | OK (diagnostic) |
| Paired comparisons C1–C5, S-a, S-b | identical rows for both models | identical | OK |
| Temporal vs hold-out | disjoint sets | identical periods | OK, unpaired (§8) |
| B1 at 2022-03-31 uses the 2021-12-31 outcome, which is also a refit label | — | observed by 2022-03-31 (window ends that day) | OK: available at prediction time |

**No unresolved overlap or leakage problem was found.** Points to state in the report rather than to fix:
1. The prevalence of y = 0 falls between Train (15.1 %) and the test periods (12.7 % temporal, 12.4 %
   hold-out). With no recalibration, calibration-in-the-large is expected to show over-prediction of y = 0.
2. The hold-out test has 249 events: wide intervals.
3. Two test quarters (Q1 and Q2 of 2022) cover only part of the seasonal cycle.
4. Fixed refit iterations (§5).

## 10. Outputs (Phase 2)

- `stage_d_tune_results.csv`: full grids.
- `stage_d_locked_config.json`.
- `stage_d_test_metrics.csv`: model × set × cutoff × metric, with CIs.
- `stage_d_comparisons.csv`.
- `stage_d_bootstrap_draws.csv.gz`.
- `stage_d_predictions.csv.gz`: row-level, git-ignored.
- Reliability plots.
- `reports/stage_d_results.md`, generated from these files.

## 11. Decisions (recorded 2026-10-25)

1. **D-Q1.** Design kept as proposed; Phase 1 approved to run. Phase 2 needs a separate approval.
2. **D-Q2.** C1 (LightGBM vs full logistic regression) is the single primary comparison.
3. **D-Q3.** Two-phase gate kept: the lock is written and committed only after the Phase 1 results are
   approved.
4. **D-Q4.** The no-gift-card sensitivity analysis is excluded for now.

## 12. Change log

**v1 → v1.1 (2026-10-25, before any model was trained on the real data).**
- `prev_quarter_active` renamed to `prev_90d_active`; the calculation is unchanged.
- §2: Phase 1 data access made explicit. Labels are built only for the 13 pre-test cutoffs, and hold-out
  membership comes from IDs, not labels.
- §3: class-imbalance handling stated explicitly (no weighting or resampling).
- §4.2: metric-agreement and iteration-cap checks recorded.
- §4.4: the reproducibility comparison is defined.
- §6: the top-k estimand is defined exactly (per-cutoff ranking, ceiling, expected handling of ties, pooled
  counts). `top_fraction_by_group` was added because the code previously had only a single-set version, which
  did not match the per-cutoff definition.
- §7: Phase 1 writes a candidate selection; the lock follows approval.
- §11: decisions recorded.

No change to the question, target, features, splits, models, grids, selection rule or metrics.
