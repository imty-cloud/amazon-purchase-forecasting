"""Candidate validation design (Stage A §6, Stage C): temporal roles + a 20 % respondent hold-out.

Temporal roles by cutoff:
  train   2018-12-31 .. 2021-03-31   fit models for tuning
  tune    2021-06-30, 2021-09-30     choose hyperparameters / thresholds
  refit   train + tune + 2021-12-31  final fit before testing
  test    2022-03-31, 2022-06-30     evaluated once
Hold-out: 20 % of respondents (seeded), never used in train, tune or refit.
"""
import numpy as np
import pandas as pd

from .config import CUTOFFS, HORIZON_DAYS, ID

TRAIN_CUTOFFS = CUTOFFS[:10]
TUNE_CUTOFFS = CUTOFFS[10:12]
REFIT_CUTOFFS = CUTOFFS[:13]
TEST_CUTOFFS = CUTOFFS[13:15]
HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 20261024


def holdout_respondents(ids, fraction=HOLDOUT_FRACTION, seed=HOLDOUT_SEED):
    """Deterministic: depends only on the sorted set of IDs, the fraction and the seed."""
    u = np.sort(pd.unique(np.asarray(ids, dtype=str)))
    k = int(round(fraction * len(u)))
    pick = np.random.default_rng(seed).permutation(len(u))[:k]
    return pd.Index(np.sort(u[pick]), name=ID)


def label_window_end(c):
    return pd.Timestamp(c) + pd.Timedelta(days=HORIZON_DAYS)


def check_embargo(fit_cutoffs, eval_cutoffs):
    """Every label used for fitting must be observed by the earliest evaluation cutoff."""
    return label_window_end(max(fit_cutoffs)) <= min(eval_cutoffs)


def assign(panel, holdout_ids):
    """Adds `holdout` (bool) and boolean role columns. Hold-out respondents get no fitting role."""
    p = panel.copy()
    p["holdout"] = p[ID].isin(holdout_ids)
    fit_ok = ~p["holdout"]
    p["in_train"] = p["cutoff"].isin(TRAIN_CUTOFFS) & fit_ok
    p["in_tune"] = p["cutoff"].isin(TUNE_CUTOFFS) & fit_ok
    p["in_refit"] = p["cutoff"].isin(REFIT_CUTOFFS) & fit_ok
    p["in_test_temporal"] = p["cutoff"].isin(TEST_CUTOFFS) & fit_ok        # seen respondents
    p["in_test_holdout"] = p["cutoff"].isin(TEST_CUTOFFS) & p["holdout"]   # unseen respondents
    p["in_holdout_train_period"] = p["cutoff"].isin(TRAIN_CUTOFFS) & p["holdout"]  # diagnostic
    return p


def check_design(p):
    """Raises if any respondent of the hold-out reaches a fitting role or roles overlap wrongly."""
    fit = p["in_train"] | p["in_tune"] | p["in_refit"]
    if (fit & p["holdout"]).any():
        raise AssertionError("hold-out respondent used for fitting")
    if set(p.loc[p["holdout"], ID]) & set(p.loc[fit, ID]):
        raise AssertionError("hold-out respondent shares an ID with fitting rows")
    if (p["in_train"] & p["in_tune"]).any() or ((p["in_train"] | p["in_tune"] | p["in_refit"])
                                               & (p["in_test_temporal"] | p["in_test_holdout"])).any():
        raise AssertionError("a row has a fitting role and an evaluation role")
    if not (check_embargo(TRAIN_CUTOFFS, TUNE_CUTOFFS) and check_embargo(REFIT_CUTOFFS, TEST_CUTOFFS)):
        raise AssertionError("embargo violated")
