"""Model specifications for Stage D (protocol: reports/stage_d_protocol.md).

Every model is fitted on y (1 = purchase) and scored with `p_no_purchase` = P(y = 0).
All preprocessing is inside the estimator, so it is fitted on the fitting rows only.
"""
import itertools

import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .features import FEATURE_COLUMNS

SEED = 20261024
RFM_FEATURES = ["recency_days", "frequency_days", "monetary_total"]
FULL_FEATURES = list(FEATURE_COLUMNS)
NO_PREV_FEATURES = [f for f in FEATURE_COLUMNS if f != "prev_90d_active"]
FEATURE_SETS = {"rfm": RFM_FEATURES, "full": FULL_FEATURES, "no_prev": NO_PREV_FEATURES}

# logistic-regression preprocessing groups
_MISSING_CONT = ["median_gap_days", "recency_to_median_gap"]   # NaN when frequency_days = 1
_BINARY_MISSING = ["prev_90d_active"]                       # NaN when no prior history
_CATEGORICAL = ["cutoff_quarter"]
LR_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)

# LightGBM: fixed settings + a small grid; trees chosen by early stopping on the tune set
LGBM_FIXED = dict(objective="binary", learning_rate=0.03, n_estimators=3000, subsample=1.0,
                  colsample_bytree=1.0, reg_lambda=1.0, random_state=SEED, deterministic=True,
                  force_row_wise=True, n_jobs=4, verbose=-1)
LGBM_GRID = {"num_leaves": (7, 15, 31), "min_child_samples": (50, 200)}
EARLY_STOPPING_ROUNDS = 200


def make_logreg(features, C):
    features = list(features)
    log_cols = [f for f in features if f not in _MISSING_CONT + _BINARY_MISSING + _CATEGORICAL]
    blocks = []
    if log_cols:
        blocks.append(("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                        ("scale", StandardScaler())]), log_cols))
    for f in [x for x in _MISSING_CONT if x in features]:
        blocks.append((f, Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                    ("scale", StandardScaler())]), [f]))
    for f in [x for x in _BINARY_MISSING if x in features]:
        blocks.append((f, Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                    ("scale", StandardScaler())]), [f]))
    if "cutoff_quarter" in features:
        blocks.append(("quarter", OneHotEncoder(categories=[[1.0, 2.0, 3.0, 4.0]], drop="first",
                                                handle_unknown="error"), ["cutoff_quarter"]))
    return Pipeline([("prep", ColumnTransformer(blocks, remainder="drop")),
                     ("lr", LogisticRegression(C=C, max_iter=5000))])


def lgbm_configs():
    keys = list(LGBM_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(LGBM_GRID[k] for k in keys))]


def ap_no_purchase_eval(y_true, y_pred):
    """LightGBM eval metric: AP for y = 0. y_pred is a monotone score for y = 1 (probability with
    the built-in objective), so -y_pred ranks y = 0; AP is invariant to the monotone transform."""
    return "ap_no_purchase", float(average_precision_score(1 - np.asarray(y_true), -np.asarray(y_pred))), True


def fit_lgbm_early_stopping(X_fit, y_fit, X_tune, y_tune, config):
    m = lgb.LGBMClassifier(**LGBM_FIXED, **config, metric="None")
    m.fit(X_fit, y_fit, eval_X=(X_tune,), eval_y=(y_tune,), eval_metric=ap_no_purchase_eval,
          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)])
    return m


def make_lgbm_refit(config, n_estimators):
    """Final model: same settings, the iteration count selected on tune, no early stopping."""
    params = {**LGBM_FIXED, **config, "n_estimators": int(n_estimators)}
    return lgb.LGBMClassifier(**params)


def select(results, key="tune_ap_y0", tiebreak=()):
    """Highest tune AP for y = 0; exact ties broken by `tiebreak` (simpler model first)."""
    return sorted(results, key=lambda r: (-r[key], *[r[t] for t in tiebreak]))[0]
