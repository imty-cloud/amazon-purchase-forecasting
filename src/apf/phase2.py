"""Stage D Phase 2: refit with the locked configurations and evaluate the test sets once
(reports/stage_d_protocol.md §5-§8).

`main` checks the lock *before* any data is read or any test label is built. `run_phase2` is the
pure computation; it is tested on synthetic tables only.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import models as Mo
from .config import CUTOFFS, ID, OUTPUT_DIR, PURCHASES_CSV
from .evaluation import (PersistenceBaseline, PrevalenceBaseline, RecencyRule, event, metrics,
                         p_no_purchase, paired_cluster_bootstrap, percentile_ci, reliability_table,
                         top_fraction_by_group)
from .lock import LockError, release_test_rows, require_lock, rows_before_test, sha256
from .splits import REFIT_CUTOFFS, TEST_CUTOFFS, TRAIN_CUTOFFS

SETS = {"temporal_test": "in_test_temporal", "holdout_test": "in_test_holdout",
        "holdout_train_period": "in_holdout_train_period"}
COMPARISONS = [("C1", "M3_lgbm_full", "M2_lr_full", "primary"),
               ("C2", "M2_lr_full", "M1_lr_rfm", "secondary"),
               ("C3", "M3_lgbm_full", "B2_recency", "secondary"),
               ("C4", "M2_lr_full", "B2_recency", "secondary"),
               ("C5", "M3_lgbm_full", "B1_persistence", "secondary"),
               ("S-a", "M3_lgbm_full", "S2_lgbm_no_prev", "sensitivity"),
               ("S-b", "M2_lr_full", "S1_lr_no_prev", "sensitivity")]
RESULT_FILE = "stage_d_test_metrics.csv"


def _boot_fns():
    def ap(y, p, s):
        return average_precision_score(event(y), s)

    def auc(y, p, s):
        e = event(y)
        return roc_auc_score(e, s) if 0 < e.sum() < len(e) else np.nan

    def brier(y, p, s):
        return float(np.mean((event(y) - p) ** 2))
    return {"ap_y0": ap, "roc_auc": auc, "brier": brier}


class _Model:
    """Uniform wrapper: fit(frame, y), p0(frame), score0(frame)."""

    def __init__(self, kind, spec=None):
        self.kind, self.spec = kind, spec or {}

    def fit(self, F, y):
        k, s = self.kind, self.spec
        if k == "B0":
            self.m = PrevalenceBaseline().fit(F, y)
        elif k == "B1":
            self.m = PersistenceBaseline().fit(F, y)
        elif k == "B2":
            self.m = RecencyRule().fit(F, y)
        elif k == "lr":
            self.cols = Mo.FEATURE_SETS[s["feature_set"]]
            self.m = Mo.make_logreg(self.cols, s["C"]).fit(F, y)
        elif k == "lgbm":
            self.cols = Mo.FEATURE_SETS[s["feature_set"]]
            cfg = {"num_leaves": s["num_leaves"], "min_child_samples": s["min_child_samples"]}
            self.m = Mo.make_lgbm_refit(cfg, s["best_iteration"]).fit(F[self.cols], y)
        self.n_fit, self.p_fit = len(F), float(event(y).mean())
        return self

    def p0(self, F):
        if self.kind in ("B0", "B1", "B2"):
            return self.m.predict_p0(F)
        return p_no_purchase(self.m, F[self.cols] if self.kind == "lgbm" else F)

    def score0(self, F):
        return self.m.score0(F) if self.kind == "B2" else self.p0(F)


def model_specs(configurations):
    cf = configurations
    out = {"B0_prevalence": ("B0", None), "B1_persistence": ("B1", None), "B2_recency": ("B2", None)}
    for mid in ("M1_lr_rfm", "M2_lr_full", "S1_lr_no_prev"):
        out[mid] = ("lr", cf[mid])
    for mid in ("M3_lgbm_full", "S2_lgbm_no_prev"):
        out[mid] = ("lgbm", cf[mid])
    return out


def run_phase2(dev, test, configurations, B=1000, seed=20261024):
    """dev: rows before the first test cutoff; test: test-cutoff rows. Both carry the role columns
    from `splits.assign`. Returns a dict of result tables."""
    refit, train = dev[dev["in_refit"]], dev[dev["in_train"]]
    for name, fr in (("refit", refit), ("train", train)):
        assert not fr["holdout"].any(), f"hold-out respondent in {name}"
    assert set(refit["cutoff"]) <= set(REFIT_CUTOFFS) and set(train["cutoff"]) <= set(TRAIN_CUTOFFS)
    assert set(test["cutoff"]) <= set(TEST_CUTOFFS) and not set(dev["cutoff"]) & set(TEST_CUTOFFS)
    sets = {"temporal_test": test[test["in_test_temporal"]], "holdout_test": test[test["in_test_holdout"]],
            "holdout_train_period": dev[dev["in_holdout_train_period"]]}
    specs = model_specs(configurations)
    final = {m: _Model(*s).fit(refit, refit["y"].to_numpy()) for m, s in specs.items()}
    phase1 = {m: _Model(*s).fit(train, train["y"].to_numpy()) for m, s in specs.items()}  # diagnostic

    rows, rel, preds, draws_all, meta = [], [], [], {}, {"fit_rows": {}, "lightgbm_trees": {}}
    for set_name, S in sets.items():
        fitted = phase1 if set_name == "holdout_train_period" else final
        y, cut = S["y"].to_numpy(), S["cutoff"].to_numpy()
        P = {}
        for m, mod in fitted.items():
            p0, s0 = mod.p0(S), mod.score0(S)
            P[m] = (p0, s0)
            base = {"set": set_name, "model": m, "fit_rows": mod.n_fit}
            for scope, mask in [("pooled", np.ones(len(S), bool))] + [(str(pd.Timestamp(c).date()), cut == c)
                                                                    for c in sorted(set(cut))]:
                r = metrics(y[mask], p0[mask], s0[mask], train_prevalence0=mod.p_fit)
                for frac in (0.10, 0.20):
                    t = top_fraction_by_group(y[mask], s0[mask], cut[mask], frac)
                    r[f"precision_top{int(frac * 100)}"], r[f"recall_top{int(frac * 100)}"] = \
                        t["precision_y0"], t["recall_y0"]
                rows.append({**base, "scope": scope, **r})
            rt = reliability_table(y, p0).reset_index()
            rel.append(rt.assign(set=set_name, model=m))
            preds.append(pd.DataFrame({ID: S[ID].to_numpy(), "cutoff": cut, "set": set_name, "model": m,
                                       "y": y, "p_no_purchase": p0, "score0": s0}))
        d = paired_cluster_bootstrap(y, P, S[ID].to_numpy(), _boot_fns(), B=B, seed=seed)
        draws_all[set_name] = d.assign(set=set_name)
        meta["fit_rows"][set_name] = {m: mod.n_fit for m, mod in fitted.items()}
        meta["lightgbm_trees"][set_name] = {m: int(mod.m.booster_.current_iteration())
                                            for m, mod in fitted.items() if mod.kind == "lgbm"}

    met = pd.DataFrame(rows)
    draws = pd.concat(draws_all.values(), ignore_index=True)
    ci = (draws.groupby(["set", "model", "metric"])["value"]
          .apply(lambda v: pd.Series(percentile_ci(v.to_numpy()), index=["ci_low", "ci_high"]))
          .unstack().reset_index())
    comps = []
    for set_name in ("temporal_test", "holdout_test"):
        w = draws[draws["set"] == set_name].pivot_table(index=["draw", "metric"], columns="model", values="value")
        pooled = met[(met["set"] == set_name) & (met["scope"] == "pooled")].set_index("model")
        for cid, a, b, role in COMPARISONS:
            for metric in ("ap_y0", "roc_auc", "brier"):
                diff = (w[a] - w[b]).xs(metric, level="metric").to_numpy()
                lo, hi = percentile_ci(diff)
                comps.append({"set": set_name, "comparison": cid, "role": role, "model_a": a, "model_b": b,
                              "metric": metric, "estimate": float(pooled.loc[a, metric] - pooled.loc[b, metric]),
                              "ci_low": lo, "ci_high": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
    # temporal vs hold-out test, per model: unpaired (disjoint respondents, independent draws)
    tvh = []
    for m in specs:
        dt = draws[(draws["set"] == "temporal_test") & (draws["model"] == m) & (draws["metric"] == "ap_y0")]
        dh = draws[(draws["set"] == "holdout_test") & (draws["model"] == m) & (draws["metric"] == "ap_y0")]
        diff = dt.sort_values("draw")["value"].to_numpy() - dh.sort_values("draw")["value"].to_numpy()
        pt = met[(met["set"] == "temporal_test") & (met["scope"] == "pooled") & (met["model"] == m)]["ap_y0"].item()
        ph = met[(met["set"] == "holdout_test") & (met["scope"] == "pooled") & (met["model"] == m)]["ap_y0"].item()
        lo, hi = percentile_ci(diff)
        tvh.append({"model": m, "ap_temporal": pt, "ap_holdout": ph, "difference": pt - ph,
                    "ci_low": lo, "ci_high": hi})
    meta.update(B=B, seed=seed, rows={k: int(len(v)) for k, v in sets.items()},
                refit_cutoffs=sorted(str(pd.Timestamp(c).date()) for c in set(refit["cutoff"])))
    return {"metrics": met, "ci": ci, "comparisons": pd.DataFrame(comps),
            "temporal_vs_holdout": pd.DataFrame(tvh), "reliability": pd.concat(rel, ignore_index=True),
            "draws": draws, "predictions": pd.concat(preds, ignore_index=True), "meta": meta}


def load_table():
    """All cutoffs, features + labels + roles. Called only after the lock check in `main`."""
    from .data import load_transactions
    from .features import build_feature_panel, make_daily_store
    from .splits import assign, holdout_respondents
    from .target import build_panel, eligible_respondents, purchase_days
    tx, _ = load_transactions(PURCHASES_CSV)
    store, days = make_daily_store(tx), purchase_days(tx)
    lab = build_panel(days, CUTOFFS)[[ID, "cutoff", "y"]]
    tab = lab.merge(build_feature_panel(store, CUTOFFS), on=[ID, "cutoff"], validate="one_to_one")
    return assign(tab, holdout_respondents(eligible_respondents(days, CUTOFFS[-1])))


def main(out_dir=OUTPUT_DIR, B=1000, loader=None):
    payload = require_lock()                                   # 1. lock first, before any data
    sel_file = OUTPUT_DIR / payload["selected"]["phase1_selection_file"]
    if sha256(sel_file) != payload["selected"]["phase1_selection_sha256"]:
        raise LockError("Phase 1 selection file differs from the one recorded in the lock")
    if (out_dir / RESULT_FILE).exists():                       # 2. Phase 2 runs once
        raise LockError(f"{RESULT_FILE} already exists in {out_dir}: Phase 2 has already been run")
    table = (loader or load_table)()                           # 3. only now are test labels built
    dev = rows_before_test(table)
    test, payload = release_test_rows(table)                   # re-checks the lock
    res = run_phase2(dev, test, payload["selected"]["configurations"], B=B,
                     seed=payload["selected"]["seeds"]["bootstrap"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res["metrics"].to_csv(out_dir / RESULT_FILE, index=False)
    res["ci"].to_csv(out_dir / "stage_d_test_metric_cis.csv", index=False)
    res["comparisons"].to_csv(out_dir / "stage_d_comparisons.csv", index=False)
    res["temporal_vs_holdout"].to_csv(out_dir / "stage_d_temporal_vs_holdout.csv", index=False)
    res["reliability"].to_csv(out_dir / "stage_d_reliability.csv", index=False)
    res["draws"].to_csv(out_dir / "stage_d_bootstrap_draws.csv.gz", index=False, compression="gzip")
    res["predictions"].to_csv(out_dir / "stage_d_predictions.csv.gz", index=False, compression="gzip")
    (out_dir / "stage_d_phase2_meta.json").write_text(json.dumps(
        {**res["meta"], "lock_protocol_sha256": payload["protocol_sha256"],
         "lock_phase1_commit": payload["selected"]["phase1_commit"]}, indent=1, default=str))
    return res
