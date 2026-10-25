"""Writes reports/stage_d_results.md from the Phase 2 aggregate outputs (protocol v1.2 §6-§10).

Nothing is fitted, recomputed or tuned here. Every number in the report is read from a file in outputs/ (or
is a file checksum or a git commit) and is recorded with its source in outputs/stage_d_report_provenance.json;
scripts/check_stage_d_report.py re-reads every source independently.
Usage: .venv/bin/python scripts/build_stage_d_report.py
"""
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TM, CI, CMP, ST, TVH, REL = ("stage_d_test_metrics.csv", "stage_d_test_metric_cis.csv", "stage_d_comparisons.csv",
                             "stage_d_temporal_stability.csv", "stage_d_temporal_vs_holdout.csv",
                             "stage_d_reliability.csv")
META, LOCK = "stage_d_phase2_meta.json", "stage_d_locked_config.json"
CSV = {f: pd.read_csv(OUT / f) for f in (TM, CI, CMP, ST, TVH, REL)}
JSON = {f: json.loads((OUT / f).read_text()) for f in (META, LOCK)}
AGGREGATE = [TM, CI, CMP, ST, TVH, REL, META, "figures/stage_d_temporal_stability.png",
             "figures/stage_d_reliability.png"]
ROW_LEVEL = ["stage_d_predictions.csv.gz", "stage_d_bootstrap_draws.csv.gz",
             "stage_d_bootstrap_draws_by_cutoff.csv.gz"]

MODELS = ["B0_prevalence", "B1_persistence", "B2_recency", "M1_lr_rfm", "M2_lr_full", "S1_lr_no_prev",
          "M3_lgbm_full", "S2_lgbm_no_prev"]
STAB = ["B0_prevalence", "B1_persistence", "B2_recency", "M1_lr_rfm", "M2_lr_full", "M3_lgbm_full"]
SKILLED = STAB[1:]
NAME = {"B0_prevalence": "B0 prevalence", "B1_persistence": "B1 persistence", "B2_recency": "B2 recency rule",
        "M1_lr_rfm": "M1 LR, RFM features", "M2_lr_full": "M2 LR, full features (primary model)",
        "S1_lr_no_prev": "S1 LR, without prev_90d_active", "M3_lgbm_full": "M3 LightGBM, full features",
        "S2_lgbm_no_prev": "S2 LightGBM, without prev_90d_active"}
SETS = ["temporal_test", "holdout_test", "holdout_train_period"]
TEST_SETS = SETS[:2]
CUTS = ["2022-03-31", "2022-06-30"]
SET_WORD = {"temporal_test": "temporal", "holdout_test": "hold-out"}

REC = []                                            # provenance, one record per number in the report


def _put(rec, text):
    rec["text"] = text
    REC.append(rec)
    return f"\x00{len(REC) - 1}\x00"


def _fmt(x, fmt):
    return "n/a" if isinstance(x, float) and math.isnan(x) else format(x, fmt)


def row(file, **where):
    d = CSV[file]
    m = pd.Series(True, index=d.index)
    for k, val in where.items():
        m &= d[k].astype(str) == str(val)
    assert m.sum() == 1, (file, where, int(m.sum()))
    return d[m].iloc[0]


def v(file, column, fmt=".3f", **where):
    """One value from outputs/<file>: column `column` of the single row matching `where`."""
    x = row(file, **where)[column]
    x = x.item() if hasattr(x, "item") else x
    return _put({"source": "csv", "file": f"outputs/{file}", "where": where, "column": column, "fmt": fmt},
                _fmt(x, fmt))


def j(file, *path, fmt="s"):
    x = JSON[file]
    for p in path:
        x = x[p]
    return _put({"source": "json", "file": f"outputs/{file}", "path": list(path), "fmt": fmt}, _fmt(x, fmt))


def sha(path):
    return _put({"source": "sha256", "file": path}, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def git(ref):
    out = subprocess.run(["git", "rev-parse", ref], cwd=ROOT, capture_output=True, text=True, check=True)
    return _put({"source": "git", "ref": ref}, out.stdout.strip())


def lab(m):
    return f"{NAME[m]} (`{m}`)"


def ci(file, metric, fmt=".3f", **where):
    return f"[{v(file, 'ci_low', fmt, metric=metric, **where)}, {v(file, 'ci_high', fmt, metric=metric, **where)}]"


def pooled(s, m, col):
    return row(TM, set=s, model=m, scope="pooled")[col]


# ---- interval readings (protocol §8) -------------------------------------------------------------

def reading(r):
    if r["role"] == "primary":
        if str(r["bonf_ci_excludes_0"]) == "True":
            return "supported (95 % and Bonferroni intervals exclude 0)"
        if r["ci_excludes_0"]:
            return "supported at 95 %; Bonferroni interval includes 0"
        return "no clear difference (both intervals include 0)"
    if r["estimate"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0:
        return "identical scores (estimate and interval exactly 0)"
    return "supported (interval excludes 0)" if r["ci_excludes_0"] else "no clear difference (interval includes 0)"


def cmp_cells(s, cid, metric, bonf):
    k = dict(set=s, comparison=cid, metric=metric)
    fmt = "+.4f"
    out = (f"{v(CMP, 'estimate', fmt, **k)} | [{v(CMP, 'ci_low', fmt, **k)}, {v(CMP, 'ci_high', fmt, **k)}]")
    if bonf:
        out += f" | [{v(CMP, 'bonf_ci_low', fmt, **k)}, {v(CMP, 'bonf_ci_high', fmt, **k)}]"
    return out + f" | {reading(row(CMP, **k))}"


def comparison_table(ids, metrics, bonf):
    head = ("| ID | Comparison (first − second) | Set | Metric | Estimate | 95 % CI | "
            + ("Bonferroni 98.33 % CI | " if bonf else "") + "Reading |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    mname = {"ap_y0": "AP", "roc_auc": "ROC-AUC", "brier": "Brier"}
    for cid in ids:
        for metric in metrics:
            for s in TEST_SETS:
                r = row(CMP, set=s, comparison=cid, metric=metric)
                lines.append(f"| {cid} | `{r['model_a']}` − `{r['model_b']}` | {SET_WORD[s]} | {mname[metric]} | "
                             f"{cmp_cells(s, cid, metric, bonf)} |")
    return "\n".join(lines)


# ---- tables -------------------------------------------------------------------------------------

def sets_table():
    meta_fit = {"temporal_test": "final models, refit", "holdout_test": "final models, refit",
                "holdout_train_period": "Phase 1 models, fitted on Train"}
    label = {"temporal_test": "**Temporal test** (primary)", "holdout_test": "**Hold-out test**",
             "holdout_train_period": "Hold-out, training period (diagnostic)"}
    who = {"temporal_test": "non-hold-out respondents (their earlier rows are in the refit data)",
           "holdout_test": "hold-out respondents, never used for fitting or selection",
           "holdout_train_period": "hold-out respondents"}
    when = {"temporal_test": "2022-03-31, 2022-06-30", "holdout_test": "2022-03-31, 2022-06-30",
            "holdout_train_period": "2018-12-31 … 2021-03-31"}
    what = {"temporal_test": "later quarters for the same panel members",
            "holdout_test": "later quarters and new respondents jointly",
            "holdout_train_period": "new respondents in quarters already used for fitting; not a test of "
                                    "future performance"}
    lines = ["| Set | Respondents | Cutoffs | Rows | y = 0 | Prevalence of y = 0 | Models scored | "
             "Fitted on (rows) | Measures |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        k = dict(set=s, model="B0_prevalence", scope="pooled")
        lines.append(f"| {label[s]} | {who[s]} | {when[s]} | {j(META, 'rows', s, fmt=',d')} | "
                     f"{v(TM, 'n_event_y0', ',d', **k)} | {v(TM, 'prevalence_y0', **k)} | {meta_fit[s]} | "
                     f"{j(META, 'fit_rows', s, 'M2_lr_full', fmt=',d')} | {what[s]} |")
    return "\n".join(lines)


def pooled_tables(s):
    a = ["| Model | AP (y = 0) [95 % CI] | No-skill AP | ROC-AUC [95 % CI] | Brier [95 % CI] | Brier skill |",
         "|---|---|---|---|---|---|"]
    b = ["| Model | Calibration intercept | Calibration slope | Precision, top 10 % | Recall, top 10 % | "
         "Precision, top 20 % | Recall, top 20 % |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        k = dict(set=s, model=m, scope="pooled")
        a.append(f"| {lab(m)} | {v(TM, 'ap_y0', **k)} {ci(CI, 'ap_y0', set=s, model=m)} | "
                 f"{v(TM, 'ap_no_skill', **k)} | {v(TM, 'roc_auc', **k)} {ci(CI, 'roc_auc', set=s, model=m)} | "
                 f"{v(TM, 'brier', '.4f', **k)} {ci(CI, 'brier', '.4f', set=s, model=m)} | "
                 f"{v(TM, 'brier_skill', **k)} |")
        b.append(f"| {lab(m)} | {v(TM, 'cal_intercept', '+.3f', **k)} | {v(TM, 'cal_slope', **k)} | "
                 + " | ".join(v(TM, c, **k) for c in ("precision_top10", "recall_top10", "precision_top20",
                                                      "recall_top20")) + " |")
    return "\n".join(a), "\n".join(b)


def cutoff_tables(s):
    a = ["| Cutoff | Rows | y = 0 | Prevalence of y = 0 |", "|---|---|---|---|"]
    for c in CUTS:
        k = dict(set=s, cutoff=c, model="B0_prevalence")
        a.append(f"| {c} | {v(ST, 'n', ',d', **k)} | {v(ST, 'n_event_y0', ',d', **k)} | "
                 f"{v(ST, 'prevalence_y0', **k)} |")
    b = ["| Cutoff | Model | AP (y = 0) [95 % CI] | AP − prevalence | AP / prevalence | ROC-AUC | Brier | "
         "Calibration intercept | Calibration slope |", "|---|---|---|---|---|---|---|---|---|"]
    for c in CUTS:
        for m in STAB:
            k, t = dict(set=s, cutoff=c, model=m), dict(set=s, scope=c, model=m)
            b.append(f"| {c} | {lab(m)} | {v(ST, 'ap_y0', **k)} [{v(ST, 'ap_ci_low', **k)}, "
                     f"{v(ST, 'ap_ci_high', **k)}] | {v(ST, 'ap_minus_prevalence', '+.3f', **k)} | "
                     f"{v(ST, 'ap_lift', '.2f', **k)} | {v(TM, 'roc_auc', **t)} | {v(TM, 'brier', '.4f', **t)} | "
                     f"{v(ST, 'cal_intercept', '+.3f', **k)} | {v(TM, 'cal_slope', **t)} |")
    return "\n".join(a), "\n".join(b)


def calibration_table():
    head = ("| Model | Intercept, temporal test | Intercept, hold-out test | Intercept, training-period diagnostic | "
            "Slope, temporal test | Slope, hold-out test | Slope, training-period diagnostic |")
    lines = [head, "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        lines.append(f"| {lab(m)} | "
                     + " | ".join(v(TM, "cal_intercept", "+.3f", set=s, model=m, scope="pooled") for s in SETS) + " | "
                     + " | ".join(v(TM, "cal_slope", ".3f", set=s, model=m, scope="pooled") for s in SETS) + " |")
    return "\n".join(lines)


def reliability_table(s, m="M2_lr_full"):
    d = CSV[REL]
    lines = ["| Bin | Rows | Mean p0 | Observed share of y = 0 |", "|---|---|---|---|"]
    for b in sorted(d[(d["set"] == s) & (d["model"] == m)]["bin"]):
        k = dict(set=s, model=m, bin=b)
        lines.append(f"| {v(REL, 'bin', 'd', **k)} | {v(REL, 'n', ',d', **k)} | {v(REL, 'mean_p0', **k)} | "
                     f"{v(REL, 'observed_y0', **k)} |")
    return "\n".join(lines)


def tvh_table():
    lines = ["| Model | AP, temporal test | AP, hold-out test | Difference (temporal − hold-out) | 95 % CI |",
             "|---|---|---|---|---|"]
    for m in MODELS:
        lines.append(f"| {lab(m)} | {v(TVH, 'ap_temporal', model=m)} | {v(TVH, 'ap_holdout', model=m)} | "
                     f"{v(TVH, 'difference', '+.4f', model=m)} | [{v(TVH, 'ci_low', '+.4f', model=m)}, "
                     f"{v(TVH, 'ci_high', '+.4f', model=m)}] |")
    return "\n".join(lines)


def ignored(path):
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


def outputs_table():
    lines = ["| File | SHA-256 | Kind |", "|---|---|---|"]
    for f in AGGREGATE + ROW_LEVEL:
        p = f"outputs/{f}"
        assert ignored(p) == (f in ROW_LEVEL), p
        kind = "row-level, git-ignored" if f in ROW_LEVEL else "aggregate, version-controlled"
        lines.append(f"| `{p}` | `{sha(p)}` | {kind} |")
    return "\n".join(lines)


# ---- checks behind the descriptive statements -----------------------------------------------------

def ap_at(s, c, m):
    return row(ST, set=s, cutoff=c, model=m)["ap_y0"]


def short(m):
    return m.split("_")[0]


tm = CSV[TM]
nan_rows = tm[tm[["cal_slope", "cal_intercept"]].isna().any(axis=1)]
assert set(nan_rows["model"]) == {"B0_prevalence"} and len(nan_rows) == (tm["model"] == "B0_prevalence").sum()
assert tm.drop(columns=["cal_slope", "cal_intercept"]).notna().all().all()
assert CSV[CI][["ci_low", "ci_high"]].notna().all().all()
assert CSV[ST].drop(columns=["cal_intercept"]).notna().all().all()
assert CSV[CMP][["estimate", "ci_low", "ci_high"]].notna().all().all()
assert CSV[CMP].loc[CSV[CMP]["role"] == "primary", ["bonf_ci_low", "bonf_ci_high"]].notna().all().all()
assert CSV[CMP].loc[CSV[CMP]["role"] != "primary", ["bonf_ci_low", "bonf_ci_high"]].isna().all().all()
b0 = tm[tm["model"] == "B0_prevalence"]
assert (b0["roc_auc"] == 0.5).all() and (b0["brier_skill"].abs() < 1e-12).all()
sa = CSV[CMP][CSV[CMP]["comparison"] == "S-a"]
assert (sa[["estimate", "ci_low", "ci_high"]] == 0).all().all()

skilled_drop = all(ap_at(s, CUTS[1], m) < ap_at(s, CUTS[0], m) for s in TEST_SETS for m in SKILLED)
prev = {s: [row(ST, set=s, cutoff=c, model="B0_prevalence")["prevalence_y0"] for c in CUTS] for s in TEST_SETS}
assert skilled_drop and prev["temporal_test"][1] < prev["temporal_test"][0] and \
    prev["holdout_test"][1] > prev["holdout_test"][0]
orders = {(s, c): " > ".join(short(m) for m in sorted(STAB, key=lambda m: -ap_at(s, c, m)))
          for s in TEST_SETS for c in CUTS}
assert len(set(orders.values())) > 1
LEARNED, RULES = ["M1_lr_rfm", "M2_lr_full", "M3_lgbm_full"], ["B1_persistence", "B2_recency"]
assert all(pooled(s, m, "cal_intercept") > 0 for s in TEST_SETS for m in LEARNED)
assert all(pooled(s, m, "cal_intercept") < 0 for s in TEST_SETS for m in RULES)
assert all(abs(pooled("holdout_train_period", m, "cal_intercept")) < 0.05 for m in MODELS[1:])
assert all(pooled(s, m, "cal_slope") > 1 for s in SETS for m in MODELS[1:])
assert all(max(MODELS[1:], key=lambda m: pooled(s, m, "cal_slope")) in ("M3_lgbm_full", "S2_lgbm_no_prev")
           for s in SETS)
assert all(max(LEARNED, key=lambda m: pooled(s, m, "cal_intercept")) == "M1_lr_rfm" for s in TEST_SETS)
icpt = {(s, m): [row(ST, set=s, cutoff=c, model=m)["cal_intercept"] for c in CUTS] for s in TEST_SETS for m in SKILLED}
assert all(icpt["temporal_test", m][1] < icpt["temporal_test", m][0] and
           icpt["holdout_test", m][1] > icpt["holdout_test", m][0] for m in SKILLED)
assert (CSV[TVH]["ci_low"] < 0).all() and (CSV[TVH]["ci_high"] > 0).all()
assert all((r := row(CI, set=s, model="B0_prevalence", metric="roc_auc"))["ci_low"] == r["ci_high"] == 0.5
           for s in SETS)
assert all((tm[tm["set"] == s].groupby("scope")["fit_rows"].nunique() == 1).all() for s in SETS)
assert JSON[LOCK]["protocol_sha256"] == JSON[META]["lock_protocol_sha256"] == \
    hashlib.sha256((ROOT / "reports" / "stage_d_protocol.md").read_bytes()).hexdigest()


def flags(s, cid, metric="ap_y0"):
    r = row(CMP, set=s, comparison=cid, metric=metric)
    return bool(r["ci_excludes_0"]), str(r["bonf_ci_excludes_0"]) == "True", r["estimate"]


# the fixed sentences in interpretation() hold only for these interval flags and signs
for cid in ("P1", "P3"):
    assert flags("temporal_test", cid)[:2] == (True, True) and flags("holdout_test", cid)[:2] == (True, False)
    assert all(flags(s, cid)[2] > 0 for s in TEST_SETS)
assert all(flags(s, c)[0] and flags(s, c)[2] > 0 for s in TEST_SETS for c in ("C3", "C5"))
assert not any(flags(s, "S-b")[0] for s in TEST_SETS)
p2 = {s: row(CMP, set=s, comparison="P2", metric="ap_y0") for s in TEST_SETS}
P2_INCONCLUSIVE = not any(r["ci_excludes_0"] for r in p2.values())
assert flags("temporal_test", "P2", "roc_auc")[:2] == (True, True)
assert flags("temporal_test", "P2", "brier")[:2] == (True, False)
rel = CSV[REL]
small = rel[rel["set"].isin(TEST_SETS) & (rel["n"] < 30)]
assert set(small["model"]) == {"B1_persistence"} and (small["bin"] == 1).all()
assert (rel[rel["model"] == "B1_persistence"].groupby("set")["bin"].size() == 3).all()
assert (small.loc[small["set"] == "holdout_test", "observed_y0"]
        - small.loc[small["set"] == "holdout_test", "mean_p0"]).abs().min() > 0.3


def cmp_text(s, cid, metric="ap_y0", bonf=True):
    k = dict(set=s, comparison=cid, metric=metric)
    t = (f"{v(CMP, 'estimate', '+.4f', **k)} (95 % CI [{v(CMP, 'ci_low', '+.4f', **k)}, "
         f"{v(CMP, 'ci_high', '+.4f', **k)}]")
    if bonf:
        t += f"; Bonferroni 98.33 % CI [{v(CMP, 'bonf_ci_low', '+.4f', **k)}, {v(CMP, 'bonf_ci_high', '+.4f', **k)}]"
    return t + f"): {reading(row(CMP, **k))}"


def summary():
    def both(cid):
        return (f"temporal test {cmp_text('temporal_test', cid)}; hold-out test {cmp_text('holdout_test', cid)}")
    k = dict(set="temporal_test", model="B0_prevalence", scope="pooled")
    out = [f"- **Primary evaluation set:** the temporal test ({j(META, 'rows', 'temporal_test', fmt=',d')} rows, "
           f"{v(TM, 'n_event_y0', ',d', **k)} with y = 0, prevalence {v(TM, 'prevalence_y0', **k)}).",
           f"- **P1, added features (M2 LR full − M1 LR RFM), AP for y = 0:** {both('P1')}.",
           f"- **P2, non-linear model (M3 LightGBM − M2 LR full), AP for y = 0:** {both('P2')}."]
    if P2_INCONCLUSIVE:
        out.append("  The primary AP comparison does not distinguish LightGBM from logistic regression, so this "
                   "report does not rank LightGBM above logistic regression (and does not claim they are equivalent).")
    out += [f"- **P3, gain over the strongest simple rule (M2 LR full − B2 recency), AP for y = 0:** {both('P3')}.",
            "- **Temporal stability is not established.** For every model with ranking skill (B1, B2, M1, M2, M3) "
            "AP is lower at 2022-06-30 than at 2022-03-31 in both test sets; no test of this change was "
            "pre-specified, and two adjacent quarters cannot establish long-term stability (protocol §6).",
            f"- **Calibration:** the learned models under-predict y = 0 at the test cutoffs (pooled "
            f"calibration-in-the-large for M2: {v(TM, 'cal_intercept', '+.3f', set='temporal_test', model='M2_lr_full', scope='pooled')} "
            f"temporal, {v(TM, 'cal_intercept', '+.3f', set='holdout_test', model='M2_lr_full', scope='pooled')} "
            f"hold-out; {v(TM, 'cal_intercept', '+.3f', set='holdout_train_period', model='M2_lr_full', scope='pooled')} "
            "in the training-period diagnostic). No recalibration was applied."]
    return "\n".join(out)


def interpretation():
    t = dict(set="temporal_test")
    p2a = dict(comparison="P2", **t)
    lines = [
        "- **Added features (P1).** On the temporal test, M2's AP exceeds M1's with both the 95 % and the "
        "Bonferroni interval excluding 0. On the hold-out test the difference has the same sign and its 95 % "
        "interval excludes 0, but its Bonferroni interval includes 0.",
    ]
    if P2_INCONCLUSIVE:
        lines.append(
            "- **LightGBM vs logistic regression (P2).** No clear AP difference in either test set. This is not "
            "evidence that the two models perform the same. On the temporal test M3's ROC-AUC is higher by "
            f"{v(CMP, 'estimate', '+.4f', metric='roc_auc', **p2a)} (both intervals exclude 0) and its Brier score "
            f"difference is {v(CMP, 'estimate', '+.4f', metric='brier', **p2a)}, lower for M3 (95 % interval excludes 0, Bonferroni "
            "interval does not). These are small differences in secondary metrics; because the primary AP "
            "comparison is inconclusive, LightGBM is not reported as the better model.")
    lines += [
        "- **Gain over the recency rule (P3).** On the temporal test, M2's AP exceeds B2's with both intervals "
        "excluding 0. On the hold-out test the 95 % interval excludes 0 but the Bonferroni interval includes 0.",
        "- **Secondary comparisons (C3, C5; unadjusted).** M3's AP exceeds B2's and B1's in both test sets "
        "(95 % intervals exclude 0).",
        "- **Sensitivity (S-a, S-b).** Removing `prev_90d_active` leaves LightGBM's scores unchanged (S-a is "
        "exactly 0) and makes no clear difference for logistic regression (S-b intervals include 0).",
        "- **Seen vs unseen respondents.** The temporal-test and hold-out-test AP differ by amounts whose "
        "intervals all include 0 (§6). The hold-out intervals are wide, so this does not show that performance "
        "is the same for new respondents.",
        "- **Temporal stability.** Not established (§4). The results describe two adjacent quarters only.",
        "- **Calibration.** At the test cutoffs the probabilities of M1, M2 and M3 are too low on average for "
        "y = 0, while their Phase 1 versions are close to calibrated in the training-period diagnostic (§5). "
        "Probabilities should not "
        "be read as accurate risks for these quarters without recalibration, which the protocol excludes from "
        "this analysis.",
        f"- **Operating point.** Flagging the top 20 % of each test cutoff by M2's score gives precision "
        f"{v(TM, 'precision_top20', set='temporal_test', model='M2_lr_full', scope='pooled')} and recall "
        f"{v(TM, 'recall_top20', set='temporal_test', model='M2_lr_full', scope='pooled')} for y = 0 on the "
        "temporal test (no intervals).",
    ]
    return "\n".join(lines)


def render(template):
    out, pos, n = [], 0, 0
    for m in re.finditer(r"\x00(\d+)\x00", template):
        lit = template[pos:m.start()]
        out.append(lit)
        n += len(lit)
        r = REC[int(m.group(1))]
        assert "offset" not in r
        r["offset"] = n
        out.append(r["text"])
        n += len(r["text"])
        pos = m.end()
    out.append(template[pos:])
    assert all("offset" in r for r in REC)
    return "".join(out)


# ---- report -----------------------------------------------------------------------------------

L = JSON[LOCK]["selected"]
tables = {s: pooled_tables(s) for s in SETS}
cut_tables = {s: cutoff_tables(s) for s in TEST_SETS}
small_txt = "; ".join(
    f"{NAME[r.model]}, {SET_WORD[r.set]} test: {v(REL, 'n', ',d', set=r.set, model=r.model, bin=r.bin)} rows "
    f"(mean p0 {v(REL, 'mean_p0', set=r.set, model=r.model, bin=r.bin)}, observed "
    f"{v(REL, 'observed_y0', set=r.set, model=r.model, bin=r.bin)})" for r in small.itertuples())

def ho_events():
    return v(TM, "n_event_y0", ",d", set="holdout_test", model="B0_prevalence", scope="pooled")


md = f"""# Stage D — Phase 2 test results

**Run.** Phase 2 was run once, on 2026-10-25, at commit `{git('stage-d-lock-v1.2^{commit}')}` (tag `stage-d-lock-v1.2`).
- **Protocol:** `reports/stage_d_protocol.md` v1.2, SHA-256 `{sha('reports/stage_d_protocol.md')}`. The lock
  records `{j(LOCK, 'protocol_sha256')}`, and so does the Phase 2 metadata (`{j(META, 'lock_protocol_sha256')}`).
- **Lock:** `outputs/stage_d_locked_config.json`, SHA-256 `{sha('outputs/stage_d_locked_config.json')}`. It was
  written at commit `{j(LOCK, 'selected', 'head_at_lock')}` and supersedes the lock at tag
  `{j(LOCK, 'selected', 'supersedes', 'tag')}`.
- **Data:** MD5 `{j(LOCK, 'selected', 'data_md5')}`.
- **Uncertainty:** paired respondent-cluster bootstrap with B = {j(META, 'B', fmt=',d')} resamples and seed
  {j(META, 'seed', fmt='d')} (protocol §8).
- **Generated by** `scripts/build_stage_d_report.py` from the aggregate Phase 2 files in `outputs/` (§10). Each
  number's source is recorded in `outputs/stage_d_report_provenance.json` and checked by
  `scripts/check_stage_d_report.py`.

**Scope.** This report reads the Phase 2 outputs only. It fits no model and computes no new test metric. It
involves no tuning, threshold choice or recalibration on test data.

## Summary

{summary()}

## 1. Target and evaluation sets

**Target.** For a respondent and a quarter-end cutoff c, y = 1 if at least one purchase day is recorded in this
dataset in days c+1 … c+90, and y = 0 otherwise.
- **The event of interest is y = 0.** It means that **no purchase day was recorded in this dataset during the
  next 90 days**. It is not confirmed churn.
- **What y = 0 cannot rule out:** purchases through other accounts or retailers, or orders the export does not
  include. Stage B §8 found that most y = 0 observations are followed by a later recorded purchase.
- **Respondent identifier:** `Survey ResponseID` identifies a survey respondent, not a verified Amazon customer.
- **Class convention:** every model outputs p0 = P(y = 0). AP, ROC-AUC, Brier score, calibration and top-k
  precision and recall all treat y = 0 as the positive class (protocol §1).

### Evaluation sets, pooled over their cutoffs (protocol §2)

The three sets are kept separate throughout.

{sets_table()}

The final models were refit on non-hold-out rows from the cutoffs {j(META, 'refit_cutoffs', 0)} …
{j(META, 'refit_cutoffs', -1)} (protocol §5). The LightGBM refits have
{j(META, 'lightgbm_trees', 'temporal_test', 'M3_lgbm_full', fmt='d')} trees (M3) and
{j(META, 'lightgbm_trees', 'temporal_test', 'S2_lgbm_no_prev', fmt='d')} trees (S2), the locked best
iterations.
- **Primary model and baseline:** the primary model is M2 and the strongest baseline is B2. Both were fixed
  before testing (protocol §8).
- **Pooling:** pooled results combine both test cutoffs. A respondent contributes up to two rows to a test set,
  and the bootstrap resamples respondents together with all their rows.

## 2. Pooled results

Notes for the tables below:
- **AP** is average precision for y = 0. Its no-skill value is the prevalence of y = 0 in the set.
- **Brier skill** is measured against the prevalence of y = 0 in each model's fitting set.
- **Intervals** are 95 % percentile intervals from the paired respondent-cluster bootstrap.
- **B2's ranking score** is `recency_days`. Its probabilities, used for Brier score and calibration, come from an
  isotonic map fitted on the fitting rows.
- **Top-k:** within each cutoff, the ⌈k · n_c⌉ highest scores are flagged, with ties shared in expectation;
  counts are then pooled over cutoffs (protocol §6). Top-k results and calibration statistics have no intervals.
- **Calibration intercept** (calibration-in-the-large): > 0 means p0 is too low on average, so y = 0 is
  under-predicted. A slope > 1 means p0 is too moderate.

### 2.1 Temporal test (primary): pooled over both test cutoffs

{tables['temporal_test'][0]}

{tables['temporal_test'][1]}

### 2.2 Hold-out test (unseen respondents): pooled over both test cutoffs

{tables['holdout_test'][0]}

{tables['holdout_test'][1]}

### 2.3 Hold-out, training period (diagnostic): pooled over the training cutoffs

This set uses the Phase 1 models, fitted on Train, scored on hold-out respondents at the training cutoffs. It
measures generalisation to new respondents without the shift in time. It is not an estimate of future
performance, and its numbers should not be compared with the test sets as if they were.

{tables['holdout_train_period'][0]}

{tables['holdout_train_period'][1]}

## 3. Model comparisons (protocol §8)

- **Estimate** = first model minus second model, on the pooled test set.
- **Intervals** come from the paired bootstrap: every model is scored on the same resampled respondents.
- **Bonferroni:** the 98.33 % intervals adjust for the three primary comparisons within each test set. The
  secondary and sensitivity comparisons are unadjusted.
- **Readings** follow protocol §8. "Supported" means the interval excludes 0. "No clear difference" means it
  includes 0; that is not evidence of equivalence.
- **Direction:** for AP and ROC-AUC a positive estimate favours the first model; for Brier score a negative
  estimate favours the first model.

### 3.1 Primary comparisons: AP for y = 0 (primary metric)

{comparison_table(['P1', 'P2', 'P3'], ['ap_y0'], bonf=True)}

### 3.2 Primary comparisons: ROC-AUC and Brier score (secondary metrics)

{comparison_table(['P1', 'P2', 'P3'], ['roc_auc', 'brier'], bonf=True)}

### 3.3 Secondary and sensitivity comparisons (95 % intervals, unadjusted)

{comparison_table(['C3', 'C5', 'S-a', 'S-b'], ['ap_y0', 'roc_auc', 'brier'], bonf=False)}

## 4. Temporal stability by test cutoff (protocol §6)

- **AP intervals** come from respondents resampled within each cutoff.
- **ROC-AUC, Brier score and the calibration slope** are point estimates without intervals.
- **Calibration intercept:** > 0 means y = 0 is under-predicted.

### 4.1 Temporal test (primary)

{cut_tables['temporal_test'][0]}

{cut_tables['temporal_test'][1]}

### 4.2 Hold-out test

{cut_tables['holdout_test'][0]}

{cut_tables['holdout_test'][1]}

![AP per test cutoff with no-skill line, and calibration-in-the-large per cutoff](../outputs/figures/stage_d_temporal_stability.png)

**What the per-cutoff results show** (descriptive):
- **AP falls between the cutoffs.** For every model with ranking skill (B1, B2, M1, M2, M3), AP is lower at
  2022-06-30 than at 2022-03-31 in both test sets. B0's AP equals the prevalence, which fell between the two
  cutoffs in the temporal test and rose in the hold-out test.
- **The fall is not tested.** No interval for the change between cutoffs was pre-specified, so these changes
  are descriptive. Overlapping per-cutoff intervals are not a test of no change.
- **Model ordering changes between cells.** The ordering by AP point estimate is not the same in all four
  set × cutoff cells:
  - temporal test, 2022-03-31: {orders[('temporal_test', CUTS[0])]};
  - temporal test, 2022-06-30: {orders[('temporal_test', CUTS[1])]};
  - hold-out test, 2022-03-31: {orders[('holdout_test', CUTS[0])]};
  - hold-out test, 2022-06-30: {orders[('holdout_test', CUTS[1])]}.

  At this interval width, orderings within a cutoff are not reliable.
- **Calibration-in-the-large changes between cutoffs in opposite directions in the two sets.** From
  2022-03-31 to 2022-06-30, the intercept of every model with ranking skill decreases in the temporal test and
  increases in the hold-out test.
- **Stability cannot be established.** Two adjacent quarters (Q1 and Q2 2022) cannot establish long-term
  temporal stability (protocol §6). The bootstrap does not cover period-level shocks (protocol §8).

## 5. Calibration (protocol §6)

### 5.1 Pooled calibration by set

{calibration_table()}

- **Test sets:** M1, M2 and M3 have positive intercepts on both test sets, so p0 is too low on average and y = 0
  is under-predicted. M1 is furthest from 0. B1 and B2 have negative intercepts, so they slightly over-predict
  y = 0.
- **Training-period diagnostic:** every defined intercept of the Phase 1 models is close to 0. The
  under-prediction is therefore seen at the test cutoffs but not for new respondents within the training
  period. The two results come from different fits (the refit models vs the Phase 1 models), and the cause is
  not established here.
- **Slopes:** every defined pooled slope is above 1, so p0 is too moderate; M3 and S2 have the largest pooled slopes in
  each set. Per-cutoff slopes vary, and some are below 1 (§4).
- **No recalibration** was applied (protocol §6).

### 5.2 Reliability of the primary model (M2), deciles of p0

#### Temporal test, `M2_lr_full`

{reliability_table('temporal_test')}

#### Hold-out test, `M2_lr_full`

{reliability_table('holdout_test')}

![Reliability, test sets](../outputs/figures/stage_d_reliability.png)

Reliability tables for every model and set are in `outputs/stage_d_reliability.csv`. Bins are formed by value,
and tied predictions share a bin. B1 persistence takes only three values, so it has three bins, and its middle
bin is very small at the test cutoffs: {small_txt}. The large gap that bin shows in the hold-out panel of the
figure rests on these few rows.

## 6. Temporal test vs hold-out test (descriptive, protocol §8)

The two sets have disjoint respondents, so the AP difference is unpaired: it comes from independent bootstraps
of the two sets. The hold-out test has {ho_events()} y = 0 rows.

{tvh_table()}

Every interval includes 0. These intervals are wide, and they do not show that performance is the same for seen
and unseen respondents.

## 7. Undefined and degenerate results

- **B0 calibration (undefined):** B0 predicts a constant, so its calibration slope and intercept are undefined
  ("n/a") in every set and cutoff. No other metric is undefined.
- **B0 ROC-AUC and Brier skill (degenerate):** B0's ROC-AUC is
  {v(TM, 'roc_auc', set='temporal_test', model='B0_prevalence', scope='pooled')} with interval
  {ci(CI, 'roc_auc', set='temporal_test', model='B0_prevalence')} on the temporal test, and the same in every
  set and cutoff. Its Brier skill is 0 by construction, because its reference is its own fitting-set prevalence.
- **S-a (degenerate):** M3 and S2 give identical scores. For AP, ROC-AUC and Brier score, the S-a estimate and
  both interval bounds are exactly 0 in both test sets. This agrees with Phase 1, where the selected LightGBM
  model made no split on `prev_90d_active` (`reports/stage_d_phase1_results.md`).
- **Bonferroni intervals** are computed for the primary comparisons only, as the protocol specifies.
- **No interval bound** in `stage_d_test_metric_cis.csv`, `stage_d_temporal_stability.csv` or
  `stage_d_comparisons.csv` is undefined.
- **Reported without intervals** (by protocol design): calibration slope and intercept, top-k precision and
  recall, per-cutoff ROC-AUC and Brier score, and reliability bins.

## 8. What the results support

{interpretation()}

## 9. Limitations

- **Label meaning.** y = 0 is the absence of a recorded purchase day in this export over 90 days. It is not
  churn, and it does not rule out purchases elsewhere. Per-respondent export end dates are not available.
  Every target window ends before the earliest possible survey date (protocol §1).
- **Test period.** There are only two adjacent test quarters. The intervals do not cover period-level shocks,
  and no claim of long-term temporal stability is made.
- **Respondents.** The temporal-test respondents also contributed earlier rows to the refit. The temporal test
  measures later quarters for the same panel members, not new people.
- **Hold-out size.** The hold-out test has {ho_events()} y = 0 rows, so its intervals are wide. The
  training-period diagnostic uses Phase 1 models and earlier quarters, and is not a performance estimate for
  the test period.
- **Multiplicity.** The Bonferroni adjustment covers only the three primary comparisons within each test set.
  Secondary metrics and the secondary and sensitivity comparisons are unadjusted, and many metrics are
  reported.
- **Unquantified uncertainty.** Calibration statistics, top-k results and per-cutoff ROC-AUC and Brier scores
  have no intervals.
- **Calibration.** The learned models' probabilities under-state the chance of y = 0 at the test cutoffs, and
  this was not corrected (protocol §6).
- **Sample.** Respondents are survey participants who shared their purchase histories, and they may not
  represent all Amazon customers. The primary data version keeps gift cards and exact duplicates (protocol §1).
- **Single run.** Phase 2 ran once, and nothing in it was used for tuning or selection.

## 10. Outputs (protocol §10)

| Protocol §10 item | File | Status |
|---|---|---|
| Full tuning grids | `outputs/stage_d_tune_results.csv` | present (Phase 1) |
| Locked configuration | `outputs/stage_d_locked_config.json` | present (lock v1.2) |
| Metrics: model × set × cutoff | `outputs/stage_d_test_metrics.csv` | present; the intervals are in `stage_d_test_metric_cis.csv` (pooled AP, ROC-AUC, Brier score) and `stage_d_temporal_stability.csv` (per-cutoff AP), not in this file |
| P1–P3 with 95 % and Bonferroni 98.33 % intervals; secondary and sensitivity comparisons with 95 % intervals | `outputs/stage_d_comparisons.csv` | present |
| Temporal stability per test set and cutoff (§6) | `outputs/stage_d_temporal_stability.csv` | present |
| Figures | `outputs/figures/stage_d_temporal_stability.png`, `outputs/figures/stage_d_reliability.png` | present |
| Bootstrap draws | `outputs/stage_d_bootstrap_draws.csv.gz` (also `stage_d_bootstrap_draws_by_cutoff.csv.gz`) | present, row-level, git-ignored |
| Predictions | `outputs/stage_d_predictions.csv.gz` | present, row-level, git-ignored |
| Reliability plots | `outputs/figures/stage_d_reliability.png`; tables in `outputs/stage_d_reliability.csv` | present |
| This report | `reports/stage_d_results.md` | this file |

Phase 2 also wrote `outputs/stage_d_temporal_vs_holdout.csv` (§6 of this report) and
`outputs/stage_d_phase2_meta.json` (fit rows, LightGBM trees, B, seed, lock hash).

**Checksums of the Phase 2 outputs:**

{outputs_table()}

## 11. Reproducing this report

```bash
.venv/bin/python scripts/build_stage_d_report.py    # rewrites this file and the provenance record
.venv/bin/python scripts/check_stage_d_report.py    # re-reads every number from outputs/ and checks it
```
"""

text = render(md)
(ROOT / "reports" / "stage_d_results.md").write_text(text)
(OUT / "stage_d_report_provenance.json").write_text(json.dumps(REC, indent=1, default=str))
print(f"wrote reports/stage_d_results.md ({len(text):,} characters, {len(REC)} numbers with provenance)")
