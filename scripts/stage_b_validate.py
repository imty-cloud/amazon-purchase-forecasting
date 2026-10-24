"""Stage B: data and target validation. Reads the raw CSV in place; writes only to outputs/.

Usage: .venv/bin/python scripts/stage_b_validate.py
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from apf import config as C                                  # noqa: E402
from apf.data import data_version, load_transactions, read_survey_ids, verify_file  # noqa: E402
from apf.target import build_panel, eligible_respondents, labels_at, purchase_days  # noqa: E402

OUT = C.OUTPUT_DIR
ID, DATE = C.ID, C.DATE

# Counts reported in reports/audit_and_feasibility.md §5 (Stage A scan, raw data).
STAGE_A = {  # cutoff: (n, y=1, y=0)
    "2018-12-31": (4281, 3558, 723), "2019-03-31": (4378, 3563, 815),
    "2019-06-30": (4469, 3675, 794), "2019-09-30": (4550, 3837, 713),
    "2019-12-31": (4669, 3852, 817), "2020-03-31": (4728, 4056, 672),
    "2020-06-30": (4796, 4089, 707), "2020-09-30": (4842, 4279, 563),
    "2020-12-31": (4882, 4279, 603), "2021-03-31": (4915, 4340, 575),
    "2021-06-30": (4932, 4287, 645), "2021-09-30": (4956, 4432, 524),
    "2021-12-31": (4991, 4364, 627), "2022-03-31": (5001, 4359, 642),
    "2022-06-30": (5008, 4384, 624)}
STAGE_A_TOTAL = (71398, 61354, 10044)
STAGE_A_QC = {"rows": 1850717, "respondents": 5027, "exact_duplicate_extra_copies": 11624,
              "gift_rows": 41483, "missing_Title": 89740, "missing_Category": 89458,
              "missing_Shipping Address State": 87812, "missing_ASIN/ISBN (Product Code)": 973}
# Proposed (not yet approved) validation roles, reports/audit_and_feasibility.md §6.
PROPOSED_ROLES = {"train": C.CUTOFFS[:10], "validation": C.CUTOFFS[10:12],
                  "test": C.CUTOFFS[13:15]}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ds(t):
    return str(pd.Timestamp(t).date())


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    S = {"run_started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "data_path": str(C.PURCHASES_CSV), "config": {
             "START": ds(C.START), "OBSERVED_THROUGH": ds(C.OBSERVED_THROUGH),
             "HORIZON_DAYS": C.HORIZON_DAYS, "CUTOFFS": [ds(c) for c in C.CUTOFFS]}}

    # 1. file integrity
    S["files"] = [verify_file(C.PURCHASES_CSV), verify_file(C.SURVEY_CSV)]
    assert all(f["md5_match"] for f in S["files"]), S["files"]

    # 2. chunked load + raw-text quality counts
    tx, qc = load_transactions(C.PURCHASES_CSV)
    S["qc"] = qc
    S["memory_mb_reduced_table"] = round(tx.memory_usage(deep=True).sum() / 2**20, 1)

    # 3. identifiers
    sv = read_survey_ids(C.SURVEY_CSV)
    tx_ids = set(tx[ID].astype(str).unique())
    S["identifiers"] = {
        "survey_rows": int(len(sv)), "survey_unique_ids": int(sv.nunique()),
        "survey_duplicate_ids": int(sv.duplicated().sum()),
        "purchase_ids_not_in_survey": len(tx_ids - set(sv)),
        "survey_ids_without_purchases": len(set(sv) - tx_ids),
        "note": "survey.csv: only the ID column is read, for reconciliation; no survey variable is used"}

    # 4. date coverage
    d = tx[DATE]
    in_obs = tx[(d >= C.START) & (d <= C.OBSERVED_THROUGH)]
    daily = in_obs.groupby(DATE).size().reindex(
        pd.date_range(C.START, C.OBSERVED_THROUGH, freq="D"), fill_value=0)
    last = tx.groupby(ID, observed=True)[DATE].max()
    first = tx.groupby(ID, observed=True)[DATE].min()
    monthly = tx.groupby(d.dt.to_period("M")).agg(rows=(ID, "size"), respondents=(ID, "nunique"))
    monthly.index = monthly.index.astype(str)
    monthly.to_csv(OUT / "stage_b_monthly_rows.csv")
    S["coverage"] = {
        "calendar_days_START_to_OBSERVED_THROUGH": int(len(daily)),
        "days_with_zero_rows": int((daily == 0).sum()),
        "min_rows_per_day": int(daily.min()), "median_rows_per_day": float(daily.median()),
        "rows_after_OBSERVED_THROUGH": int((d > C.OBSERVED_THROUGH).sum()),
        "respondents_last_purchase_on_or_before_OBSERVED_THROUGH": int((last <= C.OBSERVED_THROUGH).sum()),
        "respondents_first_purchase_after_last_cutoff": int((first > C.CUTOFFS[-1]).sum()),
        "respondents_first_purchase_in_2018Q1": int((first < "2018-04-01").sum()),
        "last_window_end": ds(C.CUTOFFS[-1] + pd.Timedelta(days=C.HORIZON_DAYS)),
        "latest_admissible_cutoff": ds(C.OBSERVED_THROUGH - pd.Timedelta(days=C.HORIZON_DAYS)),
    }
    # days between consecutive windows that belong to no window (cutoff c+90 < next cutoff)
    gaps = []
    for a, b in zip(C.CUTOFFS[:-1], C.CUTOFFS[1:]):
        end = a + pd.Timedelta(days=C.HORIZON_DAYS)
        gaps += [ds(x) for x in pd.date_range(end + pd.Timedelta(days=1), b)]
    S["coverage"]["days_in_no_target_window"] = gaps

    # 5. panels: primary (raw) and sensitivity versions
    days = {v: purchase_days(data_version(tx, v)) for v in ("raw", "dedup", "nogift", "dedup_nogift")}
    panel = build_panel(days["raw"], C.CUTOFFS)
    pn = build_panel(days["nogift"], C.CUTOFFS).rename(
        columns={"y": "y_nogift", "target_purchase_days": "target_purchase_days_nogift"})
    S["dedup_panel_identical_to_raw"] = bool(build_panel(days["dedup"], C.CUTOFFS).equals(panel))
    S["dedup_nogift_panel_identical_to_nogift"] = bool(
        build_panel(days["dedup_nogift"], C.CUTOFFS)[[ID, "cutoff", "y"]]
        .rename(columns={"y": "y_nogift"}).equals(pn[[ID, "cutoff", "y_nogift"]]))
    panel = panel.merge(pn[[ID, "cutoff", "y_nogift"]], on=[ID, "cutoff"], how="left",
                        validate="one_to_one")
    panel["eligible_nogift"] = panel["y_nogift"].notna()
    assert len(pn) == panel["eligible_nogift"].sum(), "nogift population must be a subset of raw"
    panel["y_nogift"] = panel["y_nogift"].astype("Int8")
    panel.assign(**{c: panel[c].dt.strftime("%Y-%m-%d") for c in ("cutoff", "window_start", "window_end")}
                 ).to_csv(OUT / "panel_labels.csv.gz", index=False, compression="gzip")

    rows = []
    for c, g in panel.groupby("cutoff"):
        n, pos = len(g), int(g["y"].sum())
        gn = g[g["eligible_nogift"]]
        lo, hi = wilson(pos, n)
        a = STAGE_A[ds(c)]
        rows.append({"cutoff": ds(c), "window_start": ds(g["window_start"].iat[0]),
                     "window_end": ds(g["window_end"].iat[0]), "n": n, "y1": pos, "y0": n - pos,
                     "prevalence_y1": pos / n, "prev_ci_low": lo, "prev_ci_high": hi,
                     "pct_y0": 100 * (n - pos) / n,
                     "nogift_n": len(gn), "nogift_y1": int(gn["y_nogift"].sum()),
                     "nogift_pct_y0": 100 * (1 - gn["y_nogift"].mean()),
                     "nogift_label_flips_1to0": int(((gn["y"] == 1) & (gn["y_nogift"] == 0)).sum()),
                     "stageA_n": a[0], "stageA_y1": a[1], "stageA_y0": a[2],
                     "matches_stageA": (n, pos, n - pos) == a})
    # descriptive only (uses data after the window; never a feature or a label): does a
    # y = 0 respondent buy again later in the export? Bears on the "not churn" wording.
    last_day = days["raw"].groupby(ID)[DATE].max()
    for r in rows:
        g = panel[(panel["cutoff"] == pd.Timestamp(r["cutoff"])) & (panel["y"] == 0)]
        later = g[ID].map(last_day) > g["window_end"]
        r["y0_buying_again_after_window"] = int(later.sum())
        r["y0_buying_again_by_2022-10-31"] = int(
            g[ID].map(days["raw"][days["raw"][DATE] <= C.OBSERVED_THROUGH].groupby(ID)[DATE].max())
            .gt(g["window_end"]).sum())
    cc = pd.DataFrame(rows)
    cc.to_csv(OUT / "stage_b_cutoff_counts.csv", index=False)
    tot = (len(panel), int(panel["y"].sum()), int((panel["y"] == 0).sum()))
    S["panel"] = {"observations": tot[0], "y1": tot[1], "y0": tot[2],
                  "prevalence_y1": tot[1] / tot[0],
                  "matches_stageA_total": tot == STAGE_A_TOTAL,
                  "all_cutoffs_match_stageA": bool(cc["matches_stageA"].all()),
                  "nogift_observations": int(panel["eligible_nogift"].sum()),
                  "nogift_y1": int(panel["y_nogift"].sum()),
                  "raw_only_eligible_obs (history = gift cards only)": int((~panel["eligible_nogift"]).sum()),
                  "nogift_label_flips_1to0": int(cc["nogift_label_flips_1to0"].sum())}
    S["stageA_qc_reconciliation"] = {
        k: {"stage_a": v, "stage_b": (qc["missing"][k[8:]] if k.startswith("missing_") else qc[k]),
            "match": v == (qc["missing"][k[8:]] if k.startswith("missing_") else qc[k])}
        for k, v in STAGE_A_QC.items()}

    # 6. leakage checks on the real data
    leak = {}
    rd = days["raw"]
    for c in C.CUTOFFS:
        base = labels_at(rd, c)
        w_end = c + pd.Timedelta(days=C.HORIZON_DAYS)
        # (a) delete every day after the window: labels and population unchanged
        a = labels_at(rd[rd[DATE] <= w_end], c)
        # (b) add a purchase for every respondent after the window: unchanged
        extra = pd.DataFrame({ID: rd[ID].unique(), DATE: w_end + pd.Timedelta(days=1)})
        b = labels_at(pd.concat([rd, extra]).sort_values([ID, DATE]), c)
        # (c) population depends only on history: delete everything after the cutoff
        e = eligible_respondents(rd[rd[DATE] <= c], c)
        leak[ds(c)] = bool(base.equals(a) and base.equals(b) and e.equals(pd.Index(base[ID])))
    S["leakage_checks_real_data"] = {"per_cutoff_pass": leak, "all_pass": all(leak.values())}

    # 7. repeated respondents
    per = panel.groupby(ID).agg(k=("y", "size"), ysum=("y", "sum"))
    dist = per["k"].value_counts().sort_index()
    p = panel.sort_values([ID, "cutoff"])
    p["cut_idx"] = p["cutoff"].map({c: i for i, c in enumerate(C.CUTOFFS)})
    prev = p.groupby(ID)[["y", "cut_idx"]].shift()
    adj = prev["cut_idx"].eq(p["cut_idx"] - 1)
    trans = pd.crosstab(prev.loc[adj, "y"].astype(int), p.loc[adj, "y"]).rename_axis(
        index="y_previous_cutoff", columns="y_current")
    # one-way ANOVA ICC(1) of y by respondent (unbalanced, k0 correction)
    N, R = len(panel), len(per)
    gm = panel["y"].mean()
    m = per["ysum"] / per["k"]
    ssb = (per["k"] * (m - gm) ** 2).sum()
    ssw = ((panel["y"] - panel[ID].map(m)) ** 2).sum()
    msb, msw = ssb / (R - 1), ssw / (N - R)
    k0 = (N - (per["k"] ** 2).sum() / N) / (R - 1)
    S["repeated_respondents"] = {
        "unique_respondents": int(R), "observations": int(N),
        "cutoffs_per_respondent": {int(k): int(v) for k, v in dist.items()},
        "respondents_at_all_cutoffs": int((per["k"] == len(C.CUTOFFS)).sum()),
        "respondents_always_y1": int((per["ysum"] == per["k"]).sum()),
        "respondents_ever_y0": int((per["ysum"] < per["k"]).sum()),
        "respondents_always_y0": int((per["ysum"] == 0).sum()),
        "share_of_y0_obs_from_respondents_with_ge3_y0": float(
            panel.loc[panel["y"] == 0, ID].map(per["k"] - per["ysum"]).ge(3).mean()),
        "transition_counts": {f"{i}->{j}": int(trans.loc[i, j]) for i in (0, 1) for j in (0, 1)},
        "P(y=0 | y_prev=0)": float(trans.loc[0, 0] / trans.loc[0].sum()),
        "P(y=0 | y_prev=1)": float(trans.loc[1, 0] / trans.loc[1].sum()),
        "icc1_anova": float((msb - msw) / (msb + (k0 - 1) * msw)),
    }
    pd.DataFrame({"cutoffs_observed": dist.index, "respondents": dist.to_numpy()}).to_csv(
        OUT / "stage_b_repeated_respondents.csv", index=False)

    # 8. proposed validation roles (descriptive only; design awaits approval)
    role = {}
    for r, cs in PROPOSED_ROLES.items():
        g = panel[panel["cutoff"].isin(cs)]
        role[r] = {"cutoffs": [ds(c) for c in cs], "obs": len(g), "y0": int((g["y"] == 0).sum()),
                   "respondents": int(g[ID].nunique()),
                   "last_label_day": ds(max(cs) + pd.Timedelta(days=C.HORIZON_DAYS))}
    te = set(panel.loc[panel["cutoff"].isin(PROPOSED_ROLES["test"]), ID])
    tr = set(panel.loc[panel["cutoff"].isin(PROPOSED_ROLES["train"]), ID])
    role["test_respondents_also_in_train"] = len(te & tr)
    role["test_respondents_not_in_train"] = len(te - tr)
    role["embargo_ok"] = {
        "train_vs_validation": max(PROPOSED_ROLES["train"]) + pd.Timedelta(days=C.HORIZON_DAYS)
        <= min(PROPOSED_ROLES["validation"]),
        "refit_vs_test": C.CUTOFFS[12] + pd.Timedelta(days=C.HORIZON_DAYS) <= min(PROPOSED_ROLES["test"])}
    S["proposed_roles_pending_approval"] = role

    # 9. independent recomputation (separate process, standard library only)
    t1 = time.time()
    subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("independent_check.py")),
                    str(C.PURCHASES_CSV), str(OUT)], check=True, capture_output=True)
    ind = pd.read_csv(OUT / "independent_labels.csv.gz", dtype={ID: str})
    ind["cutoff"] = pd.to_datetime(ind["cutoff"])
    mg = panel[[ID, "cutoff", "y", "y_nogift"]].merge(ind, on=[ID, "cutoff"], how="outer",
                                                     indicator=True)
    both = mg[mg["_merge"] == "both"]
    a_ng, b_ng = both["y_nogift"].astype("Float64"), both["y_nogift_independent"].astype("Float64")
    ij = json.loads((OUT / "independent_check.json").read_text())
    S["independent_check"] = {
        "seconds": round(time.time() - t1, 1),
        "pairs_only_in_pipeline": int((mg["_merge"] == "left_only").sum()),
        "pairs_only_in_independent": int((mg["_merge"] == "right_only").sum()),
        "label_disagreements_raw": int((both["y"] != both["y_independent"]).sum()),
        # missing on one side only (eligibility differs) or both present and unequal
        "label_disagreements_nogift": int((a_ng.isna() != b_ng.isna()).sum()
                                          + (a_ng != b_ng).fillna(False).sum()),
        "rows_match": ij["rows"] == qc["rows"],
        "respondents_match": ij["respondents"] == qc["respondents"],
        "exact_duplicate_copies_match": ij["exact_duplicate_extra_copies"] == qc["exact_duplicate_extra_copies"],
        "gift_rows_match": ij["gift_rows"] == qc["gift_rows"],
        "cutoffs_match": ij["cutoffs"] == [ds(c) for c in C.CUTOFFS],
    }
    S["independent_check"]["all_match"] = bool(
        all(v for k, v in S["independent_check"].items() if k.endswith("match"))
        and not any(S["independent_check"][k] for k in (
            "pairs_only_in_pipeline", "pairs_only_in_independent",
            "label_disagreements_raw", "label_disagreements_nogift")))

    # 10. environment
    import sklearn, lightgbm                                     # noqa: E401
    S["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__}
    S["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "stage_b_validation_summary.json").write_text(json.dumps(S, indent=1, default=str))
    print(json.dumps({k: S[k] for k in ("panel", "leakage_checks_real_data", "independent_check",
                                        "runtime_seconds")}, indent=1, default=str))


if __name__ == "__main__":
    main()
