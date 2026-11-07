"""F: gift-card sensitivity of the TARGET and ELIGIBILITY. Counts only - no model is fitted or scored.
POST-HOC / exploratory. Gift-card rows count as purchase days in the locked pipeline (history and label);
here they are removed to count how many respondent-cutoff rows change eligibility or label.
Exact duplicate rows share (respondent, day) with their original, so they cannot change days: verified below."""
from post_common import *
from apf.config import CUTOFFS, DATE, ID
from apf.data import load_transactions
from apf.splits import (TRAIN_CUTOFFS, TUNE_CUTOFFS, TEST_CUTOFFS, REFIT_CUTOFFS, holdout_respondents)
from apf.target import build_panel, eligible_respondents, purchase_days

check_data()
tx, qc = load_transactions(PURCHASES_CSV)
days_all = purchase_days(tx)
days_ng = purchase_days(tx[~tx["is_gift"]])
days_dd = purchase_days(tx[~tx["is_dup_copy"]])
assert days_dd.equals(days_all), "duplicates changed purchase days"      # duplicates cannot change labels
base = build_panel(days_all, CUTOFFS)[[ID, "cutoff", "y"]]
ng = build_panel(days_ng, CUTOFFS)[[ID, "cutoff", "y"]].rename(columns={"y": "y_nogift"})
m = base.merge(ng, on=[ID, "cutoff"], how="left", validate="one_to_one")
m["ineligible_nogift"] = m["y_nogift"].isna()
m["flip_1to0"] = (~m["ineligible_nogift"]) & (m["y"] == 1) & (m["y_nogift"] == 0)
m["flip_0to1"] = (~m["ineligible_nogift"]) & (m["y"] == 0) & (m["y_nogift"] == 1)
ho = holdout_respondents(eligible_respondents(days_all, CUTOFFS[-1]))
m["holdout"] = m[ID].isin(ho)
def role(c):
    c = pd.Timestamp(c)
    return ("train" if c in TRAIN_CUTOFFS else "tune" if c in TUNE_CUTOFFS else
            "refit_only" if c in REFIT_CUTOFFS else "test" if c in TEST_CUTOFFS else "other")
m["role"] = m["cutoff"].map(role)
def summarize(g, label):
    n = len(g); keep = g[~g["ineligible_nogift"]]
    return dict(group=label, rows=n, rows_ineligible_without_gift=int(g["ineligible_nogift"].sum()),
                share_ineligible=g["ineligible_nogift"].mean(),
                flips_1to0=int(g["flip_1to0"].sum()), flips_0to1=int(g["flip_0to1"].sum()),
                share_label_changed_among_eligible=(keep["flip_1to0"] | keep["flip_0to1"]).mean() if len(keep) else np.nan,
                prevalence_y0_locked=float((g["y"] == 0).mean()),
                prevalence_y0_without_gift=float((keep["y_nogift"] == 0).mean()) if len(keep) else np.nan,
                prevalence_y0_locked_same_rows=float((keep["y"] == 0).mean()) if len(keep) else np.nan)
rows = [summarize(m, "all_rows")]
for r in ("train", "tune", "refit_only", "test"):
    rows.append(summarize(m[m["role"] == r], r))
for c in sorted(m["cutoff"].unique()):
    rows.append(summarize(m[m["cutoff"] == c], f"cutoff_{pd.Timestamp(c).date()}"))
for name, sel in (("test_temporal", (m["role"] == "test") & ~m["holdout"]), ("test_holdout", (m["role"] == "test") & m["holdout"])):
    rows.append(summarize(m[sel], name))
df = pd.DataFrame(rows)
write_csv(df, "post_f_gift_label_counts.csv")
print(df.round(4).to_string())
print("gift rows:", qc["gift_rows"], "gift respondents:", qc["gift_respondents"], "| dup extra copies:", qc["exact_duplicate_extra_copies"])
print("purchase days all vs no-gift:", len(days_all), len(days_ng))
