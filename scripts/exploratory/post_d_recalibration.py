"""D: intercept-only and Platt recalibration. Calibrator fitted on Tune with the Train model, applied to the
same Train model's predictions on both test sets. POST-HOC / exploratory (reuses test sets)."""
from post_common import *
from apf.evaluation import calibration_intercept_slope, event, metrics, reliability_table
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
logit = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
sig = lambda z: 1 / (1 + np.exp(-z))
table = get_table()
train = table[table["in_train"]]; tune = table[table["in_tune"]]
specs = model_specs(CONF)
MODELS = ["M2_lr_full", "M3_lgbm_full"]
from apf.phase2 import _Model
fit = {m: _Model(*specs[m]).fit(train, train["y"].to_numpy()) for m in MODELS}
# LightGBM spec here is the locked one (114 trees) refit on Train: same hyper-parameters, no early stopping
rows, rels, cals = [], [], []
for m in MODELS:
    e_t = event(tune["y"]); lp_t = logit(fit[m].p0(tune))
    a_int = calibration_intercept_slope(e_t, fit[m].p0(tune))["cal_intercept"]          # slope fixed at 1
    pl = LogisticRegression(C=np.inf).fit(lp_t[:, None], e_t)
    a_pl, b_pl = float(pl.intercept_[0]), float(pl.coef_[0, 0]); assert b_pl > 0
    cals.append(dict(model=m, intercept_only_a=a_int, platt_a=a_pl, platt_b=b_pl, n_tune=len(tune)))
    for sname in TEST_SETS:
        S = test_set(table, sname); y = S["y"].to_numpy(); p = fit[m].p0(S); lp = logit(p)
        variants = {"none": p, "intercept_only": sig(lp + a_int), "platt": sig(a_pl + b_pl * lp)}
        base = metrics(y, p, train_prevalence0=fit[m].p_fit)
        for v, pv in variants.items():
            r = metrics(y, pv, train_prevalence0=fit[m].p_fit)
            assert abs(r["ap_y0"] - base["ap_y0"]) < 1e-9 and abs(r["roc_auc"] - base["roc_auc"]) < 1e-9, (m, sname, v)
            rows.append(dict(set=sname, model=m, variant=v, ap_y0=r["ap_y0"], roc_auc=r["roc_auc"], brier=r["brier"],
                             cal_intercept=r["cal_intercept"], cal_slope=r["cal_slope"],
                             mean_pred=float(pv.mean()), observed=r["prevalence_y0"]))
            rels.append(reliability_table(y, pv).reset_index().assign(set=sname, model=m, variant=v))
res, rel = pd.DataFrame(rows), pd.concat(rels, ignore_index=True)
write_csv(res, "post_d_recalibration.csv"); write_csv(rel, "post_d_reliability_deciles.csv")
write_csv(pd.DataFrame(cals), "post_d_calibrator_params.csv")
print(pd.DataFrame(cals).round(4).to_string()); print(res.round(4).to_string())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
for i, m in enumerate(MODELS):
    for j, s in enumerate(TEST_SETS):
        a = ax[i, j]; a.plot([0, 1], [0, 1], color="grey", lw=1)
        for v, c in (("none", "#c44"), ("intercept_only", "#48a"), ("platt", "#3a7")):
            q = rel[(rel.model == m) & (rel.set == s) & (rel.variant == v)]
            a.plot(q.mean_p0, q.observed_y0, "o-", ms=3, color=c, label=v)
        a.set_title(f"{m} · {s}", fontsize=9)
        if i == 1: a.set_xlabel("mean predicted P(y=0), decile")
        if j == 0: a.set_ylabel("observed rate y=0")
ax[0, 0].legend(fontsize=8); fig.suptitle("POST-HOC: reliability by decile (Train model, calibrator fit on Tune)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "post_d_reliability.png", dpi=130)
