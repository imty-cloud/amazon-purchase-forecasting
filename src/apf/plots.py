"""Stage D figures, drawn from saved result tables only (no model fitting, no labels).

Colours: the dataviz reference palette's first five categorical slots in fixed order, one per model;
each model also has its own marker so identity never depends on colour alone.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
STYLE = {  # model: (label, colour, marker)
    "M2_lr_full": ("LR, full", "#2a78d6", "o"),
    "M3_lgbm_full": ("LightGBM, full", "#eb6834", "s"),
    "M1_lr_rfm": ("LR, RFM only", "#1baf7a", "^"),
    "B2_recency": ("Recency rule", "#eda100", "D"),
    "B1_persistence": ("Persistence", "#e87ba4", "v"),
}
SET_TITLES = {"temporal_test": "Temporal test (seen respondents)",
              "holdout_test": "Hold-out test (unseen respondents)"}


def _axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)


def plot_temporal_stability(stab, path):
    """Two rows (AP for y = 0 with 95 % CI and the no-skill line; calibration-in-the-large) x one column
    per test set; x = test cutoff."""
    sets = [s for s in SET_TITLES if s in set(stab["set"])]
    fig, axes = plt.subplots(2, len(sets), figsize=(5.2 * len(sets), 6.4), squeeze=False, facecolor=SURFACE,
                             sharey="row")                           # same scale across test sets
    for j, set_name in enumerate(sets):
        d = stab[stab["set"] == set_name]
        cuts = sorted(d["cutoff"].unique())
        x = {c: i for i, c in enumerate(cuts)}
        ax_ap, ax_cal = axes[0, j], axes[1, j]
        prev = d.groupby("cutoff")["prevalence_y0"].first().reindex(cuts)
        ax_ap.plot(range(len(cuts)), prev.to_numpy(), linestyle="--", color=MUTED, linewidth=1.5,
                   label="No skill (prevalence of y = 0)")
        for k, (m, (lab, col, mk)) in enumerate(STYLE.items()):
            g = d[d["model"] == m].set_index("cutoff").reindex(cuts)
            off = (k - 2) * 0.06                                        # small dodge so intervals do not overlap
            xs = [x[c] + off for c in cuts]
            ax_ap.errorbar(xs, g["ap_y0"], yerr=[g["ap_y0"] - g["ap_ci_low"], g["ap_ci_high"] - g["ap_y0"]],
                           color=col, marker=mk, markersize=7, linewidth=2, capsize=3, label=lab)
            ax_cal.plot(xs, g["cal_intercept"], color=col, marker=mk, markersize=7, linewidth=2, label=lab)
        ax_cal.axhline(0, color=MUTED, linewidth=1)
        for ax in (ax_ap, ax_cal):
            _axes(ax)
            ax.set_xticks(range(len(cuts)), cuts)
            ax.set_xlim(-0.4, len(cuts) - 0.6)
        ax_ap.set_title(SET_TITLES[set_name], color=INK, fontsize=10, loc="left")
        ax_ap.set_ylabel("Average precision, y = 0 (95 % CI)", color=INK, fontsize=9)
        ax_cal.set_ylabel("Calibration-in-the-large, event y = 0\n(> 0: y = 0 under-predicted)", color=INK, fontsize=9)
        ax_cal.set_xlabel("Test cutoff", color=INK, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(rel, path):
    """Reliability (deciles of predicted P(y = 0)) for the pooled test sets."""
    sets = [s for s in SET_TITLES if s in set(rel["set"])]
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.6), squeeze=False, facecolor=SURFACE,
                             sharex=True, sharey=True)
    for j, set_name in enumerate(sets):
        ax = axes[0, j]
        d = rel[rel["set"] == set_name]
        hi = max(d["mean_p0"].max(), d["observed_y0"].max()) * 1.05
        ax.plot([0, hi], [0, hi], color=MUTED, linestyle="--", linewidth=1.2, label="Perfect calibration")
        for m, (lab, col, mk) in STYLE.items():
            g = d[d["model"] == m].sort_values("mean_p0")
            ax.plot(g["mean_p0"], g["observed_y0"], color=col, marker=mk, markersize=6, linewidth=2, label=lab)
        _axes(ax)
        ax.set_title(SET_TITLES[set_name], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Mean predicted P(y = 0) in decile", color=INK, fontsize=9)
        ax.set_ylabel("Observed share of y = 0", color=INK, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
