"""Checks reports/stage_d_results.md against outputs/, independently of the report builder.

1. Every number in the provenance record is re-read from its source, re-formatted, and must appear at its
   recorded position in the report.
2. Every other number in the report must be a structural constant (dates, section numbers, the protocol's
   fixed levels); any other number fails.
3. In tables, each number's source must agree with what a reader sees: the column header, the set, model,
   cutoff, comparison, metric, bin and file labels of its row, and the section heading.
4. Reading cells of comparison tables must follow the interval flags in stage_d_comparisons.csv.
5. Completeness: every pooled metric, interval, comparison and per-cutoff stability row is reported.
6. Wording: no claim of superiority, significance, equivalence or stability; required statements present.
Usage: .venv/bin/python scripts/check_stage_d_report.py [--report PATH] [--provenance PATH]  (exit 1 on failure)
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TM, CI, CMP, ST, TVH, REL = (f"outputs/{f}" for f in (
    "stage_d_test_metrics.csv", "stage_d_test_metric_cis.csv", "stage_d_comparisons.csv",
    "stage_d_temporal_stability.csv", "stage_d_temporal_vs_holdout.csv", "stage_d_reliability.csv"))
META = "outputs/stage_d_phase2_meta.json"
SETS = ("temporal_test", "holdout_test", "holdout_train_period")
MODELS = ("B0_prevalence", "B1_persistence", "B2_recency", "M1_lr_rfm", "M2_lr_full", "S1_lr_no_prev",
          "M3_lgbm_full", "S2_lgbm_no_prev")
METRIC_LABEL = {"AP": "ap_y0", "ROC-AUC": "roc_auc", "Brier": "brier"}


def ci3(metric):
    return [(CI, "ci_low", {"metric": metric}), (CI, "ci_high", {"metric": metric})]


def fixed(col, s):
    return [(TM, col, {"set": s, "scope": "pooled"})]


# column header -> the sources a number under that header may come from: (file, column or json path, fixed keys)
HEADER = {
    "AP (y = 0) [95 % CI]": [(TM, "ap_y0", {}), *ci3("ap_y0"), (ST, "ap_y0", {}), (ST, "ap_ci_low", {}),
                             (ST, "ap_ci_high", {})],
    "No-skill AP": [(TM, "ap_no_skill", {})],
    "ROC-AUC [95 % CI]": [(TM, "roc_auc", {}), *ci3("roc_auc")],
    "Brier [95 % CI]": [(TM, "brier", {}), *ci3("brier")],
    "Brier skill": [(TM, "brier_skill", {})],
    "Calibration intercept": [(TM, "cal_intercept", {}), (ST, "cal_intercept", {})],
    "Calibration slope": [(TM, "cal_slope", {})],
    "Precision, top 10 %": [(TM, "precision_top10", {})], "Recall, top 10 %": [(TM, "recall_top10", {})],
    "Precision, top 20 %": [(TM, "precision_top20", {})], "Recall, top 20 %": [(TM, "recall_top20", {})],
    "Rows": [(ST, "n", {}), (REL, "n", {}), (META, ("rows",), {})],
    "y = 0": [(ST, "n_event_y0", {}), (TM, "n_event_y0", {})],
    "Prevalence of y = 0": [(ST, "prevalence_y0", {}), (TM, "prevalence_y0", {})],
    "Fitted on (rows)": [(META, ("fit_rows",), {})],
    "AP − prevalence": [(ST, "ap_minus_prevalence", {})], "AP / prevalence": [(ST, "ap_lift", {})],
    "ROC-AUC": [(TM, "roc_auc", {})], "Brier": [(TM, "brier", {})],
    "Estimate": [(CMP, "estimate", {})],
    "95 % CI": [(CMP, "ci_low", {}), (CMP, "ci_high", {}), (TVH, "ci_low", {}), (TVH, "ci_high", {})],
    "Bonferroni 98.33 % CI": [(CMP, "bonf_ci_low", {}), (CMP, "bonf_ci_high", {})],
    "Intercept, temporal test": fixed("cal_intercept", "temporal_test"),
    "Intercept, hold-out test": fixed("cal_intercept", "holdout_test"),
    "Intercept, training-period diagnostic": fixed("cal_intercept", "holdout_train_period"),
    "Slope, temporal test": fixed("cal_slope", "temporal_test"),
    "Slope, hold-out test": fixed("cal_slope", "holdout_test"),
    "Slope, training-period diagnostic": fixed("cal_slope", "holdout_train_period"),
    "Bin": [(REL, "bin", {})], "Mean p0": [(REL, "mean_p0", {})],
    "Observed share of y = 0": [(REL, "observed_y0", {})],
    "AP, temporal test": [(TVH, "ap_temporal", {})], "AP, hold-out test": [(TVH, "ap_holdout", {})],
    "Difference (temporal − hold-out)": [(TVH, "difference", {})],
    "SHA-256": [("sha256", None, {})],
}
FORBIDDEN = r"superior|outperform|significan|\bproves?\b|\bconfirms?\b|\bis stable\b|\bare stable\b|" \
            r"\bremains? stable\b|stable over time|equivalent to|\bthe same as\b|\bbest model\b"
WATCH = r"stab\w*|better|equivalen\w*|\bsame\b|robust\w*|reliab\w*|consistent\w*"
REQUIRED = ["no purchase day was recorded in this dataset during the next 90 days", "It is not confirmed churn",
            "Temporal test", "Hold-out test", "Hold-out, training period (diagnostic)",
            "It is not an estimate of future performance", "Bonferroni 98.33 % CI",
            "Two adjacent quarters (Q1 and Q2 2022) cannot establish long-term temporal stability",
            "Temporal stability is not established", "does not rank LightGBM above logistic regression",
            "No recalibration", "not evidence of equivalence"]
STRUCTURAL = [r"\b\d{4}-\d{2}-\d{2}\b", r"(?m)^#+ [\d.]+", r"§\s?\d+(?:\.\d+)*", r"\b(?:95|98\.33|10|20) %",
              r"c\+1\b", r"c\+90\b", r"\b90 days\b", r"\by = [01]\b", r"P\(y = 0\)", r"Q1 and Q2 2022",
              r"(?:excludes?|excluding|includes?|exactly|close to|from|is|are|[<>]) 0\b", r"\bPhase [12]\b", r"\bSHA-256\b", r"(?:above|below|[<>]) 1\b",
              r"(?:B|C|M|P|S)-?[0-9]\b"]
NUMBER = r"(?<![\w.])[+\-−]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)"

FAIL = []


def fail(msg):
    FAIL.append(msg)


_csv = {}


def table(f):
    if f not in _csv:
        _csv[f] = pd.read_csv(ROOT / f)
    return _csv[f]


def lookup(rec):
    """The value behind a record, read from its source here (not from the builder)."""
    src = rec["source"]
    if src == "csv":
        d = table(rec["file"])
        m = pd.Series(True, index=d.index)
        for k, v in rec["where"].items():
            m &= d[k].astype(str) == str(v)
        if m.sum() != 1:
            raise LookupError(f"{rec['file']} {rec['where']}: {int(m.sum())} rows")
        x = d[m].iloc[0][rec["column"]]
        return x.item() if hasattr(x, "item") else x
    if src == "json":
        x = json.loads((ROOT / rec["file"]).read_text())
        for p in rec["path"]:
            x = x[p]
        return x
    if src == "sha256":
        return hashlib.sha256((ROOT / rec["file"]).read_bytes()).hexdigest()
    if src == "git":
        return subprocess.run(["git", "rev-parse", rec["ref"]], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    raise ValueError(src)


def expected_text(rec, x):
    if rec["source"] in ("sha256", "git"):
        return x
    return "n/a" if isinstance(x, float) and math.isnan(x) else format(x, rec["fmt"])


def set_of(text):
    t = text.lower()
    if "training period" in t or "training-period" in t:
        return "holdout_train_period"
    if "hold-out" in t:
        return "holdout_test"
    if "temporal" in t:
        return "temporal_test"
    return None


def parse_tables(text):
    """Yield (heading, header cells, rows); a row is a list of (cell text, start, end) in report offsets."""
    heading, pos, lines = "", 0, []
    for line in text.split("\n"):
        lines.append((line, pos))
        pos += len(line) + 1
    i = 0
    while i < len(lines):
        line, start = lines[i]
        if line.startswith("#"):
            heading = line
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1][0].startswith("|---"):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i][0].startswith("|"):
                ln, st = lines[i]
                cells, j = [], st + 1                  # after the leading "|"
                for c in ln[1:].split("|")[:-1]:
                    cells.append((c.strip(), j, j + len(c)))
                    j += len(c) + 1
                if len(cells) != len(header):
                    fail(f"table row with {len(cells)} cells under {len(header)} headers: {ln[:80]}")
                rows.append(cells)
                i += 1
            yield heading, header, rows
            continue
        i += 1


def ticks(cell):
    return re.findall(r"`([^`]+)`", cell)


def check_cell(rec, head, header, row, col, recs_in_row):
    h = header[col]
    cols = {name.split(" (")[0] if name.startswith("Comparison") else name: k for k, name in enumerate(header)}
    allowed = HEADER.get(h)
    if allowed is None:
        return fail(f"#{rec['id']}: number under unknown header '{h}'")
    src_ok, fixed_keys = False, {}
    for f, c, extra in allowed:
        if f == "sha256" and rec["source"] == "sha256":
            src_ok = True
        elif rec["source"] == "csv" and rec["file"] == f and rec["column"] == c and \
                all(str(rec["where"].get(k)) == str(v) for k, v in extra.items()):
            src_ok, fixed_keys = True, extra
        elif rec["source"] == "json" and rec["file"] == f and tuple(rec["path"][:len(c)]) == c:
            src_ok = True
    if not src_ok:
        return fail(f"#{rec['id']} ({rec.get('file')}:{rec.get('column', rec.get('path'))}) under header '{h}'")
    cell = lambda name: row[cols[name]][0] if name in cols else None
    where = dict(rec.get("where", {}))
    if rec["source"] == "json":
        for p in rec["path"]:
            if p in SETS:
                where["set"] = p
            if p in MODELS:
                where["model"] = p
    if rec["source"] == "sha256":
        if ticks(cell("File") or "") != [rec["file"]]:
            fail(f"#{rec['id']}: checksum of {rec['file']} in the row of {cell('File')}")
        return
    if "set" in where and "set" not in fixed_keys:
        seen = set_of(cell("Set")) if "Set" in cols else set_of(head)
        if seen != where["set"]:
            fail(f"#{rec['id']}: {where['set']} value shown under set label {seen} ({head.strip()})")
    if "model" in where:
        if "Model" in cols:
            if ticks(cell("Model")) != [where["model"]]:
                fail(f"#{rec['id']}: {where['model']} value in the row of {cell('Model')}")
        elif [t for t in ticks(head) if t in MODELS]:          # model named in the table's heading
            if [t for t in ticks(head) if t in MODELS] != [where["model"]]:
                fail(f"#{rec['id']}: {where['model']} value under heading '{head.strip()}'")
        else:                                           # no model label: the value must not depend on the model
            others = {k: v for k, v in where.items() if k != "model"}
            if rec["source"] == "csv":
                d = table(rec["file"])
                m = pd.Series(True, index=d.index)
                for k, v in others.items():
                    m &= d[k].astype(str) == str(v)
                vals = d.loc[m, rec["column"]]
            else:
                x = json.loads((ROOT / rec["file"]).read_text())
                for p in rec["path"][:-1]:
                    x = x[p]
                vals = pd.Series(list(x.values()))
            if vals.nunique() != 1:
                fail(f"#{rec['id']}: model-specific value {rec['column'] if 'column' in rec else rec['path']} "
                     "shown without a model label")
    for key in ("scope", "cutoff"):
        if key in where and key not in fixed_keys:
            if "Cutoff" in cols:
                if cell("Cutoff") != str(where[key]):
                    fail(f"#{rec['id']}: cutoff {where[key]} value in the row of {cell('Cutoff')}")
            elif where[key] != "pooled" or "pooled" not in head.lower():
                fail(f"#{rec['id']}: {key}={where[key]} shown under heading '{head.strip()}'")
    if "metric" in where and "metric" not in fixed_keys:
        if METRIC_LABEL.get(cell("Metric")) != where["metric"]:
            fail(f"#{rec['id']}: metric {where['metric']} in the row labelled {cell('Metric')}")
    if "comparison" in where:
        if cell("ID") != where["comparison"]:
            fail(f"#{rec['id']}: comparison {where['comparison']} in the row labelled {cell('ID')}")
        r = table(CMP)[(table(CMP)["comparison"] == where["comparison"])].iloc[0]
        if ticks(cell("Comparison") or "") != [r["model_a"], r["model_b"]]:
            fail(f"#{rec['id']}: {where['comparison']} is {r['model_a']} − {r['model_b']}, row shows "
                 f"{cell('Comparison')}")
    if "bin" in where and rec["column"] != "bin":
        bins = [r for r in recs_in_row[cols["Bin"]] if r["column"] == "bin"]
        if len(bins) != 1 or bins[0]["text"] != str(where["bin"]):
            fail(f"#{rec['id']}: bin {where['bin']} value in a row labelled {cell('Bin')}")


def reading_of(r):
    if r["role"] == "primary":
        if str(r["bonf_ci_excludes_0"]) == "True":
            return "supported (95 % and Bonferroni intervals exclude 0)"
        if r["ci_excludes_0"]:
            return "supported at 95 %; Bonferroni interval includes 0"
        return "no clear difference (both intervals include 0)"
    if r["estimate"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0:
        return "identical scores (estimate and interval exactly 0)"
    return "supported (interval excludes 0)" if r["ci_excludes_0"] else "no clear difference (interval includes 0)"


def main(report, provenance):
    text = report.read_text()
    recs = json.loads(provenance.read_text())
    for i, r in enumerate(recs):
        r["id"] = i
    # 1. values and positions
    for r in recs:
        try:
            want = expected_text(r, lookup(r))
        except (LookupError, KeyError, IndexError) as e:
            fail(f"#{r['id']}: source not found: {e}")
            continue
        if want != r["text"]:
            fail(f"#{r['id']}: recorded '{r['text']}', source gives '{want}' ({r})")
        if text[r["offset"]:r["offset"] + len(want)] != want:
            fail(f"#{r['id']}: '{want}' not at its recorded position {r['offset']}")
    # 2. no other number
    masked = list(text)
    for r in recs:
        masked[r["offset"]:r["offset"] + len(r["text"])] = " " * len(r["text"])
    rest = "".join(masked)
    for p in STRUCTURAL:
        rest = re.sub(p, lambda m: " " * len(m.group()), rest)
    stray = [(m.group(), text[max(0, m.start() - 40):m.end() + 20].replace("\n", " "))
             for m in re.finditer(NUMBER, rest)]
    for tok, ctx in stray:
        fail(f"number '{tok}' has no source: …{ctx}…")
    # 3. tables
    by_offset = sorted(recs, key=lambda r: r["offset"])
    in_tables, n_tables, n_rows = set(), 0, 0
    for head, header, rows in parse_tables(text):
        n_tables += 1
        cols = {name: k for k, name in enumerate(header)}
        for row in rows:
            n_rows += 1
            recs_in_row = [[r for r in by_offset if s <= r["offset"] < e] for _, s, e in row]
            for col, rs in enumerate(recs_in_row):
                for r in rs:
                    in_tables.add(r["id"])
                    check_cell(r, head, header, row, col, recs_in_row)
            # 4. readings
            if "Reading" in cols:
                c = table(CMP)
                k = (c["set"] == set_of(row[cols["Set"]][0])) & (c["comparison"] == row[cols["ID"]][0]) & \
                    (c["metric"] == METRIC_LABEL[row[cols["Metric"]][0]])
                if k.sum() != 1 or row[cols["Reading"]][0] != reading_of(c[k].iloc[0]):
                    fail(f"reading '{row[cols['Reading']][0]}' for {[x[0] for x in row[:4]]}")
    # 5. completeness
    used = {(r.get("file"), json.dumps(r.get("where"), sort_keys=True), r.get("column")) for r in recs}

    def need(f, where, col):
        if (f, json.dumps(where, sort_keys=True), col) not in used:
            fail(f"not reported: {f} {where} {col}")
    for s in SETS:
        for m in MODELS:
            for col in ("ap_y0", "ap_no_skill", "roc_auc", "brier", "brier_skill", "cal_intercept", "cal_slope",
                        "precision_top10", "recall_top10", "precision_top20", "recall_top20"):
                need(TM, {"set": s, "model": m, "scope": "pooled"}, col)
    for r in table(CI).itertuples():
        for col in ("ci_low", "ci_high"):
            need(CI, {"metric": r.metric, "set": r.set, "model": r.model}, col)
    for r in table(CMP).itertuples():
        for col in ("estimate", "ci_low", "ci_high") + (("bonf_ci_low", "bonf_ci_high") if r.role == "primary" else ()):
            need(CMP, {"set": r.set, "comparison": r.comparison, "metric": r.metric}, col)
    for r in table(ST).itertuples():
        for col in ("ap_y0", "ap_ci_low", "ap_ci_high", "ap_minus_prevalence", "ap_lift", "cal_intercept"):
            need(ST, {"set": r.set, "cutoff": r.cutoff, "model": r.model}, col)
    for r in table(TVH).itertuples():
        for col in ("ap_temporal", "ap_holdout", "difference", "ci_low", "ci_high"):
            need(TVH, {"model": r.model}, col)
    # 6. wording
    flat = " ".join(re.sub(r"[*`]", "", text).split())
    for m in re.finditer(FORBIDDEN, flat, re.I):
        fail(f"forbidden wording '{m.group()}': …{flat[max(0, m.start() - 60):m.end() + 40]}…")
    for s in REQUIRED:
        if s not in flat:
            fail(f"required statement missing: '{s}'")
    for link in re.findall(r"\]\((\.\./[^)]+)\)", text):
        if not (ROOT / "reports" / link).resolve().exists():     # links are relative to reports/
            fail(f"broken link {link}")
    watch = [(m.group(), flat[max(0, m.start() - 70):m.end() + 50]) for m in re.finditer(WATCH, flat, re.I)]
    prose = [r for r in recs if r["id"] not in in_tables]

    print(f"records: {len(recs)} (in {n_tables} tables / {n_rows} rows: {len(in_tables)}; in prose: {len(prose)})")
    print(f"numbers without a source: {len(stray)}")
    print(f"wording to review ({len(watch)}):")
    for w, ctx in watch:
        print(f"  [{w}] …{ctx}…")
    if FAIL:
        print(f"\nFAILED ({len(FAIL)}):")
        for f in FAIL[:60]:
            print("  " + f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "reports" / "stage_d_results.md"))
    ap.add_argument("--provenance", default=str(ROOT / "outputs" / "stage_d_report_provenance.json"))
    a = ap.parse_args()
    main(Path(a.report), Path(a.provenance))
