"""Independent recomputation of selected features for a seeded sample of respondents.

Standard library only; shares no code with src/apf. Streams the raw CSV with the csv module and
computes features from per-respondent dictionaries of {day: spend} and {category: [days]}.
Usage: python -I scripts/independent_feature_check.py <amazon-purchases.csv> <out.csv> [n] [seed]
"""
import csv
import datetime as dt
import random
import statistics
import sys
from collections import defaultdict

src, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
START = dt.date(2018, 1, 1)
CUTOFFS = [dt.date(y, m, d) for y in range(2018, 2023) for (m, d) in ((3, 31), (6, 30), (9, 30), (12, 31))
           if dt.date(2018, 12, 31) <= dt.date(y, m, d) <= dt.date(2022, 6, 30)]

with open(src, newline="", encoding="utf-8") as f:
    ids = sorted({row["Survey ResponseID"] for row in csv.DictReader(f)})
sample = set(random.Random(SEED).sample(ids, N))

spend = defaultdict(lambda: defaultdict(float))      # id -> day -> spend
cats = defaultdict(lambda: defaultdict(list))        # id -> category -> [days]
with open(src, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rid = row["Survey ResponseID"]
        if rid not in sample:
            continue
        day = dt.date.fromisoformat(row["Order Date"])
        if day < START:
            continue
        spend[rid][day] += float(row["Purchase Price Per Unit"]) * float(row["Quantity"])
        if row["Category"] != "":
            cats[rid][row["Category"]].append(day)


def prev_quarter_end(c):
    q_start_month = 3 * ((c.month - 1) // 3) + 1
    return dt.date(c.year, q_start_month, 1) - dt.timedelta(days=1)


rows = []
for rid in sorted(spend):
    for c in CUTOFFS:
        days = sorted(d for d in spend[rid] if d <= c)
        if not days:
            continue
        age = [(c - d).days for d in days]                  # 0 = cutoff day
        gaps = [(b - a).days for a, b in zip(days, days[1:])]
        cp = prev_quarter_end(c)
        mon = sum(spend[rid][d] for d in days)
        rows.append({
            "Survey ResponseID": rid, "cutoff": c.isoformat(),
            "recency_days": age[-1], "frequency_days": len(days), "monetary_total": mon,
            "days_30": sum(a < 30 for a in age), "days_90": sum(a < 90 for a in age),
            "days_365": sum(a < 365 for a in age),
            "categories_total": sum(any(d <= c for d in v) for v in cats[rid].values()),
            "categories_365": sum(any(0 <= (c - d).days < 365 for d in v) for v in cats[rid].values()),
            "tenure_days": (c - days[0]).days,
            "median_gap_days": statistics.median(gaps) if gaps else "",
            "active_months_12": len({(d.year, d.month) for d in days
                                     if 0 <= (c.year * 12 + c.month) - (d.year * 12 + d.month) < 12}),
            "days_same_window_last_year": sum(275 <= a <= 364 for a in age),
            "prev_90d_active": (int(any(cp < d <= cp + dt.timedelta(days=90) for d in days))
                                if days[0] <= cp else ""),
        })

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} respondent-cutoff rows for {len(spend)} sampled respondents")
