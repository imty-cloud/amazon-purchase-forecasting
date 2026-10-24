"""Independent recomputation of Stage B counts and labels (standard library only).

Shares no code with src/apf: the file is streamed with the csv module, dates are parsed with
datetime, exact duplicates are found with SHA-1 of the raw fields, cutoffs are generated
from calendar arithmetic, and labels come from sorted per-respondent date lists via bisect.
Usage: python -I scripts/independent_check.py <amazon-purchases.csv> <out_dir>
"""
import bisect
import csv
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
START, LAST_OBSERVED, H = dt.date(2018, 1, 1), dt.date(2022, 10, 31), 90
GIFT_CAT = re.compile(r"GIFT_(?:CARD|XXRD)")
GIFT_TITLE = re.compile(r"\bgift\s*cards?\b|\begift\b|\bgift\s*certificates?\b", re.IGNORECASE)


def quarter_ends(first, last):
    out, y, q = [], first.year, (first.month - 1) // 3
    while True:
        m = 3 * q + 3
        nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
        d = nxt - dt.timedelta(days=1)
        if d > last:
            return out
        if d >= first:
            out.append(d)
        q = (q + 1) % 4
        y += q == 0


cutoffs = [c for c in quarter_ends(dt.date(2018, 12, 31), dt.date(2022, 6, 30))
           if c + dt.timedelta(days=H) <= LAST_OBSERVED]

seen, dup_copies = set(), 0
days_all, days_nogift = defaultdict(set), defaultdict(set)
rows = gift_rows = 0
with open(src, newline="", encoding="utf-8") as f:
    r = csv.reader(f)
    header = next(r)
    i_date, i_title, i_cat, i_id = (header.index(c) for c in
                                    ("Order Date", "Title", "Category", "Survey ResponseID"))
    for row in r:
        rows += 1
        k = hashlib.sha1("\x1f".join(row).encode()).digest()
        if k in seen:
            dup_copies += 1
        else:
            seen.add(k)
        d = dt.date.fromisoformat(row[i_date])
        gift = bool(GIFT_CAT.search(row[i_cat]) or GIFT_TITLE.search(row[i_title]))
        gift_rows += gift
        if d >= START:
            days_all[row[i_id]].add(d)
            if not gift:
                days_nogift[row[i_id]].add(d)
del seen


def labels(days):
    res = {}
    for rid, s in days.items():
        ds = sorted(s)
        for c in cutoffs:
            if ds[0] > c:            # no purchase day in [START, c]: not eligible
                continue
            j = bisect.bisect_right(ds, c)             # first day > c
            res[(rid, c)] = int(j < len(ds) and ds[j] <= c + dt.timedelta(days=H))
    return res


lab, lab_ng = labels(days_all), labels(days_nogift)
out.mkdir(parents=True, exist_ok=True)
with gzip.open(out / "independent_labels.csv.gz", "wt", newline="") as g:
    w = csv.writer(g)
    w.writerow(["Survey ResponseID", "cutoff", "y_independent", "y_nogift_independent"])
    for (rid, c) in sorted(set(lab) | set(lab_ng)):
        w.writerow([rid, c.isoformat(), lab.get((rid, c), ""), lab_ng.get((rid, c), "")])
summary = {
    "rows": rows, "respondents": len(days_all), "exact_duplicate_extra_copies": dup_copies,
    "gift_rows": gift_rows, "cutoffs": [c.isoformat() for c in cutoffs],
    "per_cutoff": {c.isoformat(): {
        "n": sum(1 for k in lab if k[1] == c), "pos": sum(v for k, v in lab.items() if k[1] == c),
        "n_nogift": sum(1 for k in lab_ng if k[1] == c),
        "pos_nogift": sum(v for k, v in lab_ng.items() if k[1] == c)} for c in cutoffs},
}
(out / "independent_check.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "per_cutoff"}))
