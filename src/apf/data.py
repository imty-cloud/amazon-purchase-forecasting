"""Read-only, chunked loading and quality checks of amazon-purchases.csv.

Adapted from rfm_v3/rfm_study_v3.ipynb cell 6 (`md5sum`, `verify_files`, `load_purchases`):
same flags (spend, exact-duplicate copy, gift card, post-collection), but the file is read in
chunks as text, quality counts are taken on the raw strings, and only the columns needed
downstream are kept (Title, ASIN and State are dropped after flagging).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ASIN, CATEGORY, COLLECTION_END, COLUMNS, DATE, GIFT_CAT_RE, GIFT_TITLE_RE,
                     ID, ID_PATTERN, OFFICIAL_MD5, PRICE, QTY, STATE, TITLE)

DATE_FORMAT = "%Y-%m-%d"


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify_file(path, expected_md5=None):
    path = Path(path)
    exp = expected_md5 if expected_md5 is not None else OFFICIAL_MD5.get(path.name)
    got = md5sum(path)
    return {"file": path.name, "bytes": path.stat().st_size, "md5": got,
            "md5_expected": exp, "md5_match": got == exp}


def gift_flag(category, title):
    """Gift-card rule of rfm_v3 (category GIFT_CARD/GIFT_XXRD or title keywords)."""
    return (category.str.contains(GIFT_CAT_RE, na=False, regex=True)
            | title.str.contains(GIFT_TITLE_RE, case=False, na=False, regex=True)).to_numpy()


def _row_keys(ch):
    """128-bit key of the raw text of all 8 columns, for exact-duplicate detection."""
    a = pd.util.hash_pandas_object(ch, index=False, hash_key="apf-dup-key-0001").to_numpy()
    b = pd.util.hash_pandas_object(ch, index=False, hash_key="apf-dup-key-0002").to_numpy()
    return a, b


def load_transactions(path, chunksize=250_000, strict=True):
    """Returns (tx, qc).

    tx: one row per CSV row, columns ID, DATE, PRICE, QTY, spend, CATEGORY, is_gift,
        is_dup_copy (2nd+ copy of a row identical on all 8 raw columns), is_post_collection.
    qc: quality counts computed on the raw text before any type conversion.
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != COLUMNS:
        raise ValueError(f"unexpected columns: {header}")
    qc = {"rows": 0, "missing": dict.fromkeys(COLUMNS, 0), "empty_or_blank": dict.fromkeys(COLUMNS, 0),
          "unparseable_dates": 0, "dates_with_time_component": 0,
          "non_numeric": {PRICE: 0, QTY: 0}, "non_positive": {PRICE: 0, QTY: 0},
          "non_integer_quantity": 0, "ids_not_matching_pattern": 0,
          "ids_with_surrounding_whitespace": 0}
    parts, ka, kb = [], [], []
    for ch in pd.read_csv(path, chunksize=chunksize, dtype=str, keep_default_na=False,
                          na_values=[""]):
        qc["rows"] += len(ch)
        for c in COLUMNS:
            qc["missing"][c] += int(ch[c].isna().sum())
            qc["empty_or_blank"][c] += int(ch[c].str.strip().eq("").sum())
        d = pd.to_datetime(ch[DATE], format=DATE_FORMAT, errors="coerce")
        qc["unparseable_dates"] += int((d.isna() & ch[DATE].notna()).sum())
        qc["dates_with_time_component"] += int((d.notna() & d.ne(d.dt.normalize())).sum())
        num = {}
        for c in (PRICE, QTY):
            x = pd.to_numeric(ch[c], errors="coerce")
            qc["non_numeric"][c] += int((x.isna() & ch[c].notna()).sum())
            qc["non_positive"][c] += int((x <= 0).sum())
            num[c] = x
        qc["non_integer_quantity"] += int((num[QTY] % 1 != 0).sum())
        ids = ch[ID]
        qc["ids_not_matching_pattern"] += int((~ids.str.fullmatch(ID_PATTERN, na=False)).sum())
        qc["ids_with_surrounding_whitespace"] += int(ids.ne(ids.str.strip()).sum())
        a, b = _row_keys(ch)
        ka.append(a); kb.append(b)
        parts.append(pd.DataFrame({
            ID: ids.to_numpy(), DATE: d.to_numpy(),
            PRICE: num[PRICE].to_numpy(), QTY: num[QTY].to_numpy(),
            CATEGORY: ch[CATEGORY].astype("category"),
            "is_gift": gift_flag(ch[CATEGORY], ch[TITLE]),
        }))
    tx = pd.concat(parts, ignore_index=True)
    tx[CATEGORY] = tx[CATEGORY].astype("category")
    keys = pd.DataFrame({"a": np.concatenate(ka), "b": np.concatenate(kb)})
    tx["is_dup_copy"] = keys.duplicated(keep="first").to_numpy()
    tx["spend"] = tx[PRICE] * tx[QTY]
    tx["is_post_collection"] = tx[DATE] > COLLECTION_END
    if strict and (qc["unparseable_dates"] or qc["missing"][DATE] or qc["missing"][ID]
                   or qc["non_numeric"][PRICE] or qc["non_numeric"][QTY]):
        raise ValueError(f"data-quality failure: {qc}")
    qc.update(
        respondents=int(tx[ID].nunique()),
        date_min=str(tx[DATE].min().date()), date_max=str(tx[DATE].max().date()),
        exact_duplicate_extra_copies=int(tx["is_dup_copy"].sum()),
        exact_duplicate_groups=int(keys[keys.duplicated(keep=False)].drop_duplicates().shape[0]),
        gift_rows=int(tx["is_gift"].sum()),
        gift_respondents=int(tx.loc[tx["is_gift"], ID].nunique()),
        rows_after_collection_end=int(tx["is_post_collection"].sum()),
        respondents_after_collection_end=int(tx.loc[tx["is_post_collection"], ID].nunique()),
        ids_case_insensitive_collisions=int(tx[ID].nunique() - tx[ID].str.lower().nunique()),
    )
    tx[ID] = tx[ID].astype("category")
    return tx, qc


def data_version(tx, version):
    """raw | dedup | nogift | dedup_nogift (rfm_v3 cell 6, unchanged). `tx` itself is not changed."""
    if version not in {"raw", "dedup", "nogift", "dedup_nogift"}:
        raise ValueError(version)
    m = pd.Series(True, index=tx.index)
    if "dedup" in version:
        m &= ~tx["is_dup_copy"]
    if "nogift" in version:
        m &= ~tx["is_gift"]
    return tx.loc[m]


def read_survey_ids(path):
    """Only the ID column of survey.csv is read: used to reconcile identifiers, never as features."""
    return pd.read_csv(path, usecols=[ID], dtype=str)[ID]
