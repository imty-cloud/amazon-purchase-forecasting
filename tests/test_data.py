"""Chunked loader: duplicates, gift cards, missingness, identifiers, strict dates (synthetic CSVs)."""
import csv
import hashlib

import pytest

from apf import config as C
from apf.data import data_version, gift_flag, load_transactions, verify_file

HEADER = C.COLUMNS
ID1, ID2 = "R_aaaaaaaaaaaaaaa", "R_bbbbbbbbbbbbbbb"


def write(path, rows, header=HEADER):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def row(date="2020-01-05", price="9.99", qty="1.0", state="NJ", title="Pen", asin="B0001",
        cat="PEN", rid=ID1):
    return [date, price, qty, state, title, asin, cat, rid]


@pytest.mark.parametrize("chunksize", [1, 2, 1000])
def test_exact_duplicates_flag_only_extra_copies_across_chunks(tmp_path, chunksize):
    rows = [row(), row(price="9.98"), row(), row(rid=ID2), row()]
    tx, qc = load_transactions(write(tmp_path / "p.csv", rows), chunksize=chunksize)
    assert tx["is_dup_copy"].tolist() == [False, False, True, False, True]
    assert qc["exact_duplicate_extra_copies"] == 2 and qc["exact_duplicate_groups"] == 1


def test_duplicate_detection_is_on_all_columns_including_missing(tmp_path):
    rows = [row(title=""), row(title=""), row(title="Pen ")]
    tx, _ = load_transactions(write(tmp_path / "p.csv", rows))
    assert tx["is_dup_copy"].tolist() == [False, True, False]


def test_gift_rule():
    import pandas as pd
    cat = pd.Series(["GIFT_CARD", "GIFT_XXRD", "PEN", None, "PEN", "PEN", "PEN"], dtype="string")
    tit = pd.Series(["x", "x", "Amazon.com eGift Card", "Visa Gift Cards 3-pack",
                     "Gift wrap paper", "Gifted kids book", None], dtype="string")
    assert gift_flag(cat, tit).tolist() == [True, True, True, True, False, False, False]


def test_missing_values_counted_and_rows_kept(tmp_path):
    rows = [row(title="", cat="", state=""), row(asin="")]
    tx, qc = load_transactions(write(tmp_path / "p.csv", rows))
    assert len(tx) == 2
    assert qc["missing"][C.TITLE] == 1 and qc["missing"][C.CATEGORY] == 1
    assert qc["missing"][C.STATE] == 1 and qc["missing"][C.ASIN] == 1


def test_literal_na_text_is_not_missing(tmp_path):
    tx, qc = load_transactions(write(tmp_path / "p.csv", [row(title="NA", state="NA")]))
    assert qc["missing"][C.TITLE] == 0


@pytest.mark.parametrize("bad", ["2020/01/05", "05-01-2020", "2020-02-30", "2020-01-05 10:00:00"])
def test_strict_date_format(tmp_path, bad):
    with pytest.raises(ValueError):
        load_transactions(write(tmp_path / "p.csv", [row(), row(date=bad)]))


def test_missing_id_is_fatal(tmp_path):
    with pytest.raises(ValueError):
        load_transactions(write(tmp_path / "p.csv", [row(rid="")]))


def test_identifier_checks(tmp_path):
    rows = [row(rid="R_short"), row(rid=" " + ID1), row(rid=ID1.upper().replace("R_", "R_", 1)), row()]
    _, qc = load_transactions(write(tmp_path / "p.csv", rows))
    assert qc["ids_not_matching_pattern"] == 2
    assert qc["ids_with_surrounding_whitespace"] == 1
    assert qc["ids_case_insensitive_collisions"] == 1


def test_quantity_and_price_checks(tmp_path):
    _, qc = load_transactions(write(tmp_path / "p.csv", [row(qty="1.5"), row(price="0"), row()]))
    assert qc["non_integer_quantity"] == 1 and qc["non_positive"][C.PRICE] == 1


def test_wrong_header_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_transactions(write(tmp_path / "p.csv", [row()], header=HEADER[::-1]))


def test_data_version_does_not_modify_input(tmp_path):
    rows = [row(), row(), row(cat="GIFT_CARD")]
    tx, _ = load_transactions(write(tmp_path / "p.csv", rows))
    before = tx.copy()
    assert len(data_version(tx, "dedup")) == 2 and len(data_version(tx, "nogift")) == 2
    assert len(data_version(tx, "dedup_nogift")) == 1
    assert tx.equals(before)
    with pytest.raises(ValueError):
        data_version(tx, "other")


def test_verify_file(tmp_path):
    p = write(tmp_path / "p.csv", [row()])
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    assert verify_file(p, md5)["md5_match"] and not verify_file(p, "0" * 32)["md5_match"]
