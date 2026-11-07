"""POST-HOC / exploratory helpers. Reuses src/apf; changes nothing in locked files.
Table is cached outside the project (APF_POST_CACHE, default: system temp dir)."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from apf.config import ID, OUTPUT_DIR, PURCHASES_CSV, SURVEY_CSV, OFFICIAL_MD5
from apf.data import md5sum
from apf.phase2 import _Model, load_table, model_specs
from apf.splits import TEST_CUTOFFS

OUT = ROOT / "outputs" / "exploratory"
OUT.mkdir(parents=True, exist_ok=True)
LOCK = json.loads((OUTPUT_DIR / "stage_d_locked_config.json").read_text())
CONF = LOCK["selected"]["configurations"]
SEEDS = LOCK["selected"]["seeds"]
BANNER = "POST-HOC, không dùng để đổi kết luận P1–P3"
TEST_SETS = {"temporal_test": "in_test_temporal", "holdout_test": "in_test_holdout"}
CACHE = Path(os.environ.get("APF_POST_CACHE", tempfile.gettempdir())) / "post_table.pkl"


def check_data():
    for p in (PURCHASES_CSV, SURVEY_CSV):
        got = md5sum(p)
        assert got == OFFICIAL_MD5[p.name], f"MD5 mismatch {p.name}: {got}"
    assert md5sum(PURCHASES_CSV) == LOCK["selected"]["data_md5"]


def get_table():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    check_data()
    t = load_table()
    t.to_pickle(CACHE)
    return t


def write_csv(df, name):
    """CSV with the POST-HOC banner as first comment line."""
    path = OUT / name
    with open(path, "w") as f:
        f.write(f"# {BANNER}\n")
        df.to_csv(f, index=False)
    return path


def fit_final(table, model_ids):
    refit = table[table["in_refit"]]
    assert not refit["holdout"].any()
    specs = model_specs(CONF)
    return {m: _Model(*specs[m]).fit(refit, refit["y"].to_numpy()) for m in model_ids}


def test_set(table, name):
    return table[table[TEST_SETS[name]]]
