"""Fixed study constants. Changing any of these is a methodological decision (see reports/)."""
import os
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
# Raw data stay at their original location, outside the project; never copied or modified.
DATA_DIR = Path(os.environ.get("APF_DATA_DIR", PROJECT_DIR.parent))
PURCHASES_CSV = DATA_DIR / "amazon-purchases.csv"
SURVEY_CSV = DATA_DIR / "survey.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Official Dataverse v1.0 checksums (from rfm_v3/rfm_study_v3.ipynb cell 4, unchanged).
OFFICIAL_MD5 = {
    "amazon-purchases.csv": "3549d72735b1211ab009ff0177b94bd9",
    "survey.csv": "8340a44a5691c3763702683155cb5cc7",
}

# Column names (rfm_v3 cell 4, unchanged).
ID, DATE = "Survey ResponseID", "Order Date"
PRICE, QTY = "Purchase Price Per Unit", "Quantity"
STATE, TITLE, ASIN, CATEGORY = ("Shipping Address State", "Title",
                                "ASIN/ISBN (Product Code)", "Category")
COLUMNS = [DATE, PRICE, QTY, STATE, TITLE, ASIN, CATEGORY, ID]
ID_PATTERN = r"^R_[A-Za-z0-9]{15}$"

# Observation period. Exports run from 2018-01-01 to each participant's survey date, and the
# survey ran November 2022 - 20 March 2023 (Berke et al., 2024, Sci. Data 11:491,
# doi:10.1038/s41597-024-03329-6). Every history is therefore complete through 2022-10-31.
START = pd.Timestamp("2018-01-01")
OBSERVED_THROUGH = pd.Timestamp("2022-10-31")
COLLECTION_END = pd.Timestamp("2023-03-20")

# Target: >= 1 eligible purchase day in (cutoff, cutoff + 90 days], both ends in calendar days.
HORIZON_DAYS = 90
# 15 quarter-end cutoffs; each has >= 12 months of history and a fully observed window.
CUTOFFS = list(pd.date_range("2018-12-31", "2022-06-30", freq="QE"))

# Gift-card rule (rfm_v3 cell 4, unchanged). The descriptor's own rule (title keywords) is not
# fully published; its count is 40,368 rows.
GIFT_TITLE_RE = r"\bgift\s*cards?\b|\begift\b|\bgift\s*certificates?\b"
GIFT_CAT_RE = r"GIFT_(?:CARD|XXRD)"
DESCRIPTOR_GIFT_ROWS = 40368

# Survey variables were answered in 2022-23, after every cutoff: never a predictive feature.
# All survey questions are named "Q-..." (fields.csv); survey.csv is read only for ID checks.
SURVEY_COLUMN_PREFIX = "Q-"
