"""Built-in textbook triangles and a loader for the CAS Schedule P database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .triangle import from_long

# Taylor & Ashe (1983), the triangle used in Mack (1993). Cumulative, 10x10.
_GENINS = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
    [396132, 1333217, 2180715, 2985752, 3691712],
    [440832, 1288463, 2419861, 3483130],
    [359480, 1421128, 2864498],
    [376686, 1363294],
    [344014],
]

# Reinsurance Association of America triangle, 1981-1990. Cumulative, 10x10.
_RAA = [
    [5012, 8269, 10907, 11805, 13539, 16181, 18009, 18608, 18662, 18834],
    [106, 4285, 5396, 10666, 13782, 15599, 15496, 16169, 16704],
    [3410, 8992, 13873, 16141, 18735, 22214, 22863, 23466],
    [5655, 11555, 15766, 21266, 23425, 26083, 27067],
    [1092, 9565, 15836, 22169, 25955, 26180],
    [1513, 6445, 11702, 12935, 15852],
    [557, 4020, 10946, 12314],
    [1351, 6947, 13112],
    [3133, 5395],
    [2063],
]


def _to_frame(rows: list[list[float]], first_origin: int) -> pd.DataFrame:
    n = len(rows)
    data = np.full((n, n), np.nan)
    for i, row in enumerate(rows):
        data[i, : len(row)] = row
    return pd.DataFrame(
        data,
        index=pd.Index(range(first_origin, first_origin + n), name="origin"),
        columns=pd.Index(range(1, n + 1), name="dev"),
    )


def genins() -> pd.DataFrame:
    """Taylor & Ashe general insurance triangle (Mack 1993 example)."""
    return _to_frame(_GENINS, 1)


def raa() -> pd.DataFrame:
    """RAA casualty triangle, accident years 1981-1990."""
    return _to_frame(_RAA, 1981)


# --- CAS Loss Reserving Database -------------------------------------------

LINES = {
    "ppauto": "Private passenger auto liability/medical",
    "comauto": "Commercial auto/truck liability/medical",
    "wkcomp": "Workers' compensation",
    "medmal": "Medical malpractice - claims made",
    "othliab": "Other liability - occurrence",
    "prodliab": "Products liability - occurrence",
}


def _find(columns: list[str], prefix: str) -> str:
    for col in columns:
        if col.lower().startswith(prefix.lower()):
            return col
    raise KeyError(f"no column starting with {prefix!r}; got {columns}")


def load_cas(path: str | Path) -> pd.DataFrame:
    """Load one CAS Schedule P line file into a tidy long DataFrame.

    The raw files use line-specific suffixes (``CumPaidLoss_B``,
    ``CumPaidLoss_D`` ...). This normalises them to:
    company, name, origin, dev, paid, incurred, premium, single.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    cols = list(raw.columns)
    out = pd.DataFrame(
        {
            "company": raw[_find(cols, "GRCODE")],
            "name": raw[_find(cols, "GRNAME")].astype(str).str.strip(),
            "origin": raw[_find(cols, "AccidentYear")].astype(int),
            "dev": raw[_find(cols, "DevelopmentLag")].astype(int),
            "paid": raw[_find(cols, "CumPaidLoss")].astype(float),
            "incurred": raw[_find(cols, "IncurLoss")].astype(float),
            "premium": raw[_find(cols, "EarnedPremNet")].astype(float),
        }
    )
    try:
        out["single"] = raw[_find(cols, "Single")]
    except KeyError:
        pass
    return out


def company_square(
    data: pd.DataFrame, company: int, value: str = "paid"
) -> tuple[pd.DataFrame, pd.Series]:
    """Full 10x10 square (upper + actual lower triangle) and premium by origin."""
    sub = data[data["company"] == company]
    if sub.empty:
        raise KeyError(f"company {company} not found")
    full = from_long(sub, "origin", "dev", value)
    premium = sub[sub["dev"] == 1].set_index("origin")["premium"].sort_index()
    premium.name = "premium"
    return full, premium


def usable_companies(data: pd.DataFrame, value: str = "paid") -> list[int]:
    """Companies whose full square is complete, positive and has positive premium.

    Many Schedule P entries are tiny, run-off or contain zeros/negatives.
    Those break ratio-based methods, so they are filtered out up front.
    """
    keep = []
    for company, sub in data.groupby("company"):
        full = from_long(sub, "origin", "dev", value)
        if full.shape != (10, 10) or full.isna().any().any():
            continue
        if (full.to_numpy() <= 0).any():
            continue
        premium = sub[sub["dev"] == 1]["premium"]
        if len(premium) != 10 or (premium <= 0).any():
            continue
        keep.append(int(company))
    return keep
