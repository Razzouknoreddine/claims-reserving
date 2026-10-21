"""Backtesting reserve estimates against what actually happened.

The CAS Schedule P database contains the *full* square: the upper triangle as
known at year-end 1997 plus the later development. We cut the square at the
valuation date, estimate reserves from the upper part only, and compare with
the actual outstanding amount from the lower part.

No information from below the diagonal may leak into the estimate. That is
why BF uses the Cape Cod ELR (estimated from the upper triangle) rather than
the realised loss ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bf import CapeCod
from .mack import MackChainLadder
from .triangle import latest_diagonal, upper


def actual_reserve(full: pd.DataFrame) -> float:
    """Realised outstanding amount: value at the last lag minus the diagonal.

    Note: in a 10x10 square the last lag is not necessarily fully ultimate,
    so this is the 'actual to lag 10' amount, the standard choice in the
    literature (e.g. Meyers, CAS Monograph 8).
    """
    return float(full.iloc[:, -1].sum() - latest_diagonal(upper(full)).sum())


def backtest_square(full: pd.DataFrame, premium: pd.Series) -> dict:
    """Run CL (with Mack SE) and Cape Cod on one company and compare to actual."""
    tri = upper(full)
    mack = MackChainLadder().fit(tri)
    cc = CapeCod().fit(tri, premium)
    actual = actual_reserve(full)
    cl_res = mack.total_reserve_
    cc_res = float(cc.reserve_.sum())
    se = mack.total_mack_se_
    return {
        "actual": actual,
        "cl_reserve": cl_res,
        "cc_reserve": cc_res,
        "cc_elr": cc.elr,
        "mack_se": se,
        "cl_rel_error": (cl_res - actual) / actual if actual else np.nan,
        "cc_rel_error": (cc_res - actual) / actual if actual else np.nan,
        "z": (actual - cl_res) / se if se > 0 else np.nan,
    }


def summarise(results: pd.DataFrame) -> pd.Series:
    """Aggregate metrics over many companies."""
    z = results["z"].dropna()
    return pd.Series(
        {
            "companies": len(results),
            "CL median rel. error": results["cl_rel_error"].median(),
            "CL median |rel. error|": results["cl_rel_error"].abs().median(),
            "Cape Cod median rel. error": results["cc_rel_error"].median(),
            "Cape Cod median |rel. error|": results["cc_rel_error"].abs().median(),
            "share actual within CL +/- 1.96 Mack SE": (z.abs() <= 1.96).mean(),
        }
    )
