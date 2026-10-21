"""Loss triangle utilities.

A triangle is represented as a ``pandas.DataFrame``:

* index   = origin (accident) periods, oldest first
* columns = development lags 1, 2, ..., n
* values  = cumulative amounts; unknown future cells are ``NaN``

Keeping a plain DataFrame (instead of a custom class) makes every step easy to
inspect in a notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def from_long(
    df: pd.DataFrame,
    origin: str,
    dev: str,
    value: str,
) -> pd.DataFrame:
    """Pivot long-format data (one row per origin/lag) into a triangle."""
    tri = df.pivot_table(index=origin, columns=dev, values=value, aggfunc="sum")
    tri = tri.sort_index().sort_index(axis=1)
    tri.index.name = "origin"
    tri.columns.name = "dev"
    return tri.astype(float)


def to_incremental(tri: pd.DataFrame) -> pd.DataFrame:
    """Cumulative -> incremental (first column unchanged)."""
    inc = tri.diff(axis=1)
    inc.iloc[:, 0] = tri.iloc[:, 0]
    return inc


def to_cumulative(inc: pd.DataFrame) -> pd.DataFrame:
    """Incremental -> cumulative, keeping unknown cells as NaN."""
    cum = inc.cumsum(axis=1, skipna=False)
    return cum.where(inc.notna())


def upper(full: pd.DataFrame) -> pd.DataFrame:
    """Mask a full square down to what was known at the valuation date.

    Origin ``i`` (0-based, oldest first) of an ``n x n`` square is known up to
    lag ``n - i``. Everything below that anti-diagonal is set to ``NaN``.
    """
    n_rows, n_cols = full.shape
    if n_rows != n_cols:
        raise ValueError(f"expected a square, got {n_rows}x{n_cols}")
    i = np.arange(n_rows)[:, None]
    j = np.arange(n_cols)[None, :]
    return full.where(i + j < n_rows)


def latest_diagonal(tri: pd.DataFrame) -> pd.Series:
    """Most recent known value for each origin."""
    return tri.ffill(axis=1).iloc[:, -1].rename("latest")


def latest_position(tri: pd.DataFrame) -> np.ndarray:
    """0-based column position of the latest known value for each origin."""
    known = tri.notna().to_numpy()
    if not known[:, 0].all():
        raise ValueError("every origin needs a value at the first lag")
    return known.shape[1] - 1 - np.argmax(known[:, ::-1], axis=1)


def validate(tri: pd.DataFrame) -> None:
    """Raise if the triangle is not a well-formed run-off triangle."""
    if tri.shape[0] != tri.shape[1]:
        raise ValueError(f"expected a square layout, got {tri.shape}")
    expected = upper(pd.DataFrame(np.ones(tri.shape))).notna().to_numpy()
    if not np.array_equal(tri.notna().to_numpy(), expected):
        raise ValueError("triangle must be fully known above the anti-diagonal and NaN below")
