"""Deterministic chain-ladder method."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .triangle import latest_diagonal, latest_position, validate


def development_factors(tri: pd.DataFrame) -> pd.Series:
    """Volume-weighted age-to-age factors.

    f_k = sum_i C[i, k+1] / sum_i C[i, k], summed over origins i for which
    C[i, k+1] is known.
    """
    c = tri.to_numpy()
    n = c.shape[1]
    f = np.empty(n - 1)
    for k in range(n - 1):
        known = ~np.isnan(c[:, k + 1])
        denom = c[known, k].sum()
        if denom <= 0:
            raise ValueError(f"column {k + 1} has a non-positive sum; cannot compute f_{k + 1}")
        f[k] = c[known, k + 1].sum() / denom
    labels = [f"{tri.columns[k]}-{tri.columns[k + 1]}" for k in range(n - 1)]
    return pd.Series(f, index=labels, name="ldf")


def cumulative_factors(ldf: pd.Series, tail: float = 1.0) -> pd.Series:
    """Age-to-ultimate factors: CDF_k = f_k * f_{k+1} * ... * tail."""
    f = np.append(ldf.to_numpy(), tail)
    cdf = np.cumprod(f[::-1])[::-1]
    return pd.Series(cdf, name="cdf")


class ChainLadder:
    """Classic chain-ladder projection.

    Attributes after ``fit``:
        ldf_      age-to-age factors
        cdf_      age-to-ultimate factor for each lag (indexed by lag)
        latest_   latest diagonal
        ultimate_ projected ultimate per origin
        reserve_  ultimate - latest (IBNR + IBNER, i.e. outstanding)
        full_     completed square
    """

    def __init__(self, tail: float = 1.0):
        if tail < 1.0:
            raise ValueError("tail factor should be >= 1")
        self.tail = tail

    def fit(self, tri: pd.DataFrame) -> "ChainLadder":
        validate(tri)
        self.ldf_ = development_factors(tri)
        cdf = cumulative_factors(self.ldf_, self.tail)
        cdf.index = tri.columns
        self.cdf_ = cdf

        pos = latest_position(tri)
        self.latest_ = latest_diagonal(tri)
        self.cdf_at_latest_ = pd.Series(cdf.to_numpy()[pos], index=tri.index, name="cdf")
        self.ultimate_ = (self.latest_ * self.cdf_at_latest_).rename("ultimate")
        self.reserve_ = (self.ultimate_ - self.latest_).rename("reserve")

        full = tri.to_numpy().copy()
        f = self.ldf_.to_numpy()
        for k in range(len(f)):
            missing = np.isnan(full[:, k + 1])
            full[missing, k + 1] = full[missing, k] * f[k]
        self.full_ = pd.DataFrame(full, index=tri.index, columns=tri.columns)
        return self

    def summary(self) -> pd.DataFrame:
        return pd.concat([self.latest_, self.cdf_at_latest_, self.ultimate_, self.reserve_], axis=1)
