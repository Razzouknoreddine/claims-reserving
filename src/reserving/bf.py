"""Bornhuetter-Ferguson and Cape Cod methods.

Both use the chain-ladder development pattern but take the *expected* ultimate
from premium x expected loss ratio (ELR) instead of from the latest diagonal:

    reserve_i = premium_i * ELR * (1 - 1 / CDF_i)

The share (1 - 1/CDF_i) is the part of the ultimate that is still unreported.
For young origins this share is large, so BF leans on the prior; for old
origins it is small, so BF converges to chain-ladder. This is a credibility
weighting between the prior and the data.

Cape Cod estimates the ELR from the triangle itself, which matters for an
honest backtest: the prior must only use information available at the
valuation date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .chainladder import ChainLadder


class BornhuetterFerguson:
    def __init__(self, elr: float | pd.Series, tail: float = 1.0):
        self.elr = elr
        self.tail = tail

    def fit(self, tri: pd.DataFrame, premium: pd.Series) -> "BornhuetterFerguson":
        premium = premium.reindex(tri.index)
        if premium.isna().any() or (premium <= 0).any():
            raise ValueError("premium must be positive for every origin")
        cl = ChainLadder(tail=self.tail).fit(tri)
        self.cl_ = cl
        self.prior_ultimate_ = (premium * self.elr).rename("prior_ultimate")
        unreported = 1.0 - 1.0 / cl.cdf_at_latest_
        self.credibility_ = (1.0 / cl.cdf_at_latest_).rename("z")
        self.reserve_ = (self.prior_ultimate_ * unreported).rename("reserve")
        self.latest_ = cl.latest_
        self.ultimate_ = (self.latest_ + self.reserve_).rename("ultimate")
        return self

    def summary(self) -> pd.DataFrame:
        return pd.concat(
            [self.latest_, self.prior_ultimate_, self.credibility_, self.ultimate_, self.reserve_],
            axis=1,
        )


def cape_cod_elr(tri: pd.DataFrame, premium: pd.Series, tail: float = 1.0) -> float:
    """Cape Cod ELR = sum(latest) / sum(premium * reported share)."""
    cl = ChainLadder(tail=tail).fit(tri)
    used_premium = premium.reindex(tri.index) / cl.cdf_at_latest_
    return float(cl.latest_.sum() / used_premium.sum())


class CapeCod(BornhuetterFerguson):
    """Bornhuetter-Ferguson with the ELR estimated by the Cape Cod formula."""

    def __init__(self, tail: float = 1.0):
        super().__init__(elr=np.nan, tail=tail)

    def fit(self, tri: pd.DataFrame, premium: pd.Series) -> "CapeCod":
        self.elr = cape_cod_elr(tri, premium, self.tail)
        return super().fit(tri, premium)
