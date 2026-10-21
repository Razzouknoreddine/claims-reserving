"""Mack (1993) distribution-free standard error of chain-ladder reserves.

Model assumptions (C = cumulative claims, origin i, lag k):

    (1) E[C_{i,k+1} | C_{i,1..k}]   = f_k * C_{i,k}
    (2) Var[C_{i,k+1} | C_{i,1..k}] = sigma_k^2 * C_{i,k}
    (3) origins are independent

Under these assumptions the volume-weighted chain-ladder factors are unbiased,
and Mack derives the mean squared error of prediction (MSEP) of the reserve,
split into process variance and parameter (estimation) error.

Reference: T. Mack, "Distribution-free calculation of the standard error of
chain ladder reserve estimates", ASTIN Bulletin 23(2), 1993.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .chainladder import ChainLadder
from .triangle import latest_position


def sigma_squared(tri: pd.DataFrame, ldf: pd.Series) -> pd.Series:
    """Estimate sigma_k^2 for every development step.

    sigma_k^2 = 1/(n_k - 1) * sum_i C_ik * (C_{i,k+1}/C_ik - f_k)^2

    The last step has only one observation, so it is extrapolated as in Mack:
    sigma_{n-1}^2 = min(sigma_{n-2}^4 / sigma_{n-3}^2, sigma_{n-3}^2, sigma_{n-2}^2)
    """
    c = tri.to_numpy()
    f = ldf.to_numpy()
    m = len(f)
    s2 = np.full(m, np.nan)
    for k in range(m):
        known = ~np.isnan(c[:, k + 1]) & (c[:, k] > 0)
        n_k = known.sum()
        if n_k > 1:
            ck, cnext = c[known, k], c[known, k + 1]
            s2[k] = np.sum((cnext - f[k] * ck) ** 2 / ck) / (n_k - 1)
    for k in range(m):
        if np.isnan(s2[k]):
            if k >= 2 and s2[k - 2] > 0:
                s2[k] = min(s2[k - 1] ** 2 / s2[k - 2], s2[k - 2], s2[k - 1])
            elif k >= 1:
                s2[k] = s2[k - 1]
            else:
                s2[k] = 0.0
    return pd.Series(s2, index=ldf.index, name="sigma2")


class MackChainLadder:
    """Chain-ladder with Mack standard errors (no tail factor)."""

    def fit(self, tri: pd.DataFrame) -> "MackChainLadder":
        cl = ChainLadder().fit(tri)
        self.cl_ = cl
        self.ldf_ = cl.ldf_
        self.sigma2_ = sigma_squared(tri, cl.ldf_)

        c = tri.to_numpy()
        full = cl.full_.to_numpy()
        f = cl.ldf_.to_numpy()
        s2 = self.sigma2_.to_numpy()
        n_origin, n_dev = c.shape
        pos = latest_position(tri)

        # S_k: column sums used as the denominator of f_k
        s_k = np.array([c[~np.isnan(c[:, k + 1]), k].sum() for k in range(n_dev - 1)])
        ult = full[:, -1]

        process = np.zeros(n_origin)
        param = np.zeros(n_origin)
        for i in range(n_origin):
            for k in range(pos[i], n_dev - 1):
                w = s2[k] / f[k] ** 2
                process[i] += w / full[i, k]
                param[i] += w / s_k[k]
        process *= ult**2
        param *= ult**2
        mse = process + param

        # Total MSEP: individual MSEPs plus covariance from shared factor estimates
        total = mse.sum()
        for i in range(n_origin):
            younger = ult[i + 1 :].sum() if i + 1 < n_origin else 0.0
            cov = sum(2 * s2[k] / f[k] ** 2 / s_k[k] for k in range(pos[i], n_dev - 1))
            total += ult[i] * younger * cov

        idx = tri.index
        self.reserve_ = cl.reserve_
        self.ultimate_ = cl.ultimate_
        self.process_se_ = pd.Series(np.sqrt(process), index=idx, name="process_se")
        self.parameter_se_ = pd.Series(np.sqrt(param), index=idx, name="parameter_se")
        self.mack_se_ = pd.Series(np.sqrt(mse), index=idx, name="mack_se")
        self.total_reserve_ = float(cl.reserve_.sum())
        self.total_mack_se_ = float(np.sqrt(total))
        return self

    def summary(self) -> pd.DataFrame:
        out = pd.concat([self.cl_.latest_, self.ultimate_, self.reserve_, self.mack_se_], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cv"] = self.mack_se_ / self.reserve_
        return out

    def residuals(self, tri: pd.DataFrame) -> pd.DataFrame:
        """Weighted residuals r_ik = (C_{i,k+1} - f_k C_ik) / sqrt(sigma_k^2 C_ik).

        Under assumptions (1)-(2) they should look like noise with no trend
        against origin, lag or calendar period.
        """
        c = tri.to_numpy()
        f = self.ldf_.to_numpy()
        s2 = self.sigma2_.to_numpy()
        rows = []
        for k in range(c.shape[1] - 1):
            for i in range(c.shape[0]):
                if np.isnan(c[i, k + 1]) or c[i, k] <= 0 or s2[k] <= 0:
                    continue
                r = (c[i, k + 1] - f[k] * c[i, k]) / np.sqrt(s2[k] * c[i, k])
                rows.append(
                    {
                        "origin": tri.index[i],
                        "dev": tri.columns[k],
                        "calendar": i + k,
                        "fitted": f[k] * c[i, k],
                        "residual": r,
                    }
                )
        return pd.DataFrame(rows)
