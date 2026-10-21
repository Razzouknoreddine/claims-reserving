"""Unit tests for triangle handling, chain-ladder, BF and Cape Cod."""

import numpy as np
import pandas as pd

from reserving import (
    BornhuetterFerguson,
    CapeCod,
    ChainLadder,
    development_factors,
    from_long,
    latest_diagonal,
    to_cumulative,
    to_incremental,
    upper,
)
from reserving.datasets import genins


def small_triangle():
    return pd.DataFrame(
        [[100, 150, 165], [110, 160, np.nan], [120, np.nan, np.nan]],
        index=[2021, 2022, 2023],
        columns=[1, 2, 3],
        dtype=float,
    )


def test_incremental_roundtrip():
    tri = genins()
    back = to_cumulative(to_incremental(tri))
    pd.testing.assert_frame_equal(back, tri)


def test_from_long_pivots():
    long = pd.DataFrame(
        {"o": [1, 1, 2], "d": [1, 2, 1], "v": [10.0, 15.0, 12.0]}
    )
    tri = from_long(long, "o", "d", "v")
    assert tri.loc[1, 2] == 15.0
    assert np.isnan(tri.loc[2, 2])


def test_upper_masks_below_antidiagonal():
    full = pd.DataFrame(np.ones((4, 4)))
    known = upper(full).notna().to_numpy()
    assert known.sum() == 4 + 3 + 2 + 1
    assert known[0].all() and known[3, 0] and not known[3, 1]


def test_development_factors_by_hand():
    f = development_factors(small_triangle()).to_numpy()
    np.testing.assert_allclose(f, [(150 + 160) / (100 + 110), 165 / 150])


def test_chain_ladder_by_hand():
    cl = ChainLadder().fit(small_triangle())
    f1, f2 = (150 + 160) / 210, 165 / 150
    expected_ult = [165, 160 * f2, 120 * f1 * f2]
    np.testing.assert_allclose(cl.ultimate_.to_numpy(), expected_ult)
    np.testing.assert_allclose(cl.full_.iloc[:, -1].to_numpy(), expected_ult)


def test_tail_factor_scales_ultimate():
    tri = small_triangle()
    base = ChainLadder().fit(tri).ultimate_
    tail = ChainLadder(tail=1.05).fit(tri).ultimate_
    np.testing.assert_allclose(tail.to_numpy(), base.to_numpy() * 1.05)


def test_bf_equals_cl_when_prior_is_cl_ultimate():
    """If the prior ultimate equals the CL ultimate, BF reproduces CL."""
    tri = genins()
    cl = ChainLadder().fit(tri)
    premium = pd.Series(1.0, index=tri.index)
    bf = BornhuetterFerguson(elr=cl.ultimate_).fit(tri, premium)
    np.testing.assert_allclose(bf.reserve_.to_numpy(), cl.reserve_.to_numpy())


def test_bf_by_hand():
    tri = small_triangle()
    premium = pd.Series([200.0, 200.0, 200.0], index=tri.index)
    bf = BornhuetterFerguson(elr=0.8).fit(tri, premium)
    cdf = ChainLadder().fit(tri).cdf_at_latest_.to_numpy()
    np.testing.assert_allclose(bf.reserve_.to_numpy(), 160 * (1 - 1 / cdf))


def test_cape_cod_reproduces_total_latest():
    """Cape Cod ELR is chosen so the expected reported losses match the actual ones."""
    tri = genins()
    premium = pd.Series(np.linspace(8e6, 1e7, 10), index=tri.index)
    cc = CapeCod().fit(tri, premium)
    expected_reported = premium * cc.elr / cc.cl_.cdf_at_latest_
    assert np.isclose(expected_reported.sum(), latest_diagonal(tri).sum())


def test_invalid_triangle_raises():
    tri = small_triangle()
    tri.iloc[2, 1] = 5.0  # value below the diagonal
    try:
        ChainLadder().fit(tri)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
