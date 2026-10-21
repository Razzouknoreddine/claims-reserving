"""Reproduce the published results of Mack (1993) and the RAA benchmark.

These are the strongest tests in the repo: the numbers come from the
literature, not from this code.
"""

import numpy as np

from reserving import ChainLadder, MackChainLadder
from reserving.datasets import genins, raa

# Mack (1993), Table 2: Taylor & Ashe triangle
GENINS_RESERVES = [0, 94634, 469511, 709638, 984889, 1419459, 2177641, 3920301, 4278972, 4625811]
GENINS_MACK_SE = [0, 75535, 121699, 133549, 261406, 411010, 558317, 875328, 971258, 1363155]
GENINS_TOTAL_RESERVE = 18680856
GENINS_TOTAL_SE = 2447095

RAA_TOTAL_RESERVE = 52135
RAA_TOTAL_SE = 26909


def test_genins_chain_ladder_reserves():
    cl = ChainLadder().fit(genins())
    np.testing.assert_allclose(cl.reserve_.to_numpy(), GENINS_RESERVES, atol=1)
    assert abs(cl.reserve_.sum() - GENINS_TOTAL_RESERVE) <= 1


def test_genins_mack_standard_errors():
    m = MackChainLadder().fit(genins())
    np.testing.assert_allclose(m.mack_se_.to_numpy(), GENINS_MACK_SE, atol=1)
    assert abs(m.total_mack_se_ - GENINS_TOTAL_SE) <= 1


def test_raa_totals():
    m = MackChainLadder().fit(raa())
    assert abs(m.total_reserve_ - RAA_TOTAL_RESERVE) <= 1
    assert abs(m.total_mack_se_ - RAA_TOTAL_SE) <= 1


def test_process_and_parameter_error_add_up():
    m = MackChainLadder().fit(genins())
    np.testing.assert_allclose(
        m.process_se_**2 + m.parameter_se_**2, m.mack_se_**2, rtol=1e-12
    )
