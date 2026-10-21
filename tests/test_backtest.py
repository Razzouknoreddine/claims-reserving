"""Tests for the CAS loader and the backtest logic, using synthetic data."""

import numpy as np
import pandas as pd

from reserving.backtest import actual_reserve, backtest_square, summarise
from reserving.datasets import company_square, load_cas, usable_companies


def simulate_square(seed: int, n: int = 10) -> pd.DataFrame:
    """Simulate a full square from the Mack model with known factors."""
    rng = np.random.default_rng(seed)
    f = np.array([2.5, 1.6, 1.3, 1.15, 1.08, 1.05, 1.03, 1.02, 1.01])
    sigma2 = np.array([400, 200, 100, 60, 30, 20, 10, 5, 2.0])
    c = np.empty((n, n))
    c[:, 0] = rng.uniform(800, 1200, n) * 100
    for k in range(n - 1):
        mean = f[k] * c[:, k]
        sd = np.sqrt(sigma2[k] * c[:, k])
        c[:, k + 1] = np.maximum(mean + rng.normal(0, sd), c[:, k])
    return pd.DataFrame(c, index=range(1988, 1988 + n), columns=range(1, n + 1))


def test_actual_reserve_definition():
    full = simulate_square(0)
    expected = full.iloc[:, -1].sum() - sum(full.iloc[i, 9 - i] for i in range(10))
    assert np.isclose(actual_reserve(full), expected)


def test_mack_interval_coverage_on_simulated_data():
    """When the data follow the Mack model, ~95% of actuals fall in CL +/- 1.96 SE."""
    rows = []
    for seed in range(200):
        full = simulate_square(seed)
        premium = full.iloc[:, -1] / 0.7
        rows.append(backtest_square(full, premium))
    res = pd.DataFrame(rows)
    coverage = summarise(res)["share actual within CL +/- 1.96 Mack SE"]
    assert 0.85 <= coverage <= 1.0
    assert abs(res["cl_rel_error"].median()) < 0.05


def test_load_cas_normalises_columns(tmp_path):
    rows = []
    for company in (1, 2):
        for ay in range(1988, 1998):
            for lag in range(1, 11):
                value = 0.0 if (company == 2 and lag == 3) else 100.0 * lag
                rows.append(
                    {
                        "GRCODE": company,
                        "GRNAME": f"Insurer {company}",
                        "AccidentYear": ay,
                        "DevelopmentYear": ay + lag - 1,
                        "DevelopmentLag": lag,
                        "IncurLoss_D": value + 50,
                        "CumPaidLoss_D": value,
                        "BulkLoss_D": 0,
                        "EarnedPremDIR_D": 2000,
                        "EarnedPremCeded_D": 100,
                        "EarnedPremNet_D": 1900,
                        "Single": 1,
                        "PostedReserve97_D": 0,
                    }
                )
    path = tmp_path / "wkcomp_pos.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    data = load_cas(path)
    assert {"company", "origin", "dev", "paid", "incurred", "premium"} <= set(data.columns)

    full, premium = company_square(data, 1)
    assert full.shape == (10, 10)
    assert full.loc[1990, 4] == 400.0
    assert (premium == 1900).all()

    # company 2 has zeros in the triangle and must be filtered out
    assert usable_companies(data) == [1]
