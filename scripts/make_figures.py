"""Create the README figures from the built-in Taylor & Ashe triangle."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reserving import CapeCod, MackChainLadder  # noqa: E402
from reserving.datasets import genins  # noqa: E402
from reserving.plotting import plot_development, plot_reserves, plot_residuals  # noqa: E402

# Illustrative premium (Taylor & Ashe publish no premium): flat 8m per year.
PREMIUM = pd.Series(8e6, index=range(1, 11), name="premium")


def main() -> None:
    out = Path("figures")
    out.mkdir(exist_ok=True)
    tri = genins()
    mack = MackChainLadder().fit(tri)
    cc = CapeCod().fit(tri, PREMIUM)

    ax = plot_development(tri)
    ax.figure.tight_layout()
    ax.figure.savefig(out / "genins_development.png", dpi=150)

    ax = plot_reserves(mack, cc)
    ax.figure.tight_layout()
    ax.figure.savefig(out / "genins_reserves.png", dpi=150)

    axes = plot_residuals(mack.residuals(tri))
    axes[0].figure.tight_layout()
    axes[0].figure.savefig(out / "genins_residuals.png", dpi=150)
    plt.close("all")
    print("written:", *sorted(p.name for p in out.glob("genins_*.png")))


if __name__ == "__main__":
    main()
