"""Backtest chain-ladder, Mack and Cape Cod on every usable company of each line.

Run after downloading the data:
    python scripts/download_cas_data.py
    python scripts/run_backtest.py

Writes:
    results/backtest_<line>.csv     one row per company
    results/backtest_summary.csv    one row per line
    figures/backtest_errors.png     relative errors by line and method
    figures/backtest_z.png          Mack z-scores vs. standard normal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reserving.backtest import backtest_square, summarise  # noqa: E402
from reserving.datasets import LINES, company_square, load_cas, usable_companies  # noqa: E402

BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def run_line(path: Path, value: str) -> pd.DataFrame:
    data = load_cas(path)
    rows = []
    for company in usable_companies(data, value):
        full, premium = company_square(data, company, value)
        try:
            res = backtest_square(full, premium)
        except (ValueError, ZeroDivisionError):
            continue
        res["company"] = company
        res["name"] = data.loc[data["company"] == company, "name"].iloc[0]
        rows.append(res)
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_errors(all_results: dict[str, pd.DataFrame], out: Path) -> None:
    lines = [k for k, v in all_results.items() if len(v)]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    style(ax)
    pos = np.arange(len(lines))
    for offset, col, color, label in [
        (-0.18, "cl_rel_error", BLUE, "Chain-ladder"),
        (0.18, "cc_rel_error", ORANGE, "Cape Cod (BF)"),
    ]:
        data = [all_results[k][col].clip(-1, 1).to_numpy() for k in lines]
        bp = ax.boxplot(
            data, positions=pos + offset, widths=0.3, patch_artist=True,
            showfliers=False, medianprops={"color": INK, "linewidth": 1.5},
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.85)
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set_color(color)
        ax.plot([], [], color=color, linewidth=8, label=label)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(pos, lines)
    ax.set_ylabel("(estimate - actual) / actual", color=MUTED)
    ax.set_title("Reserve error by line of business (clipped to +/-100%)", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_z(all_results: dict[str, pd.DataFrame], out: Path) -> None:
    z = pd.concat([v["z"] for v in all_results.values() if len(v)]).dropna().clip(-6, 6)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style(ax)
    ax.hist(z, bins=40, density=True, color=BLUE, edgecolor=SURFACE, linewidth=1, label="observed z")
    x = np.linspace(-6, 6, 300)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color=ORANGE, linewidth=2, label="N(0, 1)")
    ax.set_xlabel("z = (actual - CL reserve) / Mack SE", color=MUTED)
    ax.set_title("Is the Mack standard error well calibrated?", color=INK, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/88-97"))
    parser.add_argument("--value", choices=["paid", "incurred"], default="paid")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    all_results, summaries = {}, {}
    for line in LINES:
        path = args.data / f"{line}.csv"
        if not path.exists():
            print(f"missing {path} - run scripts/download_cas_data.py first")
            continue
        res = run_line(path, args.value)
        all_results[line] = res
        res.to_csv(f"results/backtest_{line}.csv", index=False)
        if len(res):
            summaries[line] = summarise(res)
            print(f"{line:9s} {len(res):4d} companies")

    if not summaries:
        return
    summary = pd.DataFrame(summaries).T
    summary.to_csv("results/backtest_summary.csv")
    print(summary.T.to_string(float_format="{:.3f}".format))
    plot_errors(all_results, Path("figures/backtest_errors.png"))
    plot_z(all_results, Path("figures/backtest_z.png"))
    print("figures written to figures/")


if __name__ == "__main__":
    main()
