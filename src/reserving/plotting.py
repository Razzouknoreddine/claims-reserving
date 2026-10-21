"""Small, consistent matplotlib helpers used by the notebook and scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def style(ax, grid_axis: str = "y"):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def plot_development(tri: pd.DataFrame, ax=None, scale: float = 1e6, unit: str = "m"):
    """Cumulative claims by lag, one line per origin; older origins drawn lighter."""
    ax = style(ax or plt.subplots(figsize=(8, 4.5))[1])
    n = len(tri)
    for i, (origin, row) in enumerate(tri.iterrows()):
        shade = 0.25 + 0.75 * i / max(n - 1, 1)
        ax.plot(row.index, row.to_numpy() / scale, color=BLUE, alpha=shade, linewidth=2)
        last = row.dropna()
        ax.annotate(
            str(origin), (last.index[-1], last.iloc[-1] / scale),
            xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED, va="center",
        )
    ax.set_xlabel("development lag (years)", color=MUTED)
    ax.set_ylabel(f"cumulative paid ({unit})", color=MUTED)
    ax.set_title("Development of cumulative paid claims", color=INK, loc="left")
    return ax


def plot_reserves(mack, cape_cod=None, ax=None, scale: float = 1e6, unit: str = "m"):
    """Reserve by origin: chain-ladder with +/- 1.96 Mack SE, optional Cape Cod."""
    ax = style(ax or plt.subplots(figsize=(8, 4.5))[1])
    origins = np.arange(len(mack.reserve_))
    width = 0.38 if cape_cod is not None else 0.6
    shift = -width / 2 if cape_cod is not None else 0
    ax.bar(
        origins + shift, mack.reserve_ / scale, width=width, color=BLUE,
        yerr=1.96 * mack.mack_se_ / scale, ecolor=INK, capsize=3,
        error_kw={"linewidth": 1}, label="Chain-ladder (+/- 1.96 Mack SE)",
    )
    if cape_cod is not None:
        ax.bar(origins + width / 2, cape_cod.reserve_ / scale, width=width, color=ORANGE, label="Cape Cod")
    ax.set_xticks(origins, mack.reserve_.index)
    ax.set_xlabel("origin period", color=MUTED)
    ax.set_ylabel(f"reserve ({unit})", color=MUTED)
    ax.set_title("Reserve by origin period", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left")
    return ax


def plot_residuals(res: pd.DataFrame, axes=None):
    """Mack weighted residuals against origin, lag and calendar period."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, col, label in zip(axes, ["origin", "dev", "calendar"], ["origin", "lag", "calendar period"]):
        style(ax, "both")
        ax.scatter(res[col], res["residual"], s=28, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
        means = res.groupby(col)["residual"].mean()
        ax.plot(means.index, means.to_numpy(), color=ORANGE, linewidth=2, label="mean")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xlabel(label, color=MUTED)
    axes[0].set_ylabel("weighted residual", color=MUTED)
    axes[0].legend(frameon=False)
    axes[0].figure.suptitle("Mack residual diagnostics", color=INK, x=0.01, ha="left")
    return axes
