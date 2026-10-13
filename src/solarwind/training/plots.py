"""Figures for the model comparison: one run, and the best models against persistence."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_run(rows: pd.DataFrame, test_label: str, path: Path) -> None:
    """Test period of one model: actual and predicted over time, and predicted against actual."""
    test = rows[rows["split"] == test_label].dropna(subset=["prediction", "actual"])
    test = test.sort_values("date")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.6))
    left.plot(test["date"], test["actual"], color="black", lw=1.1, label="actual")
    left.plot(test["date"], test["prediction"], lw=0.9, label="predicted")
    left.set_title("test period")
    left.legend()
    left.grid(alpha=0.3)
    right.scatter(test["actual"], test["prediction"], s=6, alpha=0.5)
    top = max(test["actual"].max(), test["prediction"].max())
    right.plot([0, top], [0, top], color="red", lw=0.8)
    right.set_xlabel("actual CF")
    right.set_ylabel("predicted CF")
    right.set_title("predicted vs actual")
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_best(long: pd.DataFrame, best: dict[str, str], test_label: str, path: Path) -> None:
    """One panel per target: actual, persistence and the best model on the test rows."""
    targets = sorted(best)
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3.6 * len(targets)), sharex=True)
    axes = [axes] if len(targets) == 1 else list(axes)
    for ax, target in zip(axes, targets, strict=True):
        rows = long[(long["target"] == target) & (long["split"] == test_label)]
        actual = rows.drop_duplicates("date").set_index("date")["actual"].sort_index()
        ax.plot(actual.index, actual.values, color="black", lw=1.2, label="actual")
        for label in ("persistence", best[target]):
            series = rows[rows["baseline"] == label].set_index("date")["prediction"].sort_index()
            ax.plot(series.index, series.values, lw=0.9, alpha=0.85, label=label)
        ax.set_title(f"{target}: test period")
        ax.grid(alpha=0.3)
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
