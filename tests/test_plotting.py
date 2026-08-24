"""Tests for the loss-curve figure.

`losses.pdf` carried two identical dashed lines for a long time: `train.py`
recorded the validation BCE into both the `val_d` and `val_g` history columns,
and `plot_losses` faithfully drew both. Nothing failed --- the figure just
claimed to show something it did not measure --- so these pin the shape of the
plot rather than only the shape of the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from matplotlib.axes import Axes
from ran.plotting import plot_losses

if TYPE_CHECKING:
    from pathlib import Path


def _history(n: int = 12) -> dict[str, list[float]]:
    """What `train` returns now: one validation column, not two."""
    rng = np.random.default_rng(0)
    val = (0.69 + rng.normal(scale=0.01, size=n)).tolist()
    return {
        "train_d": (0.68 + rng.normal(scale=0.01, size=n)).tolist(),
        "train_g": (-0.68 + rng.normal(scale=0.01, size=n)).tolist(),
        "val_d": val,
    }


@pytest.fixture
def drawn(monkeypatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Record every `ax.plot` call instead of rendering it.

    `plot_losses` builds its own Figure and returns nothing, so intercepting the
    Axes is the only way to assert on what ends up in the legend.
    """
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record(_ax: Axes, *args: Any, **kwargs: Any) -> list[Any]:
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(Axes, "plot", record)
    return calls


class TestLossCurves:
    def test_validation_is_drawn_once(self, drawn, tmp_path: Path) -> None:
        plot_losses(_history(), save_path=tmp_path / "losses.pdf")

        labels = [k.get("label", "") for _, k in drawn]
        assert sum(label.startswith("Val") for label in labels) == 1
        assert labels == ["Train D", "Train G", "Val D"]

    def test_no_two_curves_carry_the_same_data(self, drawn, tmp_path: Path) -> None:
        """The actual regression: `val_g` was a literal copy of `val_d`."""
        plot_losses(_history(), save_path=tmp_path / "losses.pdf")

        series = [np.asarray(args[1], dtype=np.double) for args, _ in drawn]
        for i, first in enumerate(series):
            for second in series[i + 1 :]:
                assert not np.array_equal(first, second)

    def test_a_legacy_four_column_history_still_plots(
        self, drawn, tmp_path: Path
    ) -> None:
        """Runs saved before the merge carry a `val_g` key holding a copy of
        `val_d`. `--load-run` replots them, so reading it must stay optional ---
        and it must stay unread, or the duplicate comes back."""
        legacy = _history()
        legacy["val_g"] = list(legacy["val_d"])

        plot_losses(legacy, save_path=tmp_path / "losses.pdf")

        assert [k.get("label", "") for _, k in drawn] == ["Train D", "Train G", "Val D"]
