"""Tests for the loss-curve figure.

`losses.pdf` carried two identical dashed lines for a long time: `train.py`
recorded the validation BCE into both the `val_d` and `val_g` history columns,
and `plot_losses` faithfully drew both. Nothing failed --- the figure just
claimed to show something it did not measure --- so these pin the shape of the
plot rather than only the shape of the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ran.data import ArrayDataset
from ran.plotting import (
    _DETECTOR,
    _hist_ratio_panel,
    _plot_level,
    _save_fig,
    plot_levels,
    plot_losses,
    plot_selection,
)
from ran.rantypes import Events, Populations

if TYPE_CHECKING:
    from pathlib import Path

type DrawnCalls = list[tuple[tuple[Any, ...], dict[str, Any]]]


def test_filled_histograms_use_one_artist_per_distribution() -> None:
    """Falling back to Matplotlib's default bar histogram creates one
    rectangle per bin and makes the large vector PDFs needlessly expensive.
    """
    figure = Figure()
    ax = figure.add_subplot(211)
    ax_r = figure.add_subplot(212)
    nature = np.array([0.1, 0.3, 0.6, 0.8], dtype=np.single)
    mc = np.array([0.2, 0.4, 0.5, 0.9], dtype=np.single)

    _hist_ratio_panel(
        ax,
        ax_r,
        nature,
        mc,
        np.ones(4, dtype=np.single),
        bins=[0.0, 0.25, 0.5, 0.75, 1.0],
        nature_label="Data",
        mc_label="Sim",
        xlabel="x",
        title="Detector level",
    )

    assert len(ax.patches) == 3


def test_save_fig_uses_the_figure_page_without_a_second_tight_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reintroducing ``bbox_inches='tight'`` causes an extra traversal of
    every artist in the tall multi-panel figures.
    """
    calls: list[tuple[Path, dict[str, Any]]] = []

    def record(_figure: Figure, fname: Path, **kwargs: Any) -> None:
        calls.append((fname, kwargs))

    monkeypatch.setattr(Figure, "savefig", record)
    out = tmp_path / "figure.pdf"

    _save_fig(Figure(), out)

    assert calls == [(out, {})]


def test_multilevel_figure_uses_fixed_physical_edge_margins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default fractional margins turn into several blank inches as the
    stacked figure grows taller; the outer panels should use the page instead.
    """
    captured: list[Figure] = []

    def capture(figure: Figure, save_path: Path) -> None:
        del save_path
        captured.append(figure)

    monkeypatch.setattr("ran.plotting._save_fig", capture)
    values = np.array(
        [[-1.0, -0.5], [0.0, 0.2], [0.5, 0.8], [1.0, 1.2]], dtype=np.single
    )
    _plot_level(
        values,
        values + np.single(0.1),
        np.ones(4, dtype=np.single),
        _DETECTOR,
        tmp_path / "levels.pdf",
        None,
        None,
    )

    figure = captured[0]
    top = max(ax.get_position().y1 for ax in figure.axes)
    bottom = min(ax.get_position().y0 for ax in figure.axes)
    assert top >= 0.95
    assert bottom <= 0.05


def test_plot_levels_evaluates_generator_once(tmp_path: Path) -> None:
    """Detector and particle panels use identical generator weights, so the
    combined workflow must not repeat the forward pass.
    """
    z_gen = np.array([[-1.0], [0.0], [1.0]], dtype=np.single)
    x_sim = z_gen + np.single(0.1)
    truth = np.array([[-0.8], [0.1], [1.2]], dtype=np.single)
    data = truth + np.single(0.1)
    populations = Populations.create(mc=Events(z_gen, x_sim), data=data, truth=truth)
    dataset = ArrayDataset(populations.interleave(), batch_size=2)
    calls = 0

    def generator(z: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones((len(z), 1), dtype=np.single)

    detector = tmp_path / "detector.pdf"
    particle = tmp_path / "particle.pdf"
    plot_levels(dataset, generator, detector, particle)  # ty: ignore[invalid-argument-type]

    assert calls == 1
    assert detector.exists()
    assert particle.exists()


def test_plot_levels_uses_the_same_page_height_for_matching_panel_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both level figures have the same panel geometry; making the particle
    page much taller adds whitespace without adding information.
    """
    z_gen = np.array([[-1.0], [0.0], [1.0]], dtype=np.single)
    events = Events(z_gen, z_gen + np.single(0.1))
    populations = Populations.create(mc=events, data=z_gen, truth=z_gen)
    dataset = ArrayDataset(populations.interleave(), batch_size=2)
    captured: list[Figure] = []

    def capture(figure: Figure, save_path: Path) -> None:
        del save_path
        captured.append(figure)

    monkeypatch.setattr("ran.plotting._save_fig", capture)
    plot_levels(
        dataset,
        lambda z: np.ones((len(z), 1), dtype=np.single),
        tmp_path / "detector.pdf",
        tmp_path / "particle.pdf",
    )  # ty: ignore[invalid-argument-type]

    assert captured[0].get_figheight() == captured[1].get_figheight()


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
def drawn(monkeypatch: pytest.MonkeyPatch) -> DrawnCalls:
    """Record every `ax.plot` call instead of rendering it.

    `plot_losses` builds its own Figure and returns nothing, so intercepting the
    Axes is the only way to assert on what ends up in the legend.
    """
    calls: DrawnCalls = []

    def record(_ax: Axes, *args: Any, **kwargs: Any) -> list[Any]:
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(Axes, "plot", record)
    return calls


class TestLossCurves:
    def test_validation_is_drawn_once(self, drawn: DrawnCalls, tmp_path: Path) -> None:
        plot_losses(_history(), save_path=tmp_path / "losses.pdf")

        labels = [k.get("label", "") for _, k in drawn]
        assert sum(label.startswith("Val") for label in labels) == 1
        assert labels == ["Train D", "Train G", "Val D"]

    def test_no_two_curves_carry_the_same_data(
        self, drawn: DrawnCalls, tmp_path: Path
    ) -> None:
        """The actual regression: `val_g` was a literal copy of `val_d`."""
        plot_losses(_history(), save_path=tmp_path / "losses.pdf")

        series = [
            np.asarray(a=cast("list[float]", args[1]), dtype=np.double)
            for args, _ in drawn
        ]
        for i, first in enumerate(series):
            for second in series[i + 1 :]:
                assert not np.array_equal(first, second)

    def test_a_legacy_four_column_history_still_plots(
        self, drawn: DrawnCalls, tmp_path: Path
    ) -> None:
        """Runs saved before the merge carry a `val_g` key holding a copy of
        `val_d`. `--load-run` replots them, so reading it must stay optional ---
        and it must stay unread, or the duplicate comes back."""
        legacy = _history()
        legacy["val_g"] = list(legacy["val_d"])

        plot_losses(legacy, save_path=tmp_path / "losses.pdf")

        assert [k.get("label", "") for _, k in drawn] == ["Train D", "Train G", "Val D"]


@pytest.fixture
def captured_axes(monkeypatch: pytest.MonkeyPatch) -> list[Axes]:
    """Capture every Axes `plot_selection` builds via `Figure.add_subplot`,
    without mocking away the real rendering -- unlike `drawn`, this fixture
    lets matplotlib actually compute scales, transforms and view limits, which
    is what the symlog/log regression lives in.
    """
    captured: list[Axes] = []
    original_add_subplot = Figure.add_subplot

    def record_and_call(self: Figure, *args: Any, **kwargs: Any) -> Axes:
        ax = original_add_subplot(self, *args, **kwargs)
        captured.append(ax)
        return ax

    monkeypatch.setattr(Figure, "add_subplot", record_and_call)
    return captured


class TestSelectionPlot:
    def test_selection_plot_survives_a_missing_particle_curve(
        self, tmp_path: Path
    ) -> None:
        """A real measurement has no truth, so the particle curve is optional."""
        history = {
            "train_d": [0.69] * 5,
            "train_g": [0.69] * 5,
            "val_d": [0.69] * 5,
            "val_mmd": [0.05, 0.03, 0.01, 0.02, 0.04],
            "val_ess": [900.0, 850.0, 800.0, 700.0, 600.0],
        }
        out = tmp_path / "selection.pdf"
        plot_selection(history, best_epoch=2, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

        history["val_mmd_particle"] = [0.09, 0.07, 0.06, 0.06, 0.07]
        plot_selection(history, best_epoch=2, save_path=out)
        assert out.exists()

    def test_negative_and_zero_mmd_survive_a_log_style_axis(
        self, captured_axes: list[Axes], tmp_path: Path
    ) -> None:
        """`weighted_mmd` is the unbiased estimator, negative roughly half the
        time once the distributions actually match, since MMD^2 is 0 at
        P = Q. A plain log axis silently masks non-positive values -- the
        line vanishes and the view autoscales away from them -- exactly in
        the neighbourhood of a converged run's minimum. This pins both the
        axis scale and the actual view limits, since the masking happens at
        render/autoscale time, not by mutating the stored data: a check on
        `Line2D.get_ydata()` alone would pass under the old `set_yscale("log")`
        code too.
        """
        history = {
            "train_d": [0.69] * 6,
            "train_g": [0.69] * 6,
            "val_d": [0.69] * 6,
            "val_mmd": [6.6e-4, -9.3e-4, -1.0e-4, 2.0e-4, -5.0e-4, 0.0],
            "val_ess": [900.0] * 6,
        }
        plot_selection(history, best_epoch=2, save_path=tmp_path / "selection.pdf")

        ax = captured_axes[0]
        assert ax.get_yscale() != "log"

        mmd_line = ax.get_lines()[0]
        ydata = np.asarray(mmd_line.get_ydata(), dtype=np.double)
        assert (ydata < 0).any()

        ax.figure.canvas.draw()
        ylim = ax.get_ylim()
        assert ylim[0] < 0, (
            "y-limits must reach the negative data; a log axis clips the view "
            "to the smallest positive value and silently drops the rest"
        )

    def test_negative_best_epoch_skips_the_selection_marker(
        self, captured_axes: list[Axes], tmp_path: Path
    ) -> None:
        """`best_epoch` and `val_mmd` are read from different files
        (`config.json` vs `history.npz`) and can diverge; `best_epoch`
        defaults to -1. `best_epoch + 1` in the label would then read
        "selected (epoch 0)" while the axvline lands off the left edge."""
        history = {
            "train_d": [0.69] * 5,
            "train_g": [0.69] * 5,
            "val_d": [0.69] * 5,
            "val_mmd": [0.05, 0.03, 0.01, 0.02, 0.04],
            "val_ess": [900.0, 850.0, 800.0, 700.0, 600.0],
        }
        plot_selection(history, best_epoch=-1, save_path=tmp_path / "selection.pdf")

        ax = captured_axes[0]
        _, labels = ax.get_legend_handles_labels()
        assert not any(label.startswith("selected") for label in labels)
