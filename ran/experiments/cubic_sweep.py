"""Sweep the cubic detector-distortion strength s and compare RAN vs OmniFold.

For each s in linspace(0, 20, n_points), apply a deterministic non-linear
detector response r(s, z) = z + s * z**3 to fixed particle-level samples
(z_truth ~ N(0,1), z_gen ~ N(-1,1)), unfold z_gen back toward z_truth with both
RAN and OmniFold, and record Wasserstein(z_truth, z_unfolded) for each.

Usage:
    python -m ran.experiments.cubic_sweep run_point --s_index=0 --sweep_dir=...
    python -m ran.experiments.cubic_sweep collect --sweep_dir=...
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import fire
import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance


def response(s: float, z: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Deterministic non-linear detector response r(s, z) = z + s * z**3."""
    return z + s * z**3


def make_particles(
    n_samples: int, seed: int = 42
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Draw fixed particle-level samples: z_truth ~ N(0,1), z_gen ~ N(-1,1)."""
    rng = np.random.default_rng(seed)
    z_truth = rng.normal(0.0, 1.0, size=n_samples)
    z_gen = rng.normal(-1.0, 1.0, size=n_samples)
    return z_truth, z_gen


def unfolded_wasserstein(
    z_truth: npt.NDArray[np.double],
    z_gen: npt.NDArray[np.double],
    weights: npt.NDArray[np.double],
) -> float:
    """Wasserstein distance between z_truth and the weighted z_gen distribution."""
    return float(
        wasserstein_distance(
            np.asarray(z_truth).ravel(),
            np.asarray(z_gen).ravel(),
            v_weights=np.asarray(weights).ravel(),
        )
    )
