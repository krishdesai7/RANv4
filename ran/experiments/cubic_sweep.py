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

from ran.data.datasets import RAN_Dataset
from ran.train import train
from ran.baselines.omnifold import omnifold_unfold


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


def run_point(
    s_index: int,
    sweep_dir: str | Path,
    n_samples: int = 500_000,
    n_points: int = 25,
    seed: int = 42,
    batch_size: int = 1024,
    ran_epochs: int = 100,
    omnifold_niter: int = 3,
    omnifold_epochs: int = 50,
    omnifold_batch_size: int = 512,
) -> dict:
    """Train RAN and OmniFold at one sweep point and write s_{index}.json.

    Particle samples are drawn once (fixed seed) so only s varies across points.
    RAN uses g(z_gen) weights; OmniFold uses its gen-level model weights. Both
    z_unfolded distributions are the z_gen samples reweighted accordingly.
    """
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    s = float(np.linspace(0.0, 20.0, n_points)[s_index])
    z_truth, z_gen = make_particles(n_samples, seed=seed)
    x_data = response(s, z_truth)
    x_sim = response(s, z_gen)

    # --- RAN: data events carry z_truth (y=1, weight fixed to 1), sim events
    # carry z_gen (y=0, reweighted by g). Matches generate_gaussian_dataset. ---
    z = np.concatenate([z_truth, z_gen]).reshape(-1, 1).astype(np.double)
    x = np.concatenate([x_data, x_sim]).reshape(-1, 1).astype(np.double)
    y = np.concatenate(
        [np.ones(n_samples, dtype=np.ubyte), np.zeros(n_samples, dtype=np.ubyte)]
    )
    splits = RAN_Dataset(batch_size=batch_size).splits_from_arrays(z, x, y)
    g, _, _ = train(splits, dim=1, n_epochs=ran_epochs)

    raw = g(z_gen.reshape(-1, 1).astype(np.double)).numpy().ravel()
    w_ran = raw * len(raw) / raw.sum()

    # --- OmniFold: reweight z_gen toward z_truth via reco-level unfolding ---
    # OmniFold derives validation_steps = (0.2 * NTRAIN) // batch_size from the
    # ~2*n_samples reco events; if that floors to 0 its model.fit hangs forever
    # on a repeating validation dataset. Cap the batch size at 2*n/5 so there is
    # always >= 1 validation step (no-op at the real n=500k scale).
    of_batch = max(1, min(omnifold_batch_size, 2 * n_samples // 5))
    w_of = omnifold_unfold(
        x_data.reshape(-1, 1),
        x_sim.reshape(-1, 1),
        z_gen.reshape(-1, 1),
        niter=omnifold_niter,
        epochs=omnifold_epochs,
        batch_size=of_batch,
    )

    # Guard against non-finite weights (e.g. a saturated OmniFold classifier at
    # large s) so the Wasserstein call cannot crash; bad weights -> 0 mass.
    w_ran = np.where(np.isfinite(w_ran), w_ran, 0.0)
    w_of = np.where(np.isfinite(w_of), w_of, 0.0)

    ran_wd = unfolded_wasserstein(z_truth, z_gen, w_ran)
    of_wd = unfolded_wasserstein(z_truth, z_gen, w_of)

    out = {"s_index": int(s_index), "s": s, "ran_wd": ran_wd, "omnifold_wd": of_wd}
    (sweep_dir / f"s_{s_index:02d}.json").write_text(json.dumps(out, indent=2))
    print(f"s={s:.4f}  RAN={ran_wd:.6f}  OmniFold={of_wd:.6f}")
    return out


def collect(sweep_dir: str | Path, n_points: int = 25) -> None:
    """Gather all s_*.json into results.npz and a Wasserstein-vs-s PDF."""
    sweep_dir = Path(sweep_dir)
    records = [
        json.loads(f.read_text()) for f in sorted(sweep_dir.glob("s_*.json"))
    ]
    if not records:
        raise FileNotFoundError(f"No s_*.json files found in {sweep_dir}")
    records.sort(key=lambda r: r["s"])

    present = {r["s_index"] for r in records}
    missing = sorted(set(range(n_points)) - present)
    if missing:
        print(f"WARNING: missing s_index values (failed/incomplete tasks): {missing}")

    s = np.array([r["s"] for r in records])
    ran = np.array([r["ran_wd"] for r in records])
    omnifold = np.array([r["omnifold_wd"] for r in records])
    np.savez(sweep_dir / "results.npz", s=s, ran=ran, omnifold=omnifold)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s, ran, "o-", label="RAN")
    ax.plot(s, omnifold, "s-", label="OmniFold")
    ax.set_xlabel(r"$s$ (cubic distortion strength)")
    ax.set_ylabel(r"Wasserstein($z_\mathrm{truth}$, $z_\mathrm{unfolded}$)")
    ax.set_title("Unfolding performance vs detector distortion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(sweep_dir / "wasserstein_vs_s.pdf")
    plt.close(fig)
    print(
        f"Wrote {sweep_dir / 'results.npz'} and "
        f"{sweep_dir / 'wasserstein_vs_s.pdf'} ({len(records)} points)"
    )


if __name__ == "__main__":
    fire.Fire({"run_point": run_point, "collect": collect})
