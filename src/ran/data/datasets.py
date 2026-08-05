from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, SupportsFloat, cast

import numpy as np
import numpy.typing as npt

from ran.data.config import parse_gaussian_config, sigma_to_covariance

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

type Nested[T] = T | list[Nested[T]]

type Batch = tuple[dict[str, npt.NDArray[np.double]], npt.NDArray[np.ubyte]]


class ArrayDataset:
    """In-memory (z, x, y) arrays with deterministic minibatching.

    Iterating yields ``({"z": ..., "x": ...}, y)`` batches of NumPy arrays. The
    final batch is short rather than dropped whenever the split length is not a
    multiple of ``batch_size``.

    Every split holds a view onto one shared pair of base arrays; slicing is
    done with fancy indexing at batch time, so splitting costs no extra memory.

    Arguments:
        z: Particle-level features, shape (n_events, dim).
        x: Detector-level features, shape (n_events, dim).
        y: Per-event class label (1 = data, 0 = MC), shape (n_events,).
        batch_size: Events per batch.
        shuffle: Re-permute the event order before every pass. Used for the
            training split; validation and test iterate in fixed order.
        seed: Seed for the reshuffling generator.

    Each pass draws its permutation from ``(seed, pass_index)`` rather than
    from a generator carried across passes, so the order an epoch sees depends
    only on how many passes preceded it -- not on who else has iterated this
    object. Call `reset` to return to the first pass; `train` does so at the
    start of every run, which is what lets two runs over one `DatasetSplits`
    (an ensemble loop over init seeds) see identical data.
    """

    def __init__(
        self,
        z: npt.NDArray[np.double],
        x: npt.NDArray[np.double],
        y: npt.NDArray[np.ubyte],
        batch_size: int = 128,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        if not (len(z) == len(x) == len(y)):
            raise ValueError(
                f"z, x, y must share a first dimension; got {len(z)}, {len(x)}, {len(y)}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.z = z
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._pass = 0

    def reset(self) -> None:
        """Rewind to the first pass, so iteration repeats from the start."""
        self._pass = 0

    def __len__(self) -> int:
        """Number of batches per pass."""
        return int(np.ceil(len(self.y) / self.batch_size))

    @property
    def n_events(self) -> int:
        return len(self.y)

    def __iter__(self) -> Iterator[Batch]:
        order: npt.NDArray[np.intp]
        if self.shuffle:
            order = np.random.default_rng([self.seed, self._pass]).permutation(
                len(self.y)
            )
            self._pass += 1
        else:
            order = np.arange(len(self.y))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            yield {"z": self.z[idx], "x": self.x[idx]}, self.y[idx]

    def as_arrays(
        self,
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.ubyte]]:
        """Return the whole split as flat (z, x, y) arrays, in stored order.

        Callers that just want every event (plotting, metrics, the baselines)
        should use this instead of concatenating an iteration.
        """
        return self.z, self.x, self.y.reshape(-1)


class DatasetSplits(NamedTuple):
    """
    Named tuple representing dataset splits.
    Fields:
        train (ArrayDataset)
        val (ArrayDataset)
        test (ArrayDataset)
    """

    train: ArrayDataset
    val: ArrayDataset
    test: ArrayDataset


class RAN_Dataset:
    """
    Dataset class for RAN.
    Arguments:
        batch_size (int)
        seed (int): Random seed.
        cache_dir (str | Path)
        val_fraction (float)
        test_fraction (float)
    Attributes:
        dataset (tuple of (z, x, y) arrays in shuffled order)
        splits (DatasetSplits)

    Methods:
        generate_gaussian_dataset
    """

    def __init__(
        self,
        batch_size: int = 128,
        seed: int = 42,
        cache_dir: str | Path = ".cache",
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
    ) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.cache_dir = Path(cache_dir)

        if test_fraction < 0 or test_fraction > 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if val_fraction < 0 or val_fraction > 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if val_fraction + test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")

        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.dataset: (
            tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.ubyte]]
            | None
        ) = None
        self.splits: DatasetSplits | None = None

    @staticmethod
    def _round_nested(obj: Nested[SupportsFloat], ndigits: int = 10) -> Nested[float]:
        """Recursively round floats in a nested list/scalar for stable hashing."""
        if isinstance(obj, list):
            return [RAN_Dataset._round_nested(v, ndigits) for v in obj]
        return round(float(obj), ndigits)

    def _cache_key(self, parsed: dict[str, Any], n_samples: int) -> str:
        """Hash the promoted covariance matrices for a canonical cache key."""
        key_data: dict[str, Nested[float]] = {
            "mu_gen": self._round_nested(parsed["mu_gen"].tolist()),
            "mu_true": self._round_nested(parsed["mu_true"].tolist()),
            "cov_gen": self._round_nested(parsed["cov_gen"].tolist()),
            "cov_true": self._round_nested(parsed["cov_true"].tolist()),
            "cov_detector": self._round_nested(parsed["cov_detector"].tolist()),
            "n_samples": n_samples,
            "seed": self.seed,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _cache_path(self, parsed: dict[str, Any], n_samples: int) -> Path:
        cache_key: str = self._cache_key(parsed, n_samples)
        return self.cache_dir / f"gaussian_{cache_key}.npz"

    def _build_dataset(
        self,
        z: npt.NDArray[np.double],
        x: npt.NDArray[np.double],
        y: npt.NDArray[np.ubyte],
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.ubyte]]:
        """Interleave the data and MC halves with one fixed-seed permutation.

        The arrays arrive as data (y=1) stacked on MC (y=0); the splits below
        are contiguous slices, so they would otherwise be single-class. This
        shuffle happens once and is not repeated per epoch -- it defines the
        event ordering the splits cut into.
        """
        rng: np.random.Generator = np.random.default_rng(self.seed)
        order: npt.NDArray[np.intp] = rng.permutation(len(y))
        return z[order], x[order], y[order]

    def _split_dataset(
        self,
        dataset: tuple[
            npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.ubyte]
        ],
    ) -> DatasetSplits:
        """Cut the shuffled arrays into contiguous train/val/test splits.

        Test is taken off the end, validation off the end of what remains, so
        train occupies the front -- matching the nested `split_dataset` calls
        this replaced. Only the training split reshuffles between epochs.
        """
        z, x, y = dataset
        n: int = len(y)
        n_test: int = int(n * self.test_fraction)
        n_non_test: int = n - n_test
        val_of_non_test: float = self.val_fraction / (1.0 - self.test_fraction)
        n_val: int = int(n_non_test * val_of_non_test)
        n_train: int = n_non_test - n_val
        if n_train < 1 or n_val < 1 or n_test < 1:
            raise ValueError(
                f"{n} events split into train={n_train}, val={n_val}, test={n_test}; "
                "every split needs at least one event"
            )

        def _slice(lo: int, hi: int, shuffle: bool) -> ArrayDataset:
            return ArrayDataset(
                z[lo:hi],
                x[lo:hi],
                y[lo:hi],
                batch_size=self.batch_size,
                shuffle=shuffle,
                seed=self.seed,
            )

        return DatasetSplits(
            train=_slice(0, n_train, shuffle=True),
            val=_slice(n_train, n_non_test, shuffle=False),
            test=_slice(n_non_test, n, shuffle=False),
        )

    def generate_gaussian_dataset(
        self,
        config_path: str | Path | None = None,
        params: dict | None = None,
        n_samples: int = 10**6,
    ) -> DatasetSplits:
        """
        Generate a multivariate Gaussian dataset.
        Arguments:
            config_path: Path to a YAML config file.
            params: Dict with keys mu_gen, mu_true, sigma_gen, sigma_true, sigma_detector.
            n_samples: Number of samples per class (data and MC).
        Returns:
            DatasetSplits
        Exactly one of config_path or params must be provided.
        """
        if (config_path is None) == (params is None):
            raise ValueError("Exactly one of config_path or params must be provided")
        parsed: dict[str, Any]
        if config_path is not None:
            parsed = parse_gaussian_config(config_path)
        else:
            mu_gen: npt.NDArray[np.double] = np.asarray(
                params["mu_gen"], dtype=np.double
            ).ravel()  # type: ignore
            mu_true: npt.NDArray[np.double] = np.asarray(
                params["mu_true"], dtype=np.double
            ).ravel()  # type: ignore
            dim: np.ubyte = mu_gen.shape[0]
            if mu_true.shape[0] != dim:
                raise ValueError(f"mu_true has dim {mu_true.shape[0]}, expected {dim}")
            parsed = {
                "dim": dim,
                "mu_gen": mu_gen,
                "mu_true": mu_true,
                "cov_gen": sigma_to_covariance(params["sigma_gen"], dim),  # type: ignore
                "cov_true": sigma_to_covariance(params["sigma_true"], dim),  # type: ignore
                "cov_detector": sigma_to_covariance(params["sigma_detector"], dim),  # type: ignore
            }

        dim: np.ubyte = parsed["dim"]  # type: ignore
        mu_gen: npt.NDArray[np.double] = parsed["mu_gen"]
        mu_true: npt.NDArray[np.double] = parsed["mu_true"]
        cov_gen: npt.NDArray[np.double] = parsed["cov_gen"]
        cov_true: npt.NDArray[np.double] = parsed["cov_true"]
        cov_detector: npt.NDArray[np.double] = parsed["cov_detector"]

        cache_path: Path = self._cache_path(parsed, n_samples)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            logger.info("Loading dataset from cache: %s", cache_path)
            with np.load(cache_path) as data:
                z = data["z"]
                x = data["x"]
                y = data["y"]
        else:
            rng: np.random.Generator = np.random.default_rng(self.seed)

            z_true: npt.NDArray[np.double] = rng.multivariate_normal(
                mu_true,
                cov_true,
                size=n_samples,
                check_valid="raise",
                method="svd",
            )
            z_gen: npt.NDArray[np.double] = rng.multivariate_normal(
                mu_gen,
                cov_gen,
                size=n_samples,
                check_valid="raise",
                method="svd",
            )

            L_det: npt.NDArray[np.double] = np.linalg.cholesky(
                cast("npt.NDArray", cov_detector), upper=False
            )

            s_data: npt.NDArray[np.double] = rng.standard_normal(size=z_true.shape)
            x_data: npt.NDArray[np.double] = z_true + s_data @ L_det.T

            s_sim: npt.NDArray[np.double] = rng.standard_normal(size=z_gen.shape)
            x_sim: npt.NDArray[np.double] = z_gen + s_sim @ L_det.T

            y_nat: npt.NDArray[np.ubyte] = np.ones(n_samples, dtype=np.ubyte)
            y_MC: npt.NDArray[np.ubyte] = np.zeros(n_samples, dtype=np.ubyte)

            z: npt.NDArray[np.double] = np.concatenate((z_true, z_gen), axis=0)
            x: npt.NDArray[np.double] = np.concatenate((x_data, x_sim), axis=0)
            y: npt.NDArray[np.ubyte] = np.concatenate((y_nat, y_MC), axis=0)

            np.savez_compressed(cache_path, z=z, x=x, y=y)
            logger.info("Generated and saved dataset to cache: %s", cache_path)

        self.dataset = self._build_dataset(z, x, y)
        self.splits = self._split_dataset(self.dataset)
        return self.splits

    def splits_from_arrays(
        self,
        z: npt.NDArray[np.double],
        x: npt.NDArray[np.double],
        y: npt.NDArray[np.ubyte],
    ) -> DatasetSplits:
        """Build train/val/test splits directly from in-memory (z, x, y) arrays.

        z (particle level) and x (detector level) must have matching first
        dimension; y is the per-event class label (1 = data, 0 = MC).
        """
        self.dataset = self._build_dataset(z, x, y)
        self.splits = self._split_dataset(self.dataset)
        return self.splits
