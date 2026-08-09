from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..rantypes import ZXY, DatasetSplits, Events, GaussianConfig, Populations
from .config import parse_gaussian_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from logging import Logger
    from typing import Final, SupportsFloat

    from numpy.typing import NDArray

    from ..rantypes import Batch, Nested

logger: Logger = logging.getLogger(__name__)

_ONE_SOURCE_ONLY: Final = "Exactly one of config_path or params must be provided"


class ArrayDataset[T: np.floating = np.double]:
    """In-memory (z, x, y) arrays with deterministic minibatching."""

    def __init__(
        self,
        data: ZXY[T],
        batch_size: int = 128,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._pass = 0

    @property
    def size(self) -> int:
        return len(self.data)

    def reset(self) -> None:
        """Rewind to the first pass, so iteration repeats from the start."""
        self._pass = 0

    def __len__(self) -> int:
        """Number of batches per pass."""
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch[T]]:
        if self.shuffle:
            order: NDArray[np.intp] = np.random.default_rng(
                [self.seed, self._pass]
            ).permutation(self.size)
            self._pass += 1
        else:
            order = np.arange(self.size)
        for start in range(0, len(order), self.batch_size):
            idx: NDArray[np.intp] = order[start : start + self.batch_size]
            yield {"z": self.data.z[idx], "x": self.data.x[idx]}, self.data.y[idx]

    def as_arrays(self) -> ZXY[T]:
        """Return the whole split as flat labelled arrays, in stored order."""
        return self.data


class RANDataset:
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
        self.dataset: ZXY | None = None
        self.splits: DatasetSplits | None = None

    @staticmethod
    def _round_nested(obj: Nested[SupportsFloat], ndigits: int = 10) -> Nested[float]:
        """Recursively round floats in a nested list/scalar for stable hashing."""
        if isinstance(obj, list):
            return [RANDataset._round_nested(v, ndigits) for v in obj]
        return round(float(obj), ndigits)

    def _cache_key(self, parsed: GaussianConfig, n_samples: int) -> str:
        """Hash the promoted covariance matrices for a canonical cache key."""
        key_data: dict[str, Nested[float]] = {
            "mu_gen": self._round_nested(parsed.mu_gen.tolist()),
            "mu_true": self._round_nested(parsed.mu_true.tolist()),
            "cov_gen": self._round_nested(parsed.cov_gen.tolist()),
            "cov_true": self._round_nested(parsed.cov_true.tolist()),
            "cov_detector": self._round_nested(parsed.cov_detector.tolist()),
            "n_samples": n_samples,
            "seed": self.seed,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _cache_path(self, parsed: GaussianConfig, n_samples: int) -> Path:
        cache_key: str = self._cache_key(parsed, n_samples)
        return self.cache_dir / f"gaussian_{cache_key}.npz"

    def _build_dataset(self, data: ZXY) -> ZXY:
        """Shuffle so that both classes are spread across every split."""
        rng: np.random.Generator = np.random.default_rng(self.seed)
        order: NDArray[np.intp] = rng.permutation(len(data))
        return ZXY(Events(data.z[order], data.x[order]), data.y[order])

    def _split_dataset(self, dataset: ZXY) -> DatasetSplits:
        n: int = len(dataset)
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
                ZXY(
                    Events(dataset.z[lo:hi], dataset.x[lo:hi]),
                    dataset.y[lo:hi],
                ),
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
        config_path: Path | None = None,
        params: GaussianConfig | None = None,
        n_samples: int = 10**6,
    ) -> DatasetSplits:
        # Written as a nested check rather than a single XOR so that each branch
        # narrows the argument it goes on to use.
        if params is not None:
            if config_path is not None:
                raise ValueError(_ONE_SOURCE_ONLY)
            parsed: GaussianConfig = params
        elif config_path is None:
            raise ValueError(_ONE_SOURCE_ONLY)
        else:
            parsed = parse_gaussian_config(config_path)

        mu_gen: NDArray[np.double] = parsed.mu_gen
        mu_true: NDArray[np.double] = parsed.mu_true
        cov_gen: NDArray[np.double] = parsed.cov_gen
        cov_true: NDArray[np.double] = parsed.cov_true
        cov_detector: NDArray[np.double] = parsed.cov_detector

        cache_path: Path = self._cache_path(parsed, n_samples)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            logger.info("Loading dataset from cache: %s", cache_path)
            with np.load(cache_path) as cached:
                data = ZXY(Events(cached["z"], cached["x"]), cached["y"])
        else:
            rng: np.random.Generator = np.random.default_rng(self.seed)

            z_true: NDArray[np.double] = rng.multivariate_normal(
                mu_true,
                cov_true,
                size=n_samples,
                check_valid="raise",
                method="svd",
            )
            z_gen: NDArray[np.double] = rng.multivariate_normal(
                mu_gen,
                cov_gen,
                size=n_samples,
                check_valid="raise",
                method="svd",
            )

            chol_det: NDArray[np.double] = np.linalg.cholesky(cov_detector, upper=False)

            s_data: NDArray[np.double] = rng.standard_normal(size=z_true.shape)
            x_data: NDArray[np.double] = z_true + s_data @ chol_det.T

            s_sim: NDArray[np.double] = rng.standard_normal(size=z_gen.shape)
            x_sim: NDArray[np.double] = z_gen + s_sim @ chol_det.T

            data = Populations(
                mc=Events(z_gen, x_sim), data=x_data, truth=z_true
            ).interleave()

            np.savez_compressed(cache_path, z=data.z, x=data.x, y=data.y)
            logger.info("Generated and saved dataset to cache: %s", cache_path)

        return self.splits_from_data(data)

    def splits_from_data(self, data: ZXY) -> DatasetSplits:
        """Shuffle one labelled sample and cut it into train/val/test."""
        self.dataset = self._build_dataset(data)
        self.splits = self._split_dataset(self.dataset)
        return self.splits
