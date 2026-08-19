from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from ..rantypes import ZXY, DatasetSplits, Events, GaussianConfig, Populations
from .config import parse_gaussian_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from logging import Logger
    from typing import Final, LiteralString, SupportsFloat

    from numpy._typing import _DTypeLike
    from numpy.typing import DTypeLike, NDArray

    from ..rantypes import Batch, Nested

logger: Logger = logging.getLogger(name=__name__)

_ONE_SOURCE_ONLY: Final[LiteralString] = (
    "Exactly one of config_path or params must be provided"
)


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
        self.data: ZXY[T] = data
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.seed: int = seed
        self._pass = 0

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def dtype(self) -> np.dtype[T]:
        return self.data.dtype

    def reset(self) -> None:
        """Rewind to the first pass, so iteration repeats from the start."""
        self._pass = 0

    def __len__(self) -> int:
        """Number of batches per pass."""
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch[T]]:
        if self.shuffle:
            order: NDArray[np.intp] = np.random.default_rng(
                seed=[self.seed, self._pass]
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


class RANDataset[T: np.floating = np.double]:
    @overload
    def __init__(
        self: RANDataset[np.double],
        batch_size: int = 128,
        seed: int = 42,
        cache_dir: Path = Path(".cache"),
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
        dtype: _DTypeLike[np.double] = np.double,
    ) -> None: ...

    @overload
    def __init__(
        self,
        batch_size: int = 128,
        seed: int = 42,
        cache_dir: Path = Path(".cache"),
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
        *,
        dtype: _DTypeLike[T],
    ) -> None: ...

    @overload
    def __init__(
        self,
        batch_size: int,
        seed: int,
        cache_dir: Path,
        val_fraction: float,
        test_fraction: float,
        dtype: _DTypeLike[T],
    ) -> None: ...

    def __init__(
        self,
        batch_size: int = 128,
        seed: int = 42,
        cache_dir: Path = Path(".cache"),
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
        dtype: DTypeLike = np.double,
    ) -> None:
        self.batch_size: int = batch_size
        self.seed: int = seed
        self.cache_dir: Path = cache_dir
        self.dtype: np.dtype[T] = cast("np.dtype[T]", np.dtype(dtype))

        if test_fraction < 0 or test_fraction > 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if val_fraction < 0 or val_fraction > 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if val_fraction + test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")

        self.val_fraction: float = val_fraction
        self.test_fraction: float = test_fraction
        self.dataset: ZXY[T] | None = None
        self.splits: DatasetSplits[T] | None = None

    @staticmethod
    def _round_nested(
        obj: Nested[SupportsFloat], ndigits: int = 10, /
    ) -> Nested[float]:
        """Recursively round floats in a nested list/scalar for stable hashing."""
        if isinstance(obj, list):
            return [RANDataset._round_nested(v, ndigits) for v in obj]
        return np.round(a=float(obj), decimals=ndigits)

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
            data=json.dumps(obj=key_data, sort_keys=True).encode(encoding="utf-8")
        ).hexdigest()[:16]

    def _cache_path(self, parsed: GaussianConfig, n_samples: int) -> Path:
        cache_key: str = self._cache_key(parsed, n_samples)
        return self.cache_dir / f"gaussian_{cache_key}.npz"

    def _build_dataset(self, data: ZXY[T]) -> ZXY[T]:
        """Shuffle so that both classes are spread across every split."""
        rng: np.random.Generator = np.random.default_rng(self.seed)
        order: NDArray[np.intp] = rng.permutation(x=len(data))
        return ZXY(Events(data.z[order], data.x[order]), data.y[order])

    def _split_dataset(self, dataset: ZXY[T]) -> DatasetSplits[T]:
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

        def _slice(lo: int, hi: int, shuffle: bool) -> ArrayDataset[T]:
            return ArrayDataset(
                data=ZXY(
                    Events(dataset.z[lo:hi], dataset.x[lo:hi]),
                    dataset.y[lo:hi],
                ),
                batch_size=self.batch_size,
                shuffle=shuffle,
                seed=self.seed,
            )

        return DatasetSplits(
            train=_slice(0, n_train, shuffle=True),
            val=_slice(lo=n_train, hi=n_non_test, shuffle=False),
            test=_slice(lo=n_non_test, hi=n, shuffle=False),
        )

    def generate_gaussian_dataset(
        self,
        config_path: Path | None = None,
        *,
        params: GaussianConfig | None = None,
        n_samples: int = 10**6,
    ) -> DatasetSplits[T]:
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

        mu_gen: NDArray[T] = parsed.mu_gen.astype(dtype=self.dtype)
        mu_true: NDArray[T] = parsed.mu_true.astype(dtype=self.dtype)
        cov_gen: NDArray[T] = parsed.cov_gen.astype(dtype=self.dtype)
        cov_true: NDArray[T] = parsed.cov_true.astype(dtype=self.dtype)
        cov_detector: NDArray[T] = parsed.cov_detector.astype(dtype=self.dtype)

        cache_path: Path = self._cache_path(parsed, n_samples)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            logger.info("Loading dataset from cache: %s", cache_path)
            with np.load(file=cache_path) as cached:
                data: ZXY[T] = ZXY(
                    Events(
                        z=cached["z"].astype(dtype=self.dtype),
                        x=cached["x"].astype(dtype=self.dtype),
                    ),
                    y=cached["y"],
                )
        else:
            rng: np.random.Generator = np.random.default_rng(self.seed)

            z_true: NDArray[np.double] = rng.multivariate_normal(
                mean=mu_true,
                cov=cov_true,
                size=n_samples,
                check_valid="raise",
                method="svd",
            )
            z_gen: NDArray[np.double] = rng.multivariate_normal(
                mean=mu_gen,
                cov=cov_gen,
                size=n_samples,
                check_valid="raise",
                method="svd",
            )

            chol_det: NDArray[np.double] = np.linalg.cholesky(cov_detector, upper=False)

            s_data: NDArray[np.double] = rng.standard_normal(size=z_true.shape)
            x_data: NDArray[np.double] = z_true + s_data @ chol_det.T

            s_sim: NDArray[np.double] = rng.standard_normal(size=z_gen.shape)
            x_sim: NDArray[np.double] = z_gen + s_sim @ chol_det.T

            data: ZXY[T] = Populations(
                mc=Events(
                    z=z_gen.astype(dtype=self.dtype), x=x_sim.astype(dtype=self.dtype)
                ),
                data=x_data.astype(dtype=self.dtype),
                truth=z_true.astype(dtype=self.dtype),
            ).interleave()

            np.savez_compressed(file=cache_path, z=data.z, x=data.x, y=data.y)
            logger.info("Generated and saved dataset to cache: %s", cache_path)

        return self.splits_from_data(data)

    def splits_from_data(self, data: ZXY[T]) -> DatasetSplits[T]:
        """Shuffle one labelled sample and cut it into train/val/test."""
        self.dataset = self._build_dataset(data)
        self.splits = self._split_dataset(self.dataset)
        return self.splits
