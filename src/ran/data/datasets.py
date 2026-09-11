from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..rantypes import (
    CACHE_DIR,
    EVENT_DTYPE,
    ZXY,
    DatasetSplits,
    Events,
    GaussianConfig,
    Populations,
)
from ..timing import note
from .config import parse_gaussian_config

if TYPE_CHECKING:
    from collections.abc import Mapping
    from logging import Logger
    from pathlib import Path
    from typing import Any, Final, LiteralString, SupportsFloat

    from jax import Array
    from jaxtyping import Float
    from numpy.typing import NDArray

    from ..rantypes import Nested

logger: Logger = logging.getLogger(name=__name__)

_ONE_SOURCE_ONLY: Final[LiteralString] = (
    "Exactly one of config_path or params must be provided"
)

# Bumped whenever the generator changes, because the cache key is otherwise a
# pure function of the physics config -- an old .npz would be silently reused
# and hand back a sample drawn from a different stream.
_RNG_VERSION: Final[LiteralString] = "jax-v1"


def _draw_gaussian(
    seed: int,
    /,
    *,
    mu_true: NDArray[np.double],
    mu_gen: NDArray[np.double],
    cov_true: NDArray[np.double],
    cov_gen: NDArray[np.double],
    cov_detector: NDArray[np.double],
    n_samples: int,
) -> tuple[
    Float[Array, "n d"], Float[Array, "n d"], Float[Array, "n d"], Float[Array, "n d"]
]:
    k_true, k_gen, k_data, k_sim = jax.random.split(jax.random.key(seed), num=4)

    z_true: Float[Array, "n d"] = jax.random.multivariate_normal(
        k_true, mu_true, cov_true, shape=(n_samples,), method="svd"
    )
    z_gen: Float[Array, "n d"] = jax.random.multivariate_normal(
        k_gen, mu_gen, cov_gen, shape=(n_samples,), method="svd"
    )

    smear: Float[Array, "d d"] = jnp.linalg.cholesky(jnp.asarray(a=cov_detector)).T

    x_data: Float[Array, "n d"] = (
        z_true + jax.random.normal(key=k_data, shape=z_true.shape) @ smear
    )
    x_sim: Float[Array, "n d"] = (
        z_gen + jax.random.normal(key=k_sim, shape=z_gen.shape) @ smear
    )
    return z_true, z_gen, x_data, x_sim


class ArrayDataset:
    def __init__(
        self,
        data: ZXY,
        batch_size: int = 128,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.data: ZXY = data
        self.batch_size: int = batch_size
        self.seed: int = seed

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def dtype(self) -> np.dtype[np.single]:
        return self.data.dtype

    def __len__(self) -> int:
        return (self.size + self.batch_size - 1) // self.batch_size

    def as_arrays(self) -> ZXY:
        return self.data


class RANDataset:
    def __init__(
        self,
        batch_size: int = 128,
        seed: int = 42,
        cache_dir: Path = CACHE_DIR,
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
    ) -> None:
        self.batch_size: int = batch_size
        self.seed: int = seed
        self.cache_dir: Path = cache_dir
        self.dtype: np.dtype[np.single] = np.dtype(EVENT_DTYPE)

        if test_fraction < 0 or test_fraction > 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if val_fraction < 0 or val_fraction > 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if val_fraction + test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")

        self.val_fraction: float = val_fraction
        self.test_fraction: float = test_fraction
        self.dataset: ZXY | None = None
        self.splits: DatasetSplits | None = None

    @staticmethod
    def _round_nested(
        obj: Nested[SupportsFloat], ndigits: int = 10, /
    ) -> Nested[float]:
        """Recursively round floats in a nested list/scalar for stable hashing."""
        if isinstance(obj, list):
            return [RANDataset._round_nested(v, ndigits) for v in obj]
        return float(np.round(a=float(obj), decimals=ndigits))

    def _cache_key(self, parsed: GaussianConfig, n_samples: int) -> str:
        """Hash the promoted covariance matrices for a canonical cache key."""
        key_data: dict[str, Nested[float] | str] = {
            "mu_gen": self._round_nested(parsed.mu_gen.tolist()),
            "mu_true": self._round_nested(parsed.mu_true.tolist()),
            "cov_gen": self._round_nested(parsed.cov_gen.tolist()),
            "cov_true": self._round_nested(parsed.cov_true.tolist()),
            "cov_detector": self._round_nested(parsed.cov_detector.tolist()),
            "n_samples": n_samples,
            "seed": self.seed,
            "rng": _RNG_VERSION,
            # Without this a float32 and a float64 run share one file
            "dtype": str(object=self.dtype),
        }
        return hashlib.sha256(
            data=json.dumps(obj=key_data, sort_keys=True).encode(encoding="utf-8")
        ).hexdigest()[:16]

    def _cache_path(self, parsed: GaussianConfig, n_samples: int) -> Path:
        cache_key: str = self._cache_key(parsed, n_samples)
        return self.cache_dir / f"gaussian_{cache_key}.npz"

    def _build_dataset(self, data: ZXY) -> ZXY:
        """Shuffle so that both classes are spread across every split."""
        rng: np.random.Generator = np.random.default_rng(self.seed)
        order: NDArray[np.intp] = rng.permutation(x=len(data))
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

        def _slice(lo: int, hi: int) -> ArrayDataset:
            return ArrayDataset(
                data=ZXY(
                    Events(dataset.z[lo:hi], dataset.x[lo:hi]),
                    dataset.y[lo:hi],
                ),
                batch_size=self.batch_size,
                seed=self.seed,
            )

        return DatasetSplits(
            train=_slice(0, n_train),
            val=_slice(lo=n_train, hi=n_non_test),
            test=_slice(lo=n_non_test, hi=n),
        )

    def generate_gaussian_dataset(
        self,
        config_path: Path | None = None,
        *,
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

        # The parameters go into `_draw_gaussian` at the float64 they were parsed
        # in, and the sample narrows to `np.single` once on the way out because the draw
        # upcasts again and costs precision in the Cholesky whenever `np.single`
        cache_path: Path = self._cache_path(parsed, n_samples)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            note("gaussian cache hit", to="data")
            logger.info("Loading dataset from cache: %s", cache_path)
            with np.load(file=cache_path) as cached:
                arrays: Mapping[str, NDArray[Any]] = cast(
                    "Mapping[str, NDArray[Any]]", cached
                )
                data: ZXY = ZXY(
                    Events(
                        z=arrays["z"].astype(dtype=self.dtype),
                        x=arrays["x"].astype(dtype=self.dtype),
                    ),
                    y=arrays["y"],
                )
        else:
            note("gaussian generated", to="data")
            z_true, z_gen, x_data, x_sim = _draw_gaussian(
                self.seed,
                mu_true=parsed.mu_true,
                mu_gen=parsed.mu_gen,
                cov_true=parsed.cov_true,
                cov_gen=parsed.cov_gen,
                cov_detector=parsed.cov_detector,
                n_samples=n_samples,
            )

            data: ZXY = Populations(
                mc=Events(
                    z=np.asarray(a=z_gen, dtype=self.dtype),
                    x=np.asarray(a=x_sim, dtype=self.dtype),
                ),
                data=np.asarray(a=x_data, dtype=self.dtype),
                truth=np.asarray(a=z_true, dtype=self.dtype),
            ).interleave()

            # Uncompressed, for the reason spelled out in `ran.data.download`:
            # these are incompressible floats, so DEFLATE is a large read tax
            # for a few percent of disk. Existing compressed caches still load.
            np.savez(file=cache_path, z=data.z, x=data.x, y=data.y)
            logger.info("Generated and saved dataset to cache: %s", cache_path)

        return self.splits_from_data(data)

    def splits_from_data(self, data: ZXY) -> DatasetSplits:
        """Shuffle one labelled sample and cut it into train/val/test."""
        self.dataset: ZXY = self._build_dataset(data)
        self.splits: DatasetSplits = self._split_dataset(self.dataset)
        return self.splits
