from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

import jax
import jax.numpy as jnp
import numpy as np

from ..rantypes import ZXY, DatasetSplits, Events, GaussianConfig, Populations
from .config import parse_gaussian_config

if TYPE_CHECKING:
    from logging import Logger
    from typing import Final, LiteralString, SupportsFloat

    from jaxtyping import Array, Float
    from numpy._typing import _DTypeLike
    from numpy.typing import DTypeLike, NDArray

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
    """Draw the four Gaussian populations: (z_true, z_gen, x_data, x_sim).

    Pinned to CPU. Generation runs once and is then npz-cached, so there is
    nothing to gain from the accelerator --- and this function is reachable from
    ``ran.baselines._shared`` on a cache miss.

    No ``check_valid`` equivalent is needed: ``parse_gaussian_config`` has already
    asserted positive-definiteness with a Cholesky factorization.
    """
    with jax.default_device(jax.devices(backend="cpu")[0]):
        k_true, k_gen, k_data, k_sim = jax.random.split(jax.random.key(seed), 4)

        z_true = jax.random.multivariate_normal(
            k_true, mu_true, cov_true, (n_samples,), method="svd"
        )
        z_gen = jax.random.multivariate_normal(
            k_gen, mu_gen, cov_gen, (n_samples,), method="svd"
        )

        chol_det = jnp.linalg.cholesky(jnp.asarray(cov_detector))
        smear = chol_det.T

        x_data = z_true + jax.random.normal(k_data, z_true.shape) @ smear
        x_sim = z_gen + jax.random.normal(k_sim, z_gen.shape) @ smear
    return z_true, z_gen, x_data, x_sim


class ArrayDataset[T: np.floating = np.double]:
    """One host-resident split of (z, x, y), plus how it should be batched.

    This is a container, not an iterator. Batch order is drawn on device, per
    epoch, by ``ran.device.train_indices`` --- so ``batch_size`` and ``seed`` are
    carried here as the split's own parameters and read by
    ``DeviceSplits.from_splits``, but nothing iterates this object.
    """

    def __init__(
        self,
        data: ZXY[T],
        batch_size: int = 128,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.data: ZXY[T] = data
        self.batch_size: int = batch_size
        self.seed: int = seed

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def dtype(self) -> np.dtype[T]:
        return self.data.dtype

    def __len__(self) -> int:
        """Number of batches per pass, counting a short trailing one."""
        return (self.size + self.batch_size - 1) // self.batch_size

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
        key_data: dict[str, Nested[float] | str] = {
            "mu_gen": self._round_nested(parsed.mu_gen.tolist()),
            "mu_true": self._round_nested(parsed.mu_true.tolist()),
            "cov_gen": self._round_nested(parsed.cov_gen.tolist()),
            "cov_true": self._round_nested(parsed.cov_true.tolist()),
            "cov_detector": self._round_nested(parsed.cov_detector.tolist()),
            "n_samples": n_samples,
            "seed": self.seed,
            "rng": _RNG_VERSION,
            # Without this a float32 and a float64 run share one file, and
            # whichever ran first decides the precision on disk.
            "dtype": str(self.dtype),
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

        def _slice(lo: int, hi: int) -> ArrayDataset[T]:
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

        # The parameters go into `_draw_gaussian` at the float64 they were parsed
        # in, and the sample narrows to `T` once on the way out. Pre-narrowing
        # them bought nothing -- the draw upcasts again -- and cost precision in
        # the detector Cholesky whenever `T` was float32.
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
            z_true, z_gen, x_data, x_sim = _draw_gaussian(
                self.seed,
                mu_true=parsed.mu_true,
                mu_gen=parsed.mu_gen,
                cov_true=parsed.cov_true,
                cov_gen=parsed.cov_gen,
                cov_detector=parsed.cov_detector,
                n_samples=n_samples,
            )

            data: ZXY[T] = Populations(
                mc=Events(
                    z=np.asarray(z_gen, dtype=self.dtype),
                    x=np.asarray(x_sim, dtype=self.dtype),
                ),
                data=np.asarray(x_data, dtype=self.dtype),
                truth=np.asarray(z_true, dtype=self.dtype),
            ).interleave()

            np.savez_compressed(file=cache_path, z=data.z, x=data.x, y=data.y)
            logger.info("Generated and saved dataset to cache: %s", cache_path)

        return self.splits_from_data(data)

    def splits_from_data(self, data: ZXY[T]) -> DatasetSplits[T]:
        """Shuffle one labelled sample and cut it into train/val/test."""
        self.dataset = self._build_dataset(data)
        self.splits = self._split_dataset(self.dataset)
        return self.splits
