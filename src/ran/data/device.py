from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from typing import Final, Self

    import numpy as np
    from jaxtyping import Array, Float, Int, PRNGKeyArray
    from numpy.typing import NDArray

    from ..rantypes import ZXY, DatasetSplits

DEFAULT_EVAL_BATCH_SIZE: Final[int] = 8192


@partial(jax.tree_util.register_dataclass, data_fields=["z", "x", "y"], meta_fields=[])
@dataclass(frozen=True, eq=False, slots=True)
class TrainSplit:
    z: Float[Array, "n d"]
    x: Float[Array, "n d"]
    y: Float[Array, " n"]

    @property
    def n(self) -> int:
        return self.z.shape[0]

    @classmethod
    def from_zxy(cls, data: ZXY, /) -> Self:
        return cls(
            z=jnp.asarray(data.z),
            x=jnp.asarray(data.x),
            y=jnp.asarray(data.y, dtype=data.z.dtype),
        )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["z", "x", "y", "mask"],
    meta_fields=[],
)
@dataclass(frozen=True, eq=False, slots=True)
class EvalSplit:
    z: Float[Array, "nb bs d"]
    x: Float[Array, "nb bs d"]
    y: Float[Array, "nb bs"]
    mask: Float[Array, "nb bs"]

    @property
    def n_batches(self) -> int:
        return self.z.shape[0]

    @classmethod
    def from_zxy(cls, data: ZXY, /, *, batch_size: int) -> Self:
        n: int = len(data)
        size: int = min(batch_size, n)
        n_batches: int = -(-n // size)
        pad: int = n_batches * size - n
        dtype: dtype[np.single] = data.z.dtype
        dim: int = data.z.shape[1]

        def _pad2d(arr: NDArray[np.floating]) -> Float[Array, "nb bs d"]:
            # Edge padding repeats a real row; `mask` is what keeps it out of
            # every sum, so the value only has to be finite.
            wide: Float[Array, "nb bs d"] = jnp.pad(
                array=jnp.asarray(a=arr), pad_width=((0, pad), (0, 0)), mode="edge"
            )
            return wide.reshape(n_batches, size, dim)

        y: Float[Array, "nb bs"] = jnp.pad(
            array=jnp.asarray(a=data.y, dtype=dtype), pad_width=(0, pad), mode="edge"
        )
        mask: Float[Array, "nb bs"] = jnp.concatenate(
            arrays=[jnp.ones(shape=n, dtype=dtype), jnp.zeros(shape=pad, dtype=dtype)]
        )
        return cls(
            z=_pad2d(arr=data.z),
            x=_pad2d(arr=data.x),
            y=y.reshape(n_batches, size),
            mask=mask.reshape(n_batches, size),
        )


class DeviceSplits(NamedTuple):
    train: TrainSplit
    val: EvalSplit
    test: EvalSplit
    data_seed: int

    @classmethod
    def from_splits(
        cls,
        splits: DatasetSplits,
        /,
        *,
        batch_size: int | None = None,
        eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    ) -> Self:
        del batch_size
        return cls(
            train=TrainSplit.from_zxy(splits.train.as_arrays()),
            val=EvalSplit.from_zxy(splits.val.as_arrays(), batch_size=eval_batch_size),
            test=EvalSplit.from_zxy(
                splits.test.as_arrays(), batch_size=eval_batch_size
            ),
            data_seed=splits.train.seed,
        )


def grouping(n: int, batch_size: int, n_disc_steps: int) -> tuple[int, int]:
    n_batches: int = n // batch_size
    if n_batches < 1:
        raise ValueError(
            f"{n} events do not fill a single batch of {batch_size}; "
            "lower batch_size or use more events"
        )
    per_group: int = min(n_disc_steps, n_batches)
    return n_batches // per_group, per_group


def train_indices(
    key: PRNGKeyArray, n: int, batch_size: int, n_disc_steps: int
) -> Int[Array, "groups disc batch"]:
    groups, per_group = grouping(n, batch_size, n_disc_steps)
    keep: int = groups * per_group * batch_size
    order: Int[Array, " n"] = jax.random.permutation(key, x=n)
    return order[:keep].reshape(groups, per_group, batch_size)


def gather(
    split: TrainSplit, idx: Int[Array, " b"], /
) -> tuple[Float[Array, "b d"], Float[Array, "b d"], Float[Array, " b"]]:
    return (
        jnp.take(a=split.z, indices=idx, axis=0),
        jnp.take(a=split.x, indices=idx, axis=0),
        jnp.take(a=split.y, indices=idx, axis=0),
    )
