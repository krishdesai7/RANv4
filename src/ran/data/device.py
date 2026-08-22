"""Device-resident training data, the third form alongside ``Populations``/``ZXY``.

``Populations`` is the physics form and ``ZXY`` the transport form; both are host
NumPy, because they feed SciPy, Matplotlib, npz I/O and the IBU baseline. This
module is the training form: ``DeviceSplits.from_splits`` is the single
host-to-device transfer of a run, and after it no batch crosses the boundary again.

All three forms live under ``ran.data`` because they are all the dataset, just at
different points in its trip to the accelerator.

Both splits are laid out for a single fused XLA program: the train split stays
flat and is gathered by index inside a ``lax.scan``, so XLA fuses the gather into
the first ``Dense``; the eval splits are pre-batched with a mask, so evaluation
scans with no gather at all and still sees every event exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from typing import Self

    import numpy as np
    from jaxtyping import Array, Float, Int, PRNGKeyArray
    from numpy.typing import NDArray

    from ..rantypes import ZXY, DatasetSplits

# Evaluation is forward-only and its batching is not part of the training
# contract, so it uses a wider batch than training to keep the scan short.
DEFAULT_EVAL_BATCH_SIZE: int = 8192


@partial(jax.tree_util.register_dataclass, data_fields=["z", "x", "y"], meta_fields=[])
@dataclass(frozen=True, eq=False, slots=True)
class TrainSplit:
    """A flat, device-resident split, gathered by index inside the scan."""

    z: Float[Array, "n d"]
    x: Float[Array, "n d"]
    # Promoted from `np.ubyte` to the compute dtype once, here, rather than in
    # every `1 - y` inside the trace.
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
    """A pre-batched device-resident split; padding rows carry ``mask == 0``."""

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
        dtype = data.z.dtype
        dim: int = data.z.shape[1]

        def _pad2d(arr: NDArray[np.floating]) -> Float[Array, "nb bs d"]:
            # Edge padding repeats a real row; `mask` is what keeps it out of
            # every sum, so the value only has to be finite.
            wide = jnp.pad(jnp.asarray(arr), ((0, pad), (0, 0)), mode="edge")
            return wide.reshape(n_batches, size, dim)

        y = jnp.pad(jnp.asarray(data.y, dtype=dtype), (0, pad), mode="edge")
        mask = jnp.concatenate([jnp.ones(n, dtype=dtype), jnp.zeros(pad, dtype=dtype)])
        return cls(
            z=_pad2d(data.z),
            x=_pad2d(data.x),
            y=y.reshape(n_batches, size),
            mask=mask.reshape(n_batches, size),
        )


class DeviceSplits(NamedTuple):
    """The whole dataset on device, plus the seed that drives batch order."""

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
        """Move a host ``DatasetSplits`` to device. The one H2D transfer.

        ``batch_size`` is accepted for symmetry but unused: the train split stays
        flat, and its batching is decided per epoch by :func:`train_indices`.
        """
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
    """Split one pass over ``n`` events into ``(groups, disc steps per group)``.

    ``n_disc_steps`` is clamped to the number of whole batches available. A split
    too small to fill one group still trains --- it becomes a single group with
    every batch in it, and one generator update --- which is what the host loop
    did when ``step % n_disc_steps == 0`` fired only at step 0.
    """
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
    """One epoch's batch order, grouped so the update rhythm is structural.

    Each group is ``n_disc_steps`` discriminator batches; the generator updates
    once per group, on the group's first batch. That is what the host loop used
    to express as ``step % n_disc_steps == 0``.

    The tail that does not fill a whole group is dropped. Because the permutation
    is redrawn every epoch, it is a different random tail each pass.
    """
    groups, per_group = grouping(n, batch_size, n_disc_steps)
    keep: int = groups * per_group * batch_size
    order: Int[Array, " n"] = jax.random.permutation(key, n)
    return order[:keep].reshape(groups, per_group, batch_size)


def gather(
    split: TrainSplit, idx: Int[Array, " b"], /
) -> tuple[Float[Array, "b d"], Float[Array, "b d"], Float[Array, " b"]]:
    """Pull one batch out of the flat split. Fused into the first matmul."""
    return (
        jnp.take(split.z, idx, axis=0),
        jnp.take(split.x, idx, axis=0),
        jnp.take(split.y, idx, axis=0),
    )
