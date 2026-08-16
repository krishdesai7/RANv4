"""The event data model, in its two representations.

`Populations` is the physics form and `ZXY` the transport form; see the "Data
Representations" section of CLAUDE.md for why both exist and which direction
is lossless.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .constants import TRUTH_SENTINEL

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from numpy.typing import NDArray

    from ..data import ArrayDataset


class Split(Flag):
    """Which of the train/val/test splits to draw events from."""

    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    ALL = TRAIN | VAL | TEST


# The types below take `eq=False` because they hold arrays: a generated
# `__eq__` compares fields with `==` and a generated `__hash__` hashes them,
# and both raise. Identity comparison is what these are wanted for anyway.
@dataclass(frozen=True, eq=False, slots=True)
class Events[T: np.floating = np.double]:
    """Particle-level `z` and detector-level `x` for one set of events.

    The arrays are row-aligned: row `i` of each is the same event seen at the
    two levels.
    """

    z: NDArray[T]
    x: NDArray[T]

    def __post_init__(self) -> None:
        if self.z.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"z has {self.z.shape[0]} rows and x has {self.x.shape[0]}; "
                "particle and detector level arrays must be row-aligned"
            )

    def __len__(self) -> int:
        return self.z.shape[0]

    @classmethod
    def concatenate(cls, parts: Sequence[Self]) -> Self:
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of events")
        return cls(
            np.concatenate([part.z for part in parts], axis=0),
            np.concatenate([part.x for part in parts], axis=0),
        )

    def astype[U: np.floating](self, dtype: type[U]) -> Events[U]:
        return Events[U](self.z.astype(dtype), self.x.astype(dtype))


@dataclass(frozen=True, eq=False, slots=True)
class Populations[T: np.floating = np.double]:
    """The physics view of a labelled sample.

    `mc` is the simulation, its generated particle level (`mc.z`) paired per
    event with the simulated detector response (`mc.x`); that pairing is what
    builds a response matrix. `data` is the measurement.

    `truth` is the particle-level answer key. It exists only because every
    dataset here is a closure test -- a real measurement has no such array --
    and no network may ever see it. Keeping it out of `mc` means a function
    handed the simulation cannot reach it. Construct through `create` to leave
    it out; the field itself is always present.
    """

    mc: Events[T]
    data: NDArray[T]
    truth: NDArray[T]

    def __post_init__(self) -> None:
        if self.data.shape[0] != self.truth.shape[0]:
            raise ValueError(
                f"data has {self.data.shape[0]} rows and truth has "
                f"{self.truth.shape[0]}; both describe the same nature events"
            )
        if len(self.mc) == 0 or self.data.shape[0] == 0:
            raise ValueError(
                f"populations must be nonempty, got {self.data.shape[0]} nature "
                f"and {len(self.mc)} MC events"
            )

    @classmethod
    def create(
        cls,
        mc: Events[T],
        data: NDArray[T],
        truth: NDArray[T] | None = None,
    ) -> Self:
        """Build a sample, filling `truth` with `TRUTH_SENTINEL` if there is none.

        A real measurement has no answer key. Filling the field rather than
        dropping it keeps one type for both cases, and keeps the sample
        trainable: the nature rows of `z` are `truth`, so they reach the
        generator, and only a finite value there lets `normalize_weights`
        annihilate them as intended. See `TRUTH_SENTINEL` for why not NaN.

        `truth` is particle level, so it takes its columns from `mc.z` and its
        rows from `data`.
        """
        if truth is None:
            truth = np.full(
                (data.shape[0], *mc.z.shape[1:]), TRUTH_SENTINEL, dtype=mc.z.dtype
            )
        return cls(mc=mc, data=data, truth=truth)

    @property
    def has_truth(self) -> bool:
        """Whether `truth` holds answers rather than the `create` stand-in.

        Any metric computed against a sentinel `truth` is meaningless but
        finite, so unfolding code that scores against the particle level has
        to ask rather than wait to be told.
        """
        return not bool(np.any(self.truth == TRUTH_SENTINEL))

    def astype[U: np.floating](self, dtype: type[U]) -> Populations[U]:
        """The same sample at another precision.

        RAN is float64 end to end, but the baselines are not: OmniFold trains
        under TensorFlow and IBU has to match the single-precision arithmetic
        the published results were produced with. Each casts at its own
        boundary rather than making the shared pipeline pick a side.

        `TRUTH_SENTINEL` is exact in every IEEE binary format, so `has_truth`
        answers the same question on either side of this call.
        """
        return Populations[U](
            self.mc.astype(dtype),
            self.data.astype(dtype),
            self.truth.astype(dtype),
        )

    def require_truth(self) -> NDArray[T]:
        """`truth`, or a refusal if there is none.

        Scoring against the sentinel yields a finite, meaningless number
        instead of an obvious failure, so the particle-level comparisons ask
        for the answer key through here rather than reading the field.
        """
        if not self.has_truth:
            raise ValueError(
                "this sample has no particle-level truth to score against: it "
                "was built without one, so `truth` is the sentinel stand-in"
            )
        return self.truth

    def interleave(self) -> ZXY[T]:
        """Stack into the labelled transport form, nature rows first.

        The resulting row order is an artifact of stacking rather than
        anything meaningful, so callers shuffle before splitting.
        """
        return ZXY[T](
            Events[T](
                np.concatenate([self.truth, self.mc.z], axis=0),
                np.concatenate([self.data, self.mc.x], axis=0),
            ),
            np.concatenate(
                [
                    np.ones(self.data.shape[0], dtype=np.ubyte),
                    np.zeros(len(self.mc), dtype=np.ubyte),
                ]
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class ZXY[T: np.floating = np.double]:
    """Events labelled by provenance: y = 1 for nature, y = 0 for MC.

    The transport form -- what gets shuffled, split, batched and trained on.
    `partition` converts to the physics form, `Populations.interleave` back.
    """

    events: Events[T]
    y: NDArray[np.ubyte]

    def __post_init__(self) -> None:
        if self.y.ndim != 1 or self.y.shape[0] != len(self.events):
            raise ValueError(
                f"y has shape {self.y.shape}; expected one label per event in a "
                f"one-dimensional array of length {len(self.events)}"
            )
        if np.any((self.y != 0) & (self.y != 1)):
            raise ValueError("labels must be zero (MC) or one (nature)")

    def __len__(self) -> int:
        return self.y.shape[0]

    @property
    def z(self) -> NDArray[T]:
        return self.events.z

    @property
    def x(self) -> NDArray[T]:
        return self.events.x

    @classmethod
    def concatenate(cls, parts: Sequence[Self]) -> Self:
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of labelled events")
        return cls(
            Events[T].concatenate([part.events for part in parts]),
            np.concatenate([part.y for part in parts], axis=0),
        )

    def partition(self) -> Populations[T]:
        """Separate the labelled events into the four physics populations."""
        mc: NDArray[np.bool] = self.y == 0
        nature: NDArray[np.bool] = ~mc
        return Populations(
            mc=Events[T](self.events.z[mc], self.events.x[mc]),
            data=self.events.x[nature],
            truth=self.events.z[nature],
        )


class DatasetSplits[T: np.floating = np.double](NamedTuple):
    train: ArrayDataset[T]
    val: ArrayDataset[T]
    test: ArrayDataset[T]

    def select(self, which: Split = Split.ALL) -> ZXY[T]:
        """Concatenate the requested splits into one labelled sample.

        The split a row came from is not recorded on the result: it is a
        property of the query, not of the events, and nothing downstream of
        this call can act on it.
        """
        chosen: list[ZXY[T]] = [
            split.as_arrays()
            for flag, split in (
                (Split.TRAIN, self.train),
                (Split.VAL, self.val),
                (Split.TEST, self.test),
            )
            if flag in which
        ]
        if not chosen:
            raise ValueError("select needs at least one split")
        return ZXY[T].concatenate(parts=chosen)
