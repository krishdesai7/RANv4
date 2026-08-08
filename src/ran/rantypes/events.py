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
class Events:
    """Particle-level `z` and detector-level `x` for one set of events.

    The arrays are row-aligned: row `i` of each is the same event seen at the
    two levels.
    """

    z: NDArray[np.double]
    x: NDArray[np.double]

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


@dataclass(frozen=True, eq=False, slots=True)
class ZXY:
    """Events labelled by provenance: y = 1 for nature, y = 0 for MC.

    The transport form -- what gets shuffled, split, batched and trained on.
    `partition` converts to the physics form, `Populations.interleave` back.
    """

    events: Events
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
    def z(self) -> NDArray[np.double]:
        return self.events.z

    @property
    def x(self) -> NDArray[np.double]:
        return self.events.x

    @classmethod
    def concatenate(cls, parts: Sequence[Self]) -> Self:
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of labelled events")
        return cls(
            Events.concatenate([part.events for part in parts]),
            np.concatenate([part.y for part in parts], axis=0),
        )

    def partition(self) -> Populations:
        """Separate the labelled events into the four physics populations."""
        mc: NDArray[np.bool] = self.y == 0
        nature: NDArray[np.bool] = ~mc
        return Populations(
            mc=Events(self.events.z[mc], self.events.x[mc]),
            data=self.events.x[nature],
            truth=self.events.z[nature],
        )


@dataclass(frozen=True, eq=False, slots=True)
class Populations:
    """The physics view of a labelled sample.

    `mc` is the simulation, its generated particle level (`mc.z`) paired per
    event with the simulated detector response (`mc.x`); that pairing is what
    builds a response matrix. `data` is the measurement.

    `truth` is the particle-level answer key. It exists only because every
    dataset here is a closure test -- a real measurement has no such array --
    and no network may ever see it. Keeping it out of `mc` means a function
    handed the simulation cannot reach it.
    """

    mc: Events
    data: NDArray[np.double]
    truth: NDArray[np.double]

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

    def interleave(self) -> ZXY:
        """Stack into the labelled transport form, nature rows first.

        The resulting row order is an artifact of stacking rather than
        anything meaningful, so callers shuffle before splitting.
        """
        return ZXY(
            Events(
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


class DatasetSplits(NamedTuple):
    train: ArrayDataset
    val: ArrayDataset
    test: ArrayDataset

    def select(self, which: Split = Split.ALL) -> ZXY:
        """Concatenate the requested splits into one labelled sample.

        The split a row came from is not recorded on the result: it is a
        property of the query, not of the events, and nothing downstream of
        this call can act on it.
        """
        chosen: list[ZXY] = [
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
        return ZXY.concatenate(chosen)
