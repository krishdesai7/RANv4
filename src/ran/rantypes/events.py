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


@dataclass(frozen=True, eq=False, slots=True)
class Events[T: np.floating = np.double]:
    z: NDArray[T]
    x: NDArray[T]

    def __post_init__(self) -> None:
        if self.z.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"z has {self.z.shape[0]} rows and x has {self.x.shape[0]}; "
                "particle and detector level arrays must be row-aligned"
            )

    @property
    def dtype(self) -> np.dtype[T]:
        return self.z.dtype

    def __len__(self) -> int:
        return self.z.shape[0]

    @classmethod
    def concatenate(cls, parts: Sequence[Self]) -> Self:
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of events")
        return cls(
            z=np.concatenate([part.z for part in parts], axis=0),
            x=np.concatenate([part.x for part in parts], axis=0),
        )

    def astype[U: np.floating](self, dtype: type[U]) -> Events[U]:
        return Events[U](z=self.z.astype(dtype), x=self.x.astype(dtype))


@dataclass(frozen=True, eq=False, slots=True)
class Populations[T: np.floating = np.double]:
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
        if truth is None:
            truth = np.full(
                (data.shape[0], *mc.z.shape[1:]), TRUTH_SENTINEL, dtype=mc.z.dtype
            )
        return cls(mc=mc, data=data, truth=truth)

    @property
    def has_truth(self) -> bool:
        return not np.all(a=self.truth == TRUTH_SENTINEL)

    def astype[U: np.floating](self, dtype: type[U]) -> Populations[U]:
        return Populations[U](
            mc=self.mc.astype(dtype),
            data=self.data.astype(dtype),
            truth=self.truth.astype(dtype),
        )

    def require_truth(self) -> NDArray[T]:
        if not self.has_truth:
            raise ValueError(
                "this sample has no particle-level truth to score against: it "
                "was built without one, so `truth` is the sentinel stand-in"
            )
        return self.truth

    def interleave(self) -> ZXY[T]:
        return ZXY[T](
            Events[T](
                z=np.concatenate([self.truth, self.mc.z], axis=0),
                x=np.concatenate([self.data, self.mc.x], axis=0),
            ),
            y=np.concatenate(
                [
                    np.ones(self.data.shape[0], dtype=np.ubyte),
                    np.zeros(shape=len(self.mc), dtype=np.ubyte),
                ]
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class ZXY[T: np.floating = np.double]:
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

    @property
    def dtype(self) -> np.dtype[T]:
        return self.events.dtype

    @classmethod
    def concatenate(cls, parts: Sequence[Self]) -> Self:
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of labelled events")
        return cls(
            events=Events[T].concatenate([part.events for part in parts]),
            y=np.concatenate([part.y for part in parts], axis=0),
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

    def select(self, which: Split = Split.ALL, /) -> ZXY[T]:
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
