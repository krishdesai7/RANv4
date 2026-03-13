from typing import NamedTuple
from pathlib import Path
import hashlib, json

import numpy as np
import numpy.typing as npt

import tensorflow as tf
from keras.utils import split_dataset

class DatasetSplits(NamedTuple):
    """
    Named tuple representing dataset splits.
    Fields:
        train (tf.data.Dataset)
        val (tf.data.Dataset)
        test (tf.data.Dataset)
    """
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset

class RAN_Dataset():
    """
    Dataset class for RAN.
    Arguments:
        batch_size (int)
        seed (int): Random seed.
        cache_dir (str | Path)
        val_fraction (float)
        test_fraction (float)
    Attributes:
        dataset (tf.data.Dataset)
        splits (DatasetSplits)

    Methods:
        generate_gaussian_dataset
    """
    def __init__(self,
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
        self.dataset: tf.data.Dataset | None = None
        self.splits: DatasetSplits | None = None
    
    def _cache_key(self,
        n_samples: int,
        smearing: float,
        test_fraction: float,
    ) -> str:
        return hashlib.sha256(
            json.dumps({
                "n_samples": n_samples,
                "smearing": smearing,
                "seed": self.seed,
                "test_fraction": test_fraction,
            }, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()[:16]

    def _cache_path(self, n_samples: int, smearing: float) -> Path:
        cache_key: str = self._cache_key(
            n_samples,
            smearing,
            self.test_fraction,
        )
        return self.cache_dir / f"gaussian_{cache_key}.npz"
    
    def _build_dataset(
        self,
        z: npt.NDArray[np.double],
        x: npt.NDArray[np.double],
        y: npt.NDArray[np.ubyte],
    ) -> tf.data.Dataset:
        features: dict[str, npt.NDArray[np.double]] = {
            "z": z, # Particle level
            "x": x, # Detector level
        }
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((features, y))
        dataset = dataset.shuffle(
            buffer_size=len(y),
            seed=self.seed,
            reshuffle_each_iteration=False,
            )
        return dataset

    def _split_dataset(self, dataset: tf.data.Dataset) -> DatasetSplits:
        non_test: tf.data.Dataset
        test: tf.data.Dataset
        non_test, test = split_dataset(
            dataset,
            right_size=self.test_fraction,
            shuffle=False,
        )
        val_of_non_test: float = self.val_fraction / (1.0 - self.test_fraction)
        train: tf.data.Dataset
        val: tf.data.Dataset
        train, val = split_dataset(
            non_test,
            right_size=val_of_non_test,
            shuffle=False,
        )
        train_buffer_size: tf.Tensor | int = tf.data.experimental.cardinality(train)
        if train_buffer_size == tf.data.UNKNOWN_CARDINALITY:
            train_buffer_size = self.batch_size * 10
        elif train_buffer_size == tf.data.INFINITE_CARDINALITY:
            raise ValueError("Train dataset has infinite cardinality")
        train = train.shuffle(
            buffer_size=train_buffer_size,
            seed=self.seed,
            reshuffle_each_iteration=True,
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val = val.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test = test.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return DatasetSplits(train, val, test)

    def generate_gaussian_dataset(self,
        n_samples: int = 10 ** 6,
        smearing: float = 1.0,
        ) -> DatasetSplits:
        """
        Generate a Gaussian dataset.
        Arguments:
            n_samples (int)
            smearing (float)
        Returns:
            DatasetSplits
        """
        cache_path: Path = self._cache_path(n_samples, smearing)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            with np.load(cache_path) as data:
                z: npt.NDArray[np.double] = data["z"]
                x: npt.NDArray[np.double] = data["x"]
                y: npt.NDArray[np.ubyte] = data["y"]
        else:
            rng: np.random.Generator = np.random.default_rng(self.seed)

            z_true: npt.NDArray[np.double] = rng.normal(size=(n_samples, 1))
            x_data: npt.NDArray[np.double] = rng.normal(z_true, smearing)
            y_nat: npt.NDArray[np.ubyte] = np.ones_like(z_true, dtype=np.ubyte)

            z_gen: npt.NDArray[np.double] = rng.normal(loc = 0.5,size=(n_samples, 1))
            x_sim: npt.NDArray[np.double] = rng.normal(z_gen, smearing)
            y_MC: npt.NDArray[np.ubyte] = np.zeros_like(z_gen, dtype=np.ubyte)

            z: npt.NDArray[np.double] = np.concatenate((z_true, z_gen), axis=0)
            x: npt.NDArray[np.double] = np.concatenate((x_data, x_sim), axis=0)
            y: npt.NDArray[np.ubyte] = np.concatenate((y_nat, y_MC), axis=0)

            np.savez_compressed(cache_path, z=z, x=x, y=y)
        
        self.dataset = self._build_dataset(z, x, y)
        self.splits = self._split_dataset(self.dataset)
        return self.splits