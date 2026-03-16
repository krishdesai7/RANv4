from typing import NamedTuple
from pathlib import Path
import hashlib, json

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky

import tensorflow as tf
from keras.utils import split_dataset

from ran.data.config import parse_gaussian_config, sigma_to_covariance

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
    
    def _cache_key(self, parsed: dict, n_samples: int) -> str:
        """Hash the promoted covariance matrices for a canonical cache key."""
        key_data = {
            "mu_mc": parsed["mu_mc"].tolist(),
            "mu_true": parsed["mu_true"].tolist(),
            "cov_mc": parsed["cov_mc"].tolist(),
            "cov_true": parsed["cov_true"].tolist(),
            "cov_detector": parsed["cov_detector"].tolist(),
            "n_samples": n_samples,
            "seed": self.seed,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _cache_path(self, parsed: dict, n_samples: int) -> Path:
        cache_key = self._cache_key(parsed, n_samples)
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
        config_path: str | Path | None = None,
        params: dict | None = None,
        n_samples: int = 10 ** 6,
    ) -> DatasetSplits:
        """
        Generate a multivariate Gaussian dataset.
        Arguments:
            config_path: Path to a YAML config file.
            params: Dict with keys mu_mc, mu_true, sigma_mc, sigma_true, sigma_detector.
            n_samples: Number of samples per class (data and MC).
        Returns:
            DatasetSplits
        Exactly one of config_path or params must be provided.
        """
        if (config_path is None) == (params is None):
            raise ValueError(
                "Exactly one of config_path or params must be provided"
            )

        if config_path is not None:
            parsed = parse_gaussian_config(config_path)
        else:
            mu_mc = np.asarray(params["mu_mc"], dtype=np.double).ravel()
            mu_true = np.asarray(params["mu_true"], dtype=np.double).ravel()
            dim = mu_mc.shape[0]
            if mu_true.shape[0] != dim:
                raise ValueError(
                    f"mu_true has dim {mu_true.shape[0]}, expected {dim}"
                )
            parsed = {
                "dim": dim,
                "mu_mc": mu_mc,
                "mu_true": mu_true,
                "cov_mc": sigma_to_covariance(params["sigma_mc"], dim),
                "cov_true": sigma_to_covariance(params["sigma_true"], dim),
                "cov_detector": sigma_to_covariance(params["sigma_detector"], dim),
            }

        dim: int = parsed["dim"]
        mu_mc: npt.NDArray[np.double] = parsed["mu_mc"]
        mu_true: npt.NDArray[np.double] = parsed["mu_true"]
        cov_mc: npt.NDArray[np.double] = parsed["cov_mc"]
        cov_true: npt.NDArray[np.double] = parsed["cov_true"]
        cov_detector: npt.NDArray[np.double] = parsed["cov_detector"]

        cache_path: Path = self._cache_path(parsed, n_samples)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"Loading dataset from cache: {cache_path}")
            with np.load(cache_path) as data:
                z = data["z"]
                x = data["x"]
                y = data["y"]
        else:
            rng = np.random.default_rng(self.seed)

            z_true = rng.multivariate_normal(
                mu_true, cov_true, size=n_samples,
                check_valid='raise', method='svd',
            )
            z_gen = rng.multivariate_normal(
                mu_mc, cov_mc, size=n_samples,
                check_valid='raise', method='svd',
            )

            L_det = cholesky(cov_detector, lower=True)

            s_data = rng.standard_normal(size=z_true.shape)
            x_data = z_true + s_data @ L_det.T

            s_sim = rng.standard_normal(size=z_gen.shape)
            x_sim = z_gen + s_sim @ L_det.T

            y_nat = np.ones(n_samples, dtype=np.ubyte)
            y_MC = np.zeros(n_samples, dtype=np.ubyte)

            z = np.concatenate((z_true, z_gen), axis=0)
            x = np.concatenate((x_data, x_sim), axis=0)
            y = np.concatenate((y_nat, y_MC), axis=0)

            np.savez_compressed(cache_path, z=z, x=x, y=y)
            print(f"Generated and saved dataset to cache: {cache_path}")

        self.dataset = self._build_dataset(z, x, y)
        self.splits = self._split_dataset(self.dataset)
        return self.splits