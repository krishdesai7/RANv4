from collections.abc import Iterator
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from pathlib import Path
import hashlib, json

class RAN_Dataset():
    """
    Dataset class for RAN.
    Arguments:
        batch_size (int)
        shuffle (bool)
        seed (int): Random seed.
        cache_dir (str | Path)
    Attributes:
        dataset (tf.data.Dataset)
    Methods:
        generate_gaussian_dataset
    """
    def __init__(self,
        batch_size: int = 128,
        shuffle: bool = True,
        seed: int = 42,
        cache_dir: str | Path = ".cache",
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self.dataset: tf.data.Dataset | None = None
    
    def _cache_key(self,
        n_samples: int,
        smearing: float,
    ) -> str:
        return hashlib.sha256(
            json.dumps({
                "n_samples": n_samples,
                "smearing": smearing,
                "seed": self.seed,
            }, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()[:16]

    def _cache_path(self, n_samples: int, smearing: float) -> Path:
        filename: Path = Path(f"gaussian_{self._cache_key(n_samples, smearing)}.npz")
        return self.cache_dir / filename
    
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
        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(y),
                seed = self.seed,
                )
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def generate_gaussian_dataset(self,
        n_samples: int = 10 ** 6,
        smearing: float = 1.0,
        ) -> tf.data.Dataset:
        """
        Generate a Gaussian dataset.
        Arguments:
            n_samples (int)
            smearing (float)
        Returns:
            tf.data.Dataset
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
        return self.dataset


    def __iter__(self) -> Iterator[tuple[dict[str, tf.Tensor], tf.Tensor]]:
        if self.dataset is None:
            raise ValueError("Dataset not generated")
        return iter(self.dataset)