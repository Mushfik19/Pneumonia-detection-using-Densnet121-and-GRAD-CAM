from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class ModelAdapter(Protocol):
    input_size: tuple[int, int]

    def predict(self, image_batch: np.ndarray) -> np.ndarray:
        ...

    @property
    def keras_model(self) -> tf.keras.Model:
        ...


@dataclass
class DenseNet121Adapter:
    model_path: str
    input_size: tuple[int, int] = (224, 224)

    def __post_init__(self) -> None:
        self._model = self._load_keras_model(self.model_path)

    @staticmethod
    def _load_keras_model(model_path: str) -> tf.keras.Model:
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(
                f'Model file not found at {model_file}. Add a trained DenseNet121 model to continue.'
            )

        if model_file.suffix.lower() not in {'.keras', '.h5'}:
            raise ValueError('Model must be a .keras or .h5 file.')

        return load_model(model_file)

    @property
    def keras_model(self) -> tf.keras.Model:
        return self._model

    def predict(self, image_batch: np.ndarray) -> np.ndarray:
        return self._model.predict(image_batch, verbose=0)
