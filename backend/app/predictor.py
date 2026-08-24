from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .config import Settings
from .model import DenseNet121Adapter

if TYPE_CHECKING:
    from .gradcam import GradCamGenerator

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


@dataclass
class PredictionResult:
    prediction: str
    predicted_class: str
    pneumonia_probability: float
    normal_probability: float
    confidence: float
    gradcam_available: bool
    gradcam_image_base64: str | None


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.adapter: DenseNet121Adapter | None = None
        self.gradcam_generator: GradCamGenerator | None = None
        self.model_error: str | None = None
        self._inference_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self.adapter is not None

    def load_model(self) -> None:
        if self.adapter is not None:
            return

        try:
            self.adapter = DenseNet121Adapter(
                model_path=self.settings.model_path,
                input_size=(self.settings.input_width, self.settings.input_height),
            )
            self.model_error = None
            logger.info('Loaded model successfully from %s', self.settings.model_path)

            if self.settings.enable_gradcam:
                try:
                    from .gradcam import GradCamGenerator
                    self.gradcam_generator = GradCamGenerator(self.adapter.keras_model)
                    logger.info('Initialized Grad-CAM generator successfully.')
                except Exception as exc:  # noqa: BLE001
                    self.gradcam_generator = None
                    logger.warning('Grad-CAM initialization deferred or failed: %s', exc)

        except Exception as exc:  # noqa: BLE001
            self.adapter = None
            self.gradcam_generator = None
            self.model_error = str(exc)
            logger.exception('Unable to load model from %s', self.settings.model_path)

    def _ensure_loaded(self) -> DenseNet121Adapter:
        if not self.adapter:
            raise ModelNotLoadedError(
                'Model not loaded. Add a trained DenseNet121 model and restart the backend.'
            )
        return self.adapter

    def _get_gradcam_generator(self) -> GradCamGenerator | None:
        if not self.settings.enable_gradcam:
            return None
        if self.gradcam_generator is not None:
            return self.gradcam_generator
        if self.adapter is not None:
            try:
                from .gradcam import GradCamGenerator
                self.gradcam_generator = GradCamGenerator(self.adapter.keras_model)
                return self.gradcam_generator
            except Exception:
                logger.exception('Failed on-demand Grad-CAM initialization.')
                return None
        return None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((self.settings.input_width, self.settings.input_height))
        image_np = np.asarray(resized, dtype=np.float32)
        # The model was trained with ImageDataGenerator(rescale=1./255).
        # Preserve that exact inference contract.
        image_np *= 1.0 / 255.0
        return np.expand_dims(image_np, axis=0)

    @staticmethod
    def _extract_probabilities(raw_output: np.ndarray) -> tuple[float, float]:
        values = np.asarray(raw_output, dtype=np.float32).reshape(-1)

        # flow_from_directory sorts class folders alphabetically:
        # normal = 0, pneumonia = 1. The sigmoid scalar output
        # represents the pneumonia probability.
        if values.size == 1:
            pneumonia_probability = float(values[0])
            pneumonia_probability = min(max(pneumonia_probability, 0.0), 1.0)
            normal_probability = 1.0 - pneumonia_probability
            return pneumonia_probability, normal_probability

        if values.size == 2:
            values = np.clip(values, 0.0, 1.0)
            denom = float(values.sum())
            if denom <= 0:
                normal_probability = 0.5
                pneumonia_probability = 0.5
            else:
                normal_probability = float(values[0] / denom)
                pneumonia_probability = float(values[1] / denom)
            return pneumonia_probability, normal_probability

        raise ValueError('Unexpected model output shape. Expected scalar sigmoid or two-class vector.')

    def predict(self, image: Image.Image) -> PredictionResult:
        with self._inference_lock:
            adapter = self._ensure_loaded()
            preprocessed = self._preprocess(image)
            raw_output = None

            try:
                raw_output = adapter.predict(preprocessed)
                pneumonia_probability, normal_probability = (
                    self._extract_probabilities(raw_output)
                )

                is_positive = pneumonia_probability >= self.settings.positive_threshold
                prediction_label = 'Pneumonia Detected' if is_positive else 'Normal'
                predicted_class = 'pneumonia' if is_positive else 'normal'
                confidence = pneumonia_probability if is_positive else normal_probability
                gradcam_class_index = 1 if is_positive else 0

                gradcam_available = False
                gradcam_base64 = None

                gradcam = self._get_gradcam_generator()
                if gradcam is not None:
                    try:
                        from .utils import pil_to_base64_png

                        resized_rgb = np.asarray(
                            image.resize((self.settings.input_width, self.settings.input_height)),
                            dtype=np.uint8,
                        )
                        gradcam_image = gradcam.generate(
                            preprocessed,
                            resized_rgb,
                            class_index=gradcam_class_index,
                        )
                        gradcam_base64 = pil_to_base64_png(gradcam_image)
                        gradcam_available = True
                        del gradcam_image, resized_rgb
                    except Exception:
                        logger.exception('Grad-CAM generation failed; returning prediction only.')

                return PredictionResult(
                    prediction=prediction_label,
                    predicted_class=predicted_class,
                    pneumonia_probability=round(pneumonia_probability, 4),
                    normal_probability=round(normal_probability, 4),
                    confidence=round(confidence, 4),
                    gradcam_available=gradcam_available,
                    gradcam_image_base64=gradcam_base64,
                )
            finally:
                del preprocessed
                if raw_output is not None:
                    del raw_output
