from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)


class GradCamGenerator:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    for DenseNet121 and compatible deep learning architectures.
    
    Dynamically identifies the appropriate late convolutional feature layer,
    tracks gradients with respect to the target class probability using
    GradientTape, and blends the resulting activation heatmap onto the original image.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.target_layer_name: str | None = None
        self._grad_model: tf.keras.Model | None = None
        self._is_nested: bool = False
        self._top_layers: list[tf.keras.layers.Layer] = []
        self._setup_grad_model()

    def _find_target_layer(self) -> Tuple[str, bool]:
        """
        Dynamically inspects the model architecture to identify the most suitable
        late convolutional feature / activation layer.
        """
        # Case 1: Sequential or wrapper model containing a nested backbone (e.g. densenet121)
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
                nested_model = layer
                logger.info(
                    'Inspecting nested feature backbone: %s (%s)',
                    nested_model.name,
                    type(nested_model).__name__,
                )

                # Prioritize standard DenseNet121 final convolutional & activation feature layers
                preferred_candidates = [
                    'relu',
                    'conv5_block16_concat',
                    'conv5_block16_2_conv',
                    'bn',
                ]
                for candidate in preferred_candidates:
                    try:
                        cand_layer = nested_model.get_layer(candidate)
                        out_shape = getattr(cand_layer, 'output', None)
                        if out_shape is not None and len(out_shape.shape) == 4:
                            logger.info(
                                "Selected target layer '%s' inside nested backbone '%s' (shape: %s)",
                                candidate,
                                nested_model.name,
                                out_shape.shape,
                            )
                            return candidate, True
                    except (ValueError, AttributeError):
                        continue

                # Fallback: scan backwards inside nested model for any 4D feature layer
                for sublayer in reversed(nested_model.layers):
                    try:
                        if hasattr(sublayer, 'output') and len(sublayer.output.shape) == 4:
                            logger.info(
                                "Selected 4D fallback layer '%s' inside nested backbone '%s' (shape: %s)",
                                sublayer.name,
                                nested_model.name,
                                sublayer.output.shape,
                            )
                            return sublayer.name, True
                    except Exception:
                        continue

        # Case 2: Standard flat model (Sequential or Functional)
        preferred_candidates = [
            'relu',
            'conv5_block16_concat',
            'conv5_block16_2_conv',
            'bn',
        ]
        for candidate in preferred_candidates:
            try:
                cand_layer = self.model.get_layer(candidate)
                if hasattr(cand_layer, 'output') and len(cand_layer.output.shape) == 4:
                    logger.info(
                        "Selected target layer '%s' (shape: %s)",
                        candidate,
                        cand_layer.output.shape,
                    )
                    return candidate, False
            except (ValueError, AttributeError):
                continue

        for layer in reversed(self.model.layers):
            try:
                if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                    logger.info(
                        "Selected 4D layer '%s' (shape: %s)",
                        layer.name,
                        layer.output.shape,
                    )
                    return layer.name, False
            except Exception:
                continue

        raise ValueError('No 2D convolutional feature layer found for Grad-CAM generation.')

    def _setup_grad_model(self) -> None:
        self.target_layer_name, self._is_nested = self._find_target_layer()
        logger.info(
            "Grad-CAM initialized with target layer '%s' (nested_backbone=%s)",
            self.target_layer_name,
            self._is_nested,
        )

        if self._is_nested:
            backbone = self.model.layers[0]
            feature_layer = backbone.get_layer(self.target_layer_name)
            self._grad_model = tf.keras.models.Model(
                inputs=backbone.inputs,
                outputs=[feature_layer.output, backbone.output],
            )
            self._top_layers = list(self.model.layers[1:])
        else:
            feature_layer = self.model.get_layer(self.target_layer_name)
            self._grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[feature_layer.output, self.model.outputs[0]],
            )
            self._top_layers = []

    def generate(
        self,
        preprocessed_batch: np.ndarray,
        original_rgb: np.ndarray,
        class_index: int = 1,
    ) -> Image.Image:
        """
        Generates a Grad-CAM heatmap overlay.

        Args:
            preprocessed_batch: Preprocessed image batch of shape (1, H, W, 3) normalized to [0, 1].
            original_rgb: Original RGB image as uint8 numpy array of shape (H, W, 3).
            class_index: Target class index (1 for Pneumonia, 0 for Normal).

        Returns:
            PIL Image of the blended Grad-CAM overlay.
        """
        if self._grad_model is None:
            raise RuntimeError('Grad-CAM model is not initialized.')

        with tf.GradientTape() as tape:
            if self._is_nested:
                conv_outputs, backbone_out = self._grad_model(preprocessed_batch, training=False)
                tape.watch(conv_outputs)
                x = backbone_out
                for layer in self._top_layers:
                    x = layer(x, training=False)
                predictions = x
            else:
                conv_outputs, predictions = self._grad_model(preprocessed_batch, training=False)
                tape.watch(conv_outputs)

            num_classes = predictions.shape[-1]
            if num_classes == 1:
                # Binary sigmoid classifier: class 1 is Pneumonia, class 0 is Normal
                if class_index == 1 or str(class_index).lower() == 'pneumonia':
                    target_score = predictions[:, 0]
                else:
                    target_score = 1.0 - predictions[:, 0]
            else:
                target_score = predictions[:, int(class_index)]

        gradients = tape.gradient(target_score, conv_outputs)
        if gradients is None:
            raise ValueError(
                f"Gradients could not be computed for target layer '{self.target_layer_name}'."
            )

        # Global average pooling on spatial dimensions (H, W)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

        # Weight feature maps by pooled gradients
        conv_outputs_0 = conv_outputs[0]
        heatmap = conv_outputs_0 @ pooled_gradients[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to emphasize features with positive contribution to target class
        heatmap = tf.maximum(heatmap, 0.0)

        # Normalize heatmap to [0, 1]
        max_heat = tf.reduce_max(heatmap)
        if max_heat > 0:
            heatmap = heatmap / (max_heat + 1e-8)
        heatmap_np = heatmap.numpy()

        # Resize heatmap to match dimensions of original RGB image
        target_h, target_w = original_rgb.shape[:2]
        heatmap_resized = cv2.resize(
            heatmap_np,
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        # Convert to 8-bit heatmap and apply JET colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        # Blend original X-ray (60%) and Grad-CAM heatmap (40%)
        overlay = cv2.addWeighted(original_rgb, 0.6, colormap_rgb, 0.4, 0)
        return Image.fromarray(overlay)
