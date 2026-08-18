import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class GradCamGenerator:
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def _find_last_conv_layer(self) -> str:
        for layer in reversed(self.model.layers):
            output_shape = getattr(layer, 'output_shape', None)
            if output_shape is not None and len(output_shape) == 4:
                return layer.name

        raise ValueError('No 2D convolutional feature layer found for Grad-CAM.')

    def generate(
        self,
        preprocessed_batch: np.ndarray,
        original_rgb: np.ndarray,
        class_index: int,
    ) -> Image.Image:
        last_conv_layer_name = self._find_last_conv_layer()
        feature_layer = self.model.get_layer(last_conv_layer_name)

        # The supplied saved model is a deserialized Sequential wrapper around
        # DenseNet121. Reconnect its classifier head to the nested DenseNet
        # graph so GradientTape can follow the feature maps to the sigmoid.
        if isinstance(feature_layer, tf.keras.Model):
            feature_index = self.model.layers.index(feature_layer)
            classifier_output = feature_layer.output
            for layer in self.model.layers[feature_index + 1 :]:
                classifier_output = layer(classifier_output, training=False)
            grad_model = tf.keras.models.Model(
                feature_layer.inputs,
                [feature_layer.output, classifier_output],
            )
        else:
            grad_model = tf.keras.models.Model(
                self.model.inputs,
                [feature_layer.output, self.model.outputs[0]],
            )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_batch)
            class_channel = predictions[:, class_index]

        gradients = tape.gradient(class_channel, conv_outputs)

        if gradients is None:
            raise ValueError('Gradients could not be computed for Grad-CAM generation.')

        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap_np = heatmap.numpy()

        heatmap_np = cv2.resize(
            heatmap_np,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        heatmap_uint8 = np.uint8(255 * heatmap_np)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(original_rgb, 0.6, colormap_rgb, 0.4, 0)
        return Image.fromarray(overlay)
