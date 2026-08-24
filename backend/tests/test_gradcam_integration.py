import asyncio
import base64
import io
import unittest
import numpy as np
from PIL import Image, ImageDraw
from fastapi import HTTPException, UploadFile

from backend.app.config import settings
from backend.app.gradcam import GradCamGenerator
from backend.app.main import health, predict, service
from backend.app.predictor import PredictionService


def create_synthetic_xray(pattern_type: str = "normal", size: tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Creates a synthetic chest radiograph for testing.
    - 'normal': Clear lung fields with minimal opacity.
    - 'pneumonia': Prominent consolidation / opacity pattern in one or both lung fields.
    """
    img = Image.new("RGB", size, color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    w, h = size
    # Draw spine/ribs approximation
    draw.line([(w // 2, int(h * 0.1)), (w // 2, int(h * 0.9))], fill=(70, 70, 70), width=6)

    # Left and right lung fields
    left_box = [int(w * 0.15), int(h * 0.2), int(w * 0.45), int(h * 0.8)]
    right_box = [int(w * 0.55), int(h * 0.2), int(w * 0.85), int(h * 0.8)]

    draw.ellipse(left_box, fill=(40, 40, 40), outline=(60, 60, 60))
    draw.ellipse(right_box, fill=(40, 40, 40), outline=(60, 60, 60))

    if pattern_type == "pneumonia":
        # Dense opacities in lung fields
        draw.ellipse([int(w * 0.58), int(h * 0.45), int(w * 0.80), int(h * 0.75)], fill=(190, 190, 190))
        draw.ellipse([int(w * 0.20), int(h * 0.50), int(w * 0.35), int(h * 0.70)], fill=(160, 160, 160))
    elif pattern_type == "normal":
        draw.ellipse([int(w * 0.58), int(h * 0.45), int(w * 0.80), int(h * 0.75)], fill=(45, 45, 45))

    return img


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


class TestGradCamIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        service.load_model()

    def test_01_model_loaded_and_layer_identified(self):
        self.assertTrue(service.model_loaded, f"Model failed to load: {service.model_error}")
        self.assertIsNotNone(service.gradcam_generator)
        self.assertIsNotNone(service.gradcam_generator.target_layer_name)
        self.assertIn(
            service.gradcam_generator.target_layer_name,
            ['relu', 'conv5_block16_concat', 'conv5_block16_2_conv', 'bn', 'densenet121'],
        )

    def test_02_gradcam_generation_normal_image(self):
        normal_img = create_synthetic_xray("normal")
        result = service.predict(normal_img)

        self.assertIn(result.prediction, ["Normal", "Pneumonia Detected"])
        self.assertIn(result.predicted_class, ["normal", "pneumonia"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertTrue(result.gradcam_available)
        self.assertIsNotNone(result.gradcam_image_base64)

        # Decode base64 PNG
        raw_bytes = base64.b64decode(result.gradcam_image_base64)
        cam_img = Image.open(io.BytesIO(raw_bytes))
        self.assertEqual(cam_img.size, (224, 224))
        self.assertEqual(cam_img.mode, "RGB")

        # Validate heatmap variance (non-uniform, non-blank)
        cam_array = np.array(cam_img)
        self.assertGreater(np.std(cam_array), 5.0, "Grad-CAM heatmap should not be uniform.")

    def test_03_gradcam_generation_pneumonia_image(self):
        pneumonia_img = create_synthetic_xray("pneumonia")
        result = service.predict(pneumonia_img)

        self.assertIn(result.prediction, ["Normal", "Pneumonia Detected"])
        self.assertTrue(result.gradcam_available)
        self.assertIsNotNone(result.gradcam_image_base64)

        raw_bytes = base64.b64decode(result.gradcam_image_base64)
        cam_img = Image.open(io.BytesIO(raw_bytes))
        self.assertEqual(cam_img.size, (224, 224))

        cam_array = np.array(cam_img)
        self.assertGreater(np.std(cam_array), 5.0, "Grad-CAM heatmap should have non-zero variance.")

    def test_04_large_image_handling(self):
        large_img = create_synthetic_xray("normal", size=(1024, 1024))
        result = service.predict(large_img)

        self.assertTrue(result.gradcam_available)
        raw_bytes = base64.b64decode(result.gradcam_image_base64)
        cam_img = Image.open(io.BytesIO(raw_bytes))
        self.assertEqual(cam_img.size, (224, 224))

    def test_05_api_health_endpoint(self):
        res = health()
        self.assertEqual(res["server"], "ok")
        self.assertTrue(res["model_loaded"])
        self.assertIsNone(res["model_error"])

    def test_06_api_predict_endpoint_success(self):
        img = create_synthetic_xray("pneumonia")
        img_bytes = image_to_bytes(img, "PNG")

        upload_file = UploadFile(
            file=io.BytesIO(img_bytes),
            filename="chest_xray.png",
            headers={"content-type": "image/png"},
        )

        response = asyncio.run(predict(upload_file))
        self.assertEqual(response.status_code, 200)
        import json
        data = json.loads(response.body.decode("utf-8"))

        self.assertIn("prediction", data)
        self.assertIn("predicted_class", data)
        self.assertIn("confidence", data)
        self.assertIn("pneumonia_probability", data)
        self.assertIn("normal_probability", data)
        self.assertTrue(data["gradcam_available"])
        self.assertIsNotNone(data["gradcam_image_base64"])
        self.assertIsNotNone(data["gradcam"])
        self.assertEqual(data["gradcam_image_base64"], data["gradcam"])

    def test_07_api_predict_invalid_extension(self):
        upload_file = UploadFile(
            file=io.BytesIO(b"invalid data"),
            filename="test.txt",
            headers={"content-type": "text/plain"},
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(predict(upload_file))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_08_api_predict_corrupted_image(self):
        upload_file = UploadFile(
            file=io.BytesIO(b"not-a-real-image-data-stream"),
            filename="corrupted.png",
            headers={"content-type": "image/png"},
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(predict(upload_file))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
