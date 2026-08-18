from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str
    predicted_class: str
    pneumonia_probability: float
    normal_probability: float
    confidence: float
    gradcam_available: bool
    gradcam_image_base64: str | None = None


class HealthResponse(BaseModel):
    status: str
    server: str
    model_loaded: bool
    model_path: str
    model_error: str | None = None
