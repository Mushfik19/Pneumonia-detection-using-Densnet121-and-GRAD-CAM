from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .predictor import ModelNotLoadedError, PredictionService
from .schemas import HealthResponse, PredictionResponse
from .utils import (
    load_pil_image,
    sanitize_filename,
    validate_file_metadata,
    validate_file_size,
)

app = FastAPI(title=settings.app_name, version=settings.app_version)
service = PredictionService(settings=settings)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


@app.on_event('startup')
def startup_event() -> None:
    service.load_model()


@app.get('/health', response_model=HealthResponse)
def health() -> dict:
    return {
        'status': 'ok' if service.model_loaded else 'degraded',
        'server': 'ok',
        'model_loaded': service.model_loaded,
        'model_path': settings.model_path,
        'model_error': service.model_error,
    }


@app.post('/predict', response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    try:
        sanitized_name = sanitize_filename(image.filename or 'upload.png')
        validate_file_metadata(sanitized_name, image.content_type or '')

        image_bytes = await image.read()
        validate_file_size(len(image_bytes), settings.max_upload_mb)

        pil_image = load_pil_image(image_bytes)

        prediction = service.predict(pil_image)
        payload = {
            'prediction': prediction.prediction,
            'predicted_class': prediction.predicted_class,
            'pneumonia_probability': prediction.pneumonia_probability,
            'normal_probability': prediction.normal_probability,
            'confidence': prediction.confidence,
            'gradcam_available': prediction.gradcam_available,
            'gradcam_image_base64': prediction.gradcam_image_base64,
        }
        return JSONResponse(content=payload)

    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail='Prediction failed due to an internal processing error.',
        )
    finally:
        await image.close()
