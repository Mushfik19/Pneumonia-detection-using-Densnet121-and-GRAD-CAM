from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .auth import LoginPayload, RegisterPayload, current_user, init_auth_db, login, logout, register
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
    init_auth_db()
    service.load_model()

@app.post('/auth/register')
def auth_register(payload: RegisterPayload):
    user = register(payload); token, user = login(LoginPayload(email=user['email'], password=payload.password))
    return {'message': 'Account created successfully', 'access_token': token, 'token_type': 'bearer', 'user': user}

@app.post('/auth/login')
def auth_login(payload: LoginPayload):
    token, user = login(payload)
    return {'access_token': token, 'token_type': 'bearer', 'user': user}

@app.get('/auth/me')
def auth_me(user: dict = Depends(current_user)): return {'user': user}

@app.post('/auth/logout', status_code=204)
def auth_logout(authorization: str | None = Header(default=None)): logout(authorization)


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
async def predict(image: UploadFile = File(...), user: dict = Depends(current_user)) -> JSONResponse:
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
