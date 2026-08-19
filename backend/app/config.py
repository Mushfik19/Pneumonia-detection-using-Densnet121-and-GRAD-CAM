import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = 'PneumoVision AI API'
    app_version: str = '1.0.0'
    model_path: str = os.getenv(
        'MODEL_PATH',
        str(Path(__file__).resolve().parent.parent / 'models' / 'best_finetuned.keras'),
    )
    allowed_origins: tuple[str, ...] = tuple(
        value.strip()
        for value in os.getenv(
            'ALLOWED_ORIGINS',
            'http://localhost:5173,http://127.0.0.1:5173,https://pneumonia-detection-using-densnet12.vercel.app',
        ).split(',')
        if value.strip()
    )
    max_upload_mb: int = int(os.getenv('MAX_UPLOAD_MB', '8'))
    input_width: int = int(os.getenv('MODEL_INPUT_WIDTH', '224'))
    input_height: int = int(os.getenv('MODEL_INPUT_HEIGHT', '224'))
    positive_threshold: float = float(os.getenv('PNEUMONIA_THRESHOLD', '0.5'))


settings = Settings()
