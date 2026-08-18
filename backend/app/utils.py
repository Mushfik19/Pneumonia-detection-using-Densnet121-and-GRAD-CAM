import base64
import io
import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/jpg', 'image/png'}


def sanitize_filename(filename: str) -> str:
    base = Path(filename).name
    return re.sub(r'[^A-Za-z0-9._-]', '_', base)


def validate_file_metadata(filename: str, content_type: str) -> None:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError('Unsupported file extension. Use JPG, JPEG, or PNG only.')

    if content_type not in ALLOWED_MIME_TYPES:
        raise ValueError('Unsupported MIME type. Only image/jpeg and image/png are accepted.')


def validate_file_size(file_size_bytes: int, max_upload_mb: int) -> None:
    if file_size_bytes == 0:
        raise ValueError('The uploaded image file is empty.')

    max_bytes = max_upload_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise ValueError(f'File exceeds limit of {max_upload_mb} MB.')


def load_pil_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert('RGB')
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError('Uploaded file is not a valid readable image.') from exc


def pil_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
