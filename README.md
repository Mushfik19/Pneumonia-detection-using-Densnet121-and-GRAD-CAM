# PneumoVision AI

## 1. Overview

PneumoVision AI is a research-oriented web application for chest X-ray image analysis using a DenseNet121 binary classifier.

Workflow:

1. Upload chest X-ray image (JPG/JPEG/PNG)
2. Backend validates and preprocesses image for DenseNet121
3. Model inference returns Normal vs Pneumonia probabilities
4. Grad-CAM generates a model attention overlay when compatible
5. Frontend displays prediction, confidence, chart, and local session history

## 2. Features

- Premium medical-AI dashboard UI (responsive desktop/tablet/mobile)
- Drag-and-drop upload with file validation
- Upload progress and loading states
- Prediction output with confidence and class probabilities
- Probability chart visualization
- Grad-CAM tabbed viewer (Original vs Grad-CAM)
- Download Grad-CAM image
- Session prediction history (localStorage)
- Health endpoint with model availability status
- Graceful model-missing behavior (no fake predictions)
- Explicit research/educational medical disclaimer

## 3. Architecture

- Frontend: React + Vite
- Backend: FastAPI + TensorFlow/Keras
- Model: DenseNet121 binary classifier (external trained model file)

## 4. Tech Stack

Frontend:

- React
- Vite
- Lucide React
- Recharts
- Axios

Backend:

- FastAPI
- Uvicorn
- TensorFlow / Keras
- NumPy
- Pillow
- OpenCV
- python-multipart

## 5. DenseNet121 Model

The backend expects a trained DenseNet121 binary model file.

Supported formats:

- .keras
- .h5

Deployed model path:

- backend/models/best_finetuned.keras

If the model file is missing, backend starts normally and reports:

- Model not loaded with model_error in /health
- /predict returns HTTP 503 with a clear message

## 6. Dataset Preparation

Example dataset layout for training:

dataset/
	normal/
	pneumonia/

Images should be chest radiographs and appropriately curated/labeled by the user.

## 7. Training Overview

Optional training support is included in training/train_densenet121.py with:

- Train/validation/test split
- DenseNet121 transfer learning
- Frozen-base training stage
- Fine-tuning stage
- Class weight handling for imbalance
- Early stopping
- Model checkpointing
- ReduceLROnPlateau
- Metrics: accuracy, AUC, precision, recall
- Evaluation: confusion matrix, ROC curve, F1 score

## 8. Inference Pipeline

Image pipeline in backend:

1. Validate MIME type and extension
2. Enforce file size limit
3. Load image and convert to RGB
4. Resize to model input size (default 224x224)
5. Convert to NumPy float32 array
6. Scale pixel values with `1./255` (the saved model's training preprocessing)
7. Run model inference
8. Compute binary probabilities and confidence

## 9. Grad-CAM

Grad-CAM implementation:

- Finds the last convolutional feature layer automatically
- Computes gradients for predicted output
- Builds heatmap and overlays on resized image
- Returns base64 encoded PNG for frontend visualization

If Grad-CAM is incompatible for a model, result remains valid but Grad-CAM is marked unavailable.

## 10. Project Structure

my-project/
	backend/
		app/
			__init__.py
			config.py
			gradcam.py
			main.py
			model.py
			predictor.py
			schemas.py
			utils.py
		models/
			README.md
		requirements.txt
	public/
	src/
		components/
		hooks/
		pages/
		services/
		utils/
		App.css
		App.jsx
		index.css
		main.jsx
	training/
		README.md
		requirements-training.txt
		train_densenet121.py
	.env.example
	.gitignore
	package.json
	vite.config.js

## 11. Installation

Clone project and open in terminal.

## 12. Running Frontend

Install dependencies:

npm install

Run development server:

npm run dev

Production build:

npm run build

## 13. Running Backend

From project root:

python -m venv venv

Windows:

venv\Scripts\activate

Install backend dependencies:

pip install -r backend/requirements.txt

Run API server:

uvicorn backend.app.main:app --reload

## 14. Model Installation

The deployed trained model file is:

backend/models/best_finetuned.keras

Or set custom path:

MODEL_PATH=backend/models/your_model_file.keras

## 15. API Documentation

### GET /health

Returns backend and model status.

Example response:

{
	"server": "ok",
	"model_loaded": false,
	"model_path": "backend/models/densenet121_pneumonia.keras",
	"model_error": "Model file not found ..."
}

### POST /predict

Request:

- multipart/form-data
- field name: image

Response:

{
	"prediction": "Pneumonia Detected",
	"predicted_class": "pneumonia",
	"pneumonia_probability": 0.93,
	"normal_probability": 0.07,
	"confidence": 0.93,
	"gradcam_available": true,
	"gradcam_image_base64": "..."
}

## 16. Screenshots

Add screenshots here after running the app:

- Dashboard
- Upload and prediction result
- Grad-CAM visualization

## 17. Limitations

- Predictions depend entirely on model quality and dataset domain
- Binary output only (normal vs pneumonia)
- No DICOM-native processing in current version
- No authentication or audit trails for clinical deployment

## 18. Medical Disclaimer

Research and Educational Use Only.

PneumoVision AI is not a medical device and does not provide a medical diagnosis.
Results should not replace evaluation by a qualified healthcare professional.

## 19. Future Improvements

- Multi-class thoracic pathology classification
- DICOM support and metadata handling
- User authentication and secure audit logging
- Report generation export pipeline
- Model registry and versioned deployment
- Optional PACS integration
