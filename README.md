# PneumoVision AI

## 1. Overview

**PneumoVision AI** is a medical-AI web platform and interpretability workflow for chest X-ray radiograph screening and triage simulation using a fine-tuned DenseNet121 deep neural network with integrated **Grad-CAM (Gradient-weighted Class Activation Mapping) Explainable AI (XAI)**.

### Clinical Interpretation & Interpretability Workflow:
```text
Chest X-Ray Upload (JPG / PNG)
              ↓
DenseNet121 Preprocessing (224x224, 1/255 Normalization)
              ↓
Model Inference (Normal vs. Pneumonia Probability & Confidence)
              ↓
Grad-CAM Gradient Tracking (tf.GradientTape on Target Feature Layer)
              ↓
Global Average Pooling & ReLU Rectification
              ↓
Medical Heatmap Generation (JET Colormap Blend)
              ↓
Explainable AI Result & Visual Attention Presentation
```

---

## 2. Features

- **DenseNet121 Deep Learning Backbone**: High-sensitivity binary classification (Normal vs Pneumonia).
- **Explainable AI (Grad-CAM XAI)**: Visualizes the anatomical regions that drove the model's prediction.
- **Dynamic Layer Discovery**: Automatically detects and connects to the appropriate late convolutional feature layer (`relu` / `conv5_block16_concat`).
- **Interactive Multi-View UI**:
  - Grad-CAM Attention Overlay view
  - Side-by-Side comparison (Original vs. Grad-CAM)
  - Original Radiograph view
  - Instant visualization download
- **Non-Fatal Graceful Degradation**: If explainability generation encounters an issue, core classification prediction remains fully operational.
- **Session Prediction History**: Preserves previous scans and visual thumbnail records in local storage.
- **Health & Readiness Endpoints**: Proactive API status monitoring with model loading diagnostics.
- **Comprehensive Disclaimers**: Clear medical AI research/educational interpretability notices.

---

## 3. Tech Stack

### Frontend:
- **React 19**
- **Vite 8**
- **Lucide React** (Medical & UI icons)
- **Recharts** (Class probability distribution charts)
- **Axios** (API communication with upload progress tracking)
- **Vanilla CSS Design System** (Responsive medical dashboard theme)

### Backend:
- **FastAPI** (High-performance async Python web framework)
- **Uvicorn** (ASGI server)
- **TensorFlow / Keras 2.x/3.x**
- **OpenCV (`cv2`)** (Heatmap generation and colormap blending)
- **Pillow (`PIL`)** (Image manipulation and validation)
- **NumPy** (Array mathematics)

---

## 4. Grad-CAM Explainable AI (XAI) Deep-Dive

### What is Grad-CAM?
**Grad-CAM (Gradient-weighted Class Activation Mapping)** is a visual explanation technique for convolutional neural networks. It produces a coarse 2D localization map highlighting the discriminative regions in the chest radiograph that the model relied on when making a particular classification.

### Why is Grad-CAM Used?
In medical computer vision, "black-box" models can inadvertently learn spurious background artifacts or scanner metadata. Grad-CAM provides clinical interpretability by verifying whether the model is focusing on relevant pulmonary opacities, consolidations, or infiltrates rather than irrelevant features.

### Layer Selection in DenseNet121
DenseNet121 consists of four Dense Blocks with transition layers. The final convolutional block (`conv5_block16`) captures high-level spatial and semantic features before spatial pooling:
- **Selected Target Layer**: `relu` (the final post-batch-normalization activation layer following `conv5_block16_concat`, output shape `(None, 7, 7, 1024)`).
- **Dynamic Identification**: The system inspects both nested functional backbones and top-level sequential graphs dynamically, ensuring compatibility without hardcoded brittle layer names.

### Mathematical Formulation & Pipeline
1. **Forward Pass**: Forward-propagate preprocessed batch $X \in \mathbb{R}^{1 \times 224 \times 224 \times 3}$ to extract feature activations $A^k \in \mathbb{R}^{7 \times 7}$ and class score $y^c$.
   - For Pneumonia ($c=1$): $y^c = p$ (sigmoid probability).
   - For Normal ($c=0$): $y^c = 1 - p$.
2. **Gradient Calculation via `tf.GradientTape`**:
   $$\frac{\partial y^c}{\partial A^k}$$
3. **Global Average Pooling of Gradients**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i,j}^k}$$
4. **Weighted Linear Combination & ReLU Rectification**:
   $$L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
5. **Normalization & Colormap Blending**:
   $$H = \frac{L_{\text{Grad-CAM}}^c}{\max(L_{\text{Grad-CAM}}^c) + 10^{-8}}$$
   The normalized heatmap $H \in [0, 1]$ is resized via bicubic interpolation to original radiograph dimensions $(W, H)$, mapped to the OpenCV `COLORMAP_JET` spectrum, and blended:
   $$\text{Overlay} = 0.6 \cdot I_{\text{RGB}} + 0.4 \cdot I_{\text{Colormap}}$$

---

## 5. API Documentation

### `GET /health`
Returns backend service and model readiness.

**Response Example:**
```json
{
  "status": "ok",
  "server": "ok",
  "model_loaded": true,
  "model_path": "backend/models/best_finetuned.keras",
  "model_error": null
}
```

### `POST /predict`
Uploads a chest radiograph for classification and Grad-CAM generation.

**Request:**
- `Content-Type: multipart/form-data`
- Body: `image` (binary file: PNG, JPEG, JPG up to 8MB)

**Response Schema (Backward Compatible):**
```json
{
  "prediction": "Pneumonia Detected",
  "predicted_class": "pneumonia",
  "pneumonia_probability": 0.9462,
  "normal_probability": 0.0538,
  "confidence": 0.9462,
  "gradcam_available": true,
  "gradcam_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6e...",
  "gradcam": "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6e..."
}
```

---

## 6. Installation & Local Execution

### Prerequisites
- Python 3.10+ (tested with Python 3.11 / 3.12 / 3.13)
- Node.js 18+ & npm

### Backend Setup
1. Open a terminal in the project root:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - **Windows PowerShell / CMD:**
     ```powershell
     .venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```
3. Install backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Start the FastAPI backend server:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```

### Frontend Setup
1. In another terminal window:
   ```bash
   npm install
   ```
2. Start the Vite development server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.

---

## 7. Running Tests

Run the automated integration and unit test suite covering Normal cases, Pneumonia cases, Grad-CAM variance validation, large image resizing, and error handling:

```bash
.venv\Scripts\python.exe -m unittest discover -s backend/tests -p "test_*.py"
```

Frontend build and linter validation:
```bash
npm run lint
npm run build
```

---

## 8. Medical & Explainability Disclaimer

> [!WARNING]
> **Research and Educational Use Only**:
> PneumoVision AI is not a certified medical device and does not provide clinical diagnoses.
> Grad-CAM highlights image regions that influenced the model's prediction for interpretability research. It must not replace clinical evaluation by a qualified medical professional or radiologist.
