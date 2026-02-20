"""
Lung Cancer Detection API
FastAPI backend serving trained Quantum, Random Forest, XGBoost, and Ensemble models.
Designed for deployment on Railway and consumption by a Flutter mobile app.
"""
import os
import io
import base64
import time
import traceback
import numpy as np
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# ── Project imports ───────────────────────────────────────────────────
from src.utils import load_config
from src.feature_extraction import (
    ResNet2D, DeepLearningFeatureExtractor2D, load_2d_image
)
from src.dimensionality_reduction import DimensionalityReducer
from src.hybrid_quantum import HybridQuantumClassifier
from src.classical_baseline import ClassicalBaseline
from src.universal_preprocessing import universal_preprocess

import torch

# ======================================================================
# Global state
# ======================================================================
MODELS = {}  # Will hold loaded model objects
DIM_REDUCER = None
FEATURE_EXTRACTOR = None
CONFIG = None
IS_HYBRID_QUANTUM = False
MODELS_LOADED = False

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "trained_models"
CONFIG_PATH = BASE_DIR / "config_api.yaml"


def load_all_models():
    """Load all 2D models into memory at startup."""
    global MODELS, DIM_REDUCER, FEATURE_EXTRACTOR, CONFIG
    global IS_HYBRID_QUANTUM, MODELS_LOADED

    print("=" * 60)
    print("  Loading Lung Cancer Detection Models ...")
    print("=" * 60)

    # 1. Config
    CONFIG = load_config(str(CONFIG_PATH))
    device = torch.device("cpu")
    print(f"  Device: {device}")

    # 2. Feature extractor (ResNet2D pretrained)
    print("  Loading ResNet2D feature extractor ...")
    FEATURE_EXTRACTOR = DeepLearningFeatureExtractor2D(device=device)

    # 3. Dimensionality reducer
    reducer_path = MODELS_DIR / "dim_reducer_2d.pkl"
    if reducer_path.exists():
        with open(reducer_path, "rb") as f:
            DIM_REDUCER = pickle.load(f)
        print(f"  ✓ Loaded dim_reducer from {reducer_path}")
    else:
        raise FileNotFoundError(f"dim_reducer not found at {reducer_path}")

    # 4. Quantum model
    quantum_path = MODELS_DIR / "quantum_model_2d.pkl"
    if quantum_path.exists():
        with open(quantum_path, "rb") as f:
            model_data = pickle.load(f)

        if "classifier" in model_data:
            qm = HybridQuantumClassifier(CONFIG)
            qm.load(str(quantum_path))
            IS_HYBRID_QUANTUM = True
            MODELS["quantum"] = qm
            print(f"  ✓ Loaded Hybrid Quantum model")
        else:
            from src.quantum_kernel import QuantumKernelClassifier
            qm = QuantumKernelClassifier(CONFIG)
            qm.load(str(quantum_path))
            IS_HYBRID_QUANTUM = False
            MODELS["quantum"] = qm
            print(f"  ✓ Loaded Quantum Kernel model")
    else:
        print(f"  ⚠ Quantum model not found at {quantum_path}")

    # 5. Random Forest
    rf_path = MODELS_DIR / "random_forest_model_2d.pkl"
    if rf_path.exists():
        rf = ClassicalBaseline(CONFIG, model_type="random_forest")
        rf.load(str(rf_path))
        MODELS["random_forest"] = rf
        print(f"  ✓ Loaded Random Forest model")
    else:
        print(f"  ⚠ Random Forest model not found")

    # 6. XGBoost
    xgb_path = MODELS_DIR / "xgboost_model_2d.pkl"
    if xgb_path.exists():
        xgb = ClassicalBaseline(CONFIG, model_type="xgboost")
        xgb.load(str(xgb_path))
        MODELS["xgboost"] = xgb
        print(f"  ✓ Loaded XGBoost model")
    else:
        print(f"  ⚠ XGBoost model not found")

    MODELS_LOADED = True
    print("=" * 60)
    print(f"  ✅ {len(MODELS)} models loaded successfully!")
    print("=" * 60)


# ======================================================================
# FastAPI lifespan — load models on startup
# ======================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    # Cleanup on shutdown
    MODELS.clear()


app = FastAPI(
    title="Lung Cancer Detection API",
    description=(
        "Quantum-Enhanced Lung Cancer Detection from CT/X-ray images. "
        "Serves predictions from Hybrid Quantum, Random Forest, XGBoost, "
        "and Ensemble models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Flutter app from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# Pydantic models
# ======================================================================
class Base64Request(BaseModel):
    image: str  # base64-encoded image bytes
    filename: Optional[str] = "image.jpg"


class ModelPrediction(BaseModel):
    prediction: str  # "Cancerous" or "Non-Cancerous"
    probability_cancerous: float
    probability_non_cancerous: float
    confidence: float


# ======================================================================
# Inference helpers
# ======================================================================
def run_inference(image_array: np.ndarray) -> dict:
    """
    Run the full prediction pipeline on a 2D image array.

    Args:
        image_array: (H, W) float32 array normalised to [0, 1]

    Returns:
        dict with per-model predictions and ensemble
    """
    # 1. Preprocess
    image_array = universal_preprocess(image_array, source_format="auto")

    # 2. Extract ResNet2D features → (512,)
    features = FEATURE_EXTRACTOR.extract_single(image_array)
    features = np.asarray(features, dtype=np.float64).reshape(1, -1)  # (1, 512)

    results = {}

    # 3. Quantum model
    if "quantum" in MODELS:
        if IS_HYBRID_QUANTUM:
            feats_q = np.asarray(DIM_REDUCER.transform_pca_only(features), dtype=np.float64)
        else:
            feats_q = np.asarray(DIM_REDUCER.transform(features), dtype=np.float64)

        proba = MODELS["quantum"].predict_proba(feats_q)[0]
        pred = 1 if proba[1] >= 0.5 else 0
        results["quantum"] = {
            "model_type": "Hybrid Quantum" if IS_HYBRID_QUANTUM else "Quantum Kernel",
            "prediction": "Cancerous" if pred == 1 else "Non-Cancerous",
            "probability_cancerous": round(float(proba[1]), 4),
            "probability_non_cancerous": round(float(proba[0]), 4),
            "confidence": round(float(max(proba)), 4),
        }

    # 4. Random Forest
    if "random_forest" in MODELS:
        proba = MODELS["random_forest"].predict_proba(features)[0]
        pred = 1 if proba[1] >= 0.5 else 0
        results["random_forest"] = {
            "prediction": "Cancerous" if pred == 1 else "Non-Cancerous",
            "probability_cancerous": round(float(proba[1]), 4),
            "probability_non_cancerous": round(float(proba[0]), 4),
            "confidence": round(float(max(proba)), 4),
        }

    # 5. XGBoost
    if "xgboost" in MODELS:
        proba = MODELS["xgboost"].predict_proba(features)[0]
        pred = 1 if proba[1] >= 0.5 else 0
        results["xgboost"] = {
            "prediction": "Cancerous" if pred == 1 else "Non-Cancerous",
            "probability_cancerous": round(float(proba[1]), 4),
            "probability_non_cancerous": round(float(proba[0]), 4),
            "confidence": round(float(max(proba)), 4),
        }

    # 6. Ensemble (median-based)
    if len(results) > 1:
        probs = [r["probability_cancerous"] for r in results.values()]
        ensemble_prob = float(np.median(probs))
        ensemble_pred = 1 if ensemble_prob >= 0.5 else 0
        results["ensemble"] = {
            "prediction": "Cancerous" if ensemble_pred == 1 else "Non-Cancerous",
            "probability_cancerous": round(ensemble_prob, 4),
            "probability_non_cancerous": round(1 - ensemble_prob, 4),
            "confidence": round(max(ensemble_prob, 1 - ensemble_prob), 4),
        }

        # 7. Calibrated quantum (blend 20% quantum + 80% classical)
        classical_probs = [
            r["probability_cancerous"]
            for k, r in results.items()
            if k not in ("quantum", "ensemble")
        ]
        if "quantum" in results and classical_probs:
            q_prob = results["quantum"]["probability_cancerous"]
            c_consensus = float(np.mean(classical_probs))
            cal_prob = 0.2 * q_prob + 0.8 * c_consensus
            cal_pred = 1 if cal_prob >= 0.5 else 0
            results["quantum_calibrated"] = {
                "prediction": "Cancerous" if cal_pred == 1 else "Non-Cancerous",
                "probability_cancerous": round(cal_prob, 4),
                "probability_non_cancerous": round(1 - cal_prob, 4),
                "confidence": round(max(cal_prob, 1 - cal_prob), 4),
                "raw_quantum_prob": round(q_prob, 4),
                "classical_consensus": round(c_consensus, 4),
            }

    return results


def load_image_bytes(data: bytes, filename: str = "image.jpg") -> np.ndarray:
    """Load image from raw bytes into a (H, W) float32 array."""
    from PIL import Image as PILImage

    ext = Path(filename).suffix.lower()

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        img = PILImage.open(io.BytesIO(data)).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    elif ext == ".dcm":
        # Write to temp file for SimpleITK / pydicom
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return load_2d_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    elif ext == ".npy":
        arr = np.load(io.BytesIO(data)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        if arr.max() != arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        return arr

    else:
        # Try PIL as fallback
        try:
            img = PILImage.open(io.BytesIO(data)).convert("L")
            return np.array(img, dtype=np.float32) / 255.0
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format: {ext}. Use .jpg, .png, .dcm, or .npy",
            )


# ======================================================================
# API Endpoints
# ======================================================================

@app.get("/")
async def health_check():
    """Health check — shows API status and loaded models."""
    return {
        "status": "healthy" if MODELS_LOADED else "loading",
        "models_loaded": MODELS_LOADED,
        "available_models": list(MODELS.keys()),
        "model_count": len(MODELS),
        "api_version": "1.0.0",
        "description": "Quantum-Enhanced Lung Cancer Detection API",
        "endpoints": {
            "/predict": "POST — Upload image file for prediction",
            "/predict/base64": "POST — Send base64-encoded image for prediction",
        },
        "supported_formats": ["jpg", "jpeg", "png", "dcm", "npy", "bmp", "tiff"],
        "accuracy": {
            "quantum": "93.65%",
            "random_forest": "92.06%",
            "xgboost": "85.40%",
            "ensemble": "90.79%",
        },
    }


@app.post("/predict")
async def predict_upload(file: UploadFile = File(...)):
    """
    Predict lung cancer from an uploaded CT image.

    Accepts: JPG, PNG, DICOM (.dcm), NumPy (.npy)
    Returns: Predictions from all models with probabilities.
    """
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models are still loading")

    start = time.time()

    try:
        data = await file.read()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        image = load_image_bytes(data, file.filename or "image.jpg")
        predictions = run_inference(image)
        elapsed = round(time.time() - start, 3)

        return {
            "status": "success",
            "filename": file.filename,
            "image_shape": list(image.shape),
            "inference_time_seconds": elapsed,
            "predictions": predictions,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/base64")
async def predict_base64(request: Base64Request):
    """
    Predict lung cancer from a base64-encoded CT image.

    Send JSON: {"image": "<base64_string>", "filename": "scan.jpg"}
    Returns: Predictions from all models with probabilities.
    """
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models are still loading")

    start = time.time()

    try:
        # Decode base64
        try:
            image_data = base64.b64decode(request.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")

        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image data")

        image = load_image_bytes(image_data, request.filename or "image.jpg")
        predictions = run_inference(image)
        elapsed = round(time.time() - start, 3)

        return {
            "status": "success",
            "filename": request.filename,
            "image_shape": list(image.shape),
            "inference_time_seconds": elapsed,
            "predictions": predictions,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ======================================================================
# Entry point (for local dev)
# ======================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
