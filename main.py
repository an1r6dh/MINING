import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the web server application
app = FastAPI(title="Mine Subsidence Early Warning System")

MODEL_FILE = "model.joblib"

def load_or_create_model():
    """Loads model from file or automatically generates mock model if missing."""
    if not os.path.exists(MODEL_FILE):
        print(f"Model file '{MODEL_FILE}' not found. Generating default mock model...")
        try:
            import generate_mock
        except Exception as e:
            print(f"Could not run generate_mock: {e}")
    
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        raise RuntimeError("Model file could not be loaded or generated.")

# Load model safely at startup
model = None

@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_or_create_model()
        print("[SUCCESS] Model loaded successfully on server startup.")
    except Exception as e:
        print(f"[WARNING] Could not initialize model on startup: {e}")

# 1. Home route
@app.get("/")
def home():
    return {"message": "Mine Early Warning API is active!", "model_loaded": model is not None}


# Define the exact JSON shape expected from the Kalman filter
class FilteredDataPayload(BaseModel):
    node_id: str
    filtered_tilt: float
    filtered_vibration: float
    filtered_strain: float


# Endpoint to process telemetry and return risk status
@app.post("/predict")
def predict_risk(data: FilteredDataPayload):
    global model
    if model is None:
        try:
            model = load_or_create_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail="ML Model not available on server.")

    try:
        features = [
            [data.filtered_tilt, data.filtered_vibration, data.filtered_strain]
        ]
        prediction = model.predict(features)[0]
        return {"node_id": data.node_id, "status": str(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")