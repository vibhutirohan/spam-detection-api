import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

MODEL_PATH = os.path.join("saved_models", "spam_model.joblib")

app = FastAPI(
    title="Community Spam Detection API",
    description="REST API for classifying community messages as HAM or SPAM",
    version="1.0.0"
)

# Load model once when the API starts
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)

class MessageRequest(BaseModel):
    sender: Optional[str] = Field(default="Anonymous", example="Rohan")
    message: str = Field(..., example="Earn $1000/week from home. Apply now!")

class MessageResponse(BaseModel):
    sender: str
    message: str
    label: str
    spam_probability: float
    ham_probability: float

class BatchRequest(BaseModel):
    messages: List[MessageRequest]

@app.get("/")
def root():
    return {"message": "Spam Detection API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=MessageResponse)
def predict_message(payload: MessageRequest):
    try:
        probs = model.predict_proba([payload.message])[0]
        classes = model.classes_

        score_map = dict(zip(classes, probs))

        spam_prob = round(float(score_map.get("SPAM", 0.0)) * 100, 2)
        ham_prob = round(float(score_map.get("HAM", 0.0)) * 100, 2)
        label = "SPAM" if spam_prob >= ham_prob else "HAM"

        return MessageResponse(
            sender=payload.sender or "Anonymous",
            message=payload.message,
            label=label,
            spam_probability=spam_prob,
            ham_probability=ham_prob
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
def predict_batch(payload: BatchRequest):
    try:
        texts = [item.message for item in payload.messages]
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
        classes = model.classes_

        results = []
        for item, pred, probs in zip(payload.messages, predictions, probabilities):
            score_map = dict(zip(classes, probs))

            results.append({
                "sender": item.sender or "Anonymous",
                "message": item.message,
                "label": pred,
                "spam_probability": round(float(score_map.get("SPAM", 0.0)) * 100, 2),
                "ham_probability": round(float(score_map.get("HAM", 0.0)) * 100, 2),
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))