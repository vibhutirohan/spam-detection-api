import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

MODEL_PATH = os.path.join("saved_models", "spam_model.joblib")

app = FastAPI(
    title="Community Spam Detection API",
    description="REST API for checking whether content is good or bad",
    version="2.0.1"
)

# Load model once when the API starts
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


class MessageRequest(BaseModel):
    title: str = Field(..., example="Free Offer")
    description: str = Field(..., example="Click here now to claim your free reward!")


class MessageResponse(BaseModel):
    status: str = Field(..., example="bad")


class BatchMessageRequest(BaseModel):
    title: str = Field(..., example="Special Offer")
    description: str = Field(..., example="You have won a free prize. Click now!")


class BatchRequest(BaseModel):
    messages: List[BatchMessageRequest]


@app.get("/")
def root():
    return {"message": "Spam Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=MessageResponse)
def predict_message(payload: MessageRequest):
    try:
        title = payload.title.strip()
        description = payload.description.strip()

        if not title and not description:
            raise HTTPException(
                status_code=400,
                detail="Both title and description cannot be empty"
            )

        # Combine title + description into one text input for the model
        full_text = f"{title} {description}".strip()

        prediction = str(model.predict([full_text])[0]).strip().upper()

        # HAM -> good, SPAM -> bad
        if prediction == "SPAM":
            return MessageResponse(status="bad")
        else:
            return MessageResponse(status="good")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
def predict_batch(payload: BatchRequest):
    try:
        if not payload.messages:
            raise HTTPException(
                status_code=400,
                detail="Messages list cannot be empty"
            )

        full_texts = []
        for item in payload.messages:
            title = item.title.strip()
            description = item.description.strip()

            if not title and not description:
                full_texts.append("")
            else:
                full_texts.append(f"{title} {description}".strip())

        predictions = model.predict(full_texts)

        results = []
        for item, pred in zip(payload.messages, predictions):
            pred = str(pred).strip().upper()
            status = "bad" if pred == "SPAM" else "good"

            results.append({
                "title": item.title,
                "description": item.description,
                "status": status
            })

        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))