"""
Simple FastAPI app serving a scikit-learn model.

Run locally:
    uvicorn main:app --reload

Test it:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Sentiment Analysis API",
)

# Load the model once at startup
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

class SentimentRequest(BaseModel):
    text: str = Field(
        ..., min_length=1,
        description="Review text to analyze for sentiment. Must be a non-empty string."
    )

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float | None = None

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running. See /docs for usage."}

@app.get("/health")
def health():
    """Basic health check endpoint — useful for deployment platforms & load balancers."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    pred = model.predict([request.text])[0]
    label = "positive" if pred == 1 else "negative"

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba([request.text]).max())

    return SentimentResponse(sentiment=label, confidence=confidence)

from fastapi.responses import HTMLResponse

@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Sentiment Tester</title></head>
    <body style="font-family: sans-serif; max-width: 500px; margin: 60px auto;">
        <h2>Sentiment Analyzer</h2>
        <textarea id="text" rows="4" style="width:100%; font-size:16px;"
            placeholder="Type a sentence..."></textarea><br><br>
        <button onclick="predict()" style="padding:8px 16px; font-size:16px;">Predict</button>
        <h3 id="result" style="margin-top:20px;"></h3>

        <script>
            async function predict() {
                const text = document.getElementById("text").value;
                const res = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({text: text})
            });
        const data = await res.json();
        document.getElementById("result").innerText =
            data.sentiment + " (confidence: " + (data.confidence ?? "n/a") + ")";
        }
        </script>
    </body>
    </html>
"""