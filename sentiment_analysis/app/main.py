from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from .preprocess import clean_text

# Load artifacts
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

app = FastAPI()

class Review(BaseModel):
    text: str

@app.post('/predict')
def predict(review: Review):
    cleaned = clean_text(review.text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return {'sentiment': pred}