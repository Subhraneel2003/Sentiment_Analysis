from fastapi import FastAPI
from pydantic import BaseModel
from app.models_util import predict_sentiment
app=FastAPI()
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextInput,model: str="lr", vectortize: str="tfidf"):
    prediction=predict_sentiment(payload.text,model_name=model, vectorizer_name=vectortize)
    return {'sentiment': prediction}