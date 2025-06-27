from fastapi import FastAPI
import nltk
from pydantic import BaseModel
from app.models_util import predict_sentiment
nltk.data.path.append("./nltk_data")


app = FastAPI()
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextInput,model: str="lr", vectortize: str="tfidf"):
    prediction=predict_sentiment(payload.text,model_name=model, vectorizer_name=vectortize)
    return {'sentiment': prediction}