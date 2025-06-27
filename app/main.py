from fastapi import FastAPI
import nltk
from pydantic import BaseModel
from app.models_util import predict_sentiment
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    nltk.download('stopwords')
    nltk.download('punkt')  # only if you're using it
    yield

app = FastAPI(lifespan=lifespan)
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(payload: TextInput,model: str="lr", vectortize: str="tfidf"):
    prediction=predict_sentiment(payload.text,model_name=model, vectorizer_name=vectortize)
    return {'sentiment': prediction}