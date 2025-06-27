import joblib
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
nltk.data.path.append("app/nltk_data")  # Add this line
stopword_list = stopwords.words('english')

def preprocess_text(text):
        # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Lowercase
    text = text.lower()

    # Tokenization
    tokens = tokenizer.tokenize(text)

    # Remove stopwords and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stopword_list]

    return ' '.join(filtered_tokens)

def load_model_and_vectorizer(model_name, vectorizer_name):
    model_name = model_name.lower()
    vectorizer_name = vectorizer_name.lower()
    model_path = f"Models/{model_name}_{vectorizer_name}_model.pkl"
    vectorizer_path = f"Models/{vectorizer_name}_vectorizer.pkl"

    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise ValueError(f"Vectorizer file not found: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(text, model_name='lr', vectorizer_name='tfidf'):
    cleaned_text = preprocess_text(text)
    model, vectorizer = load_model_and_vectorizer(model_name, vectorizer_name)

    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)

    return "positive" if prediction[0] == 1 else "negative"

