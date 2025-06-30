# Sentiment Analysis API with FastAPI & Machine Learning

A comprehensive sentiment analysis system built with **FastAPI** and machine learning models trained on the **IMDB Movie Review Dataset**. This project supports multiple ML models and vectorization techniques, providing dynamic sentiment prediction via REST API.

## 🚀 Features

- **Advanced Text Preprocessing**: HTML tag removal, stopword filtering, and lemmatization
- **Multiple ML Models**: 6 trained models with different algorithms and vectorization approaches
  - Logistic Regression (BoW & TF-IDF)
  - Support Vector Machine (BoW & TF-IDF)
  - Multinomial Naive Bayes (BoW & TF-IDF)
- **Dual Vectorization**: Bag of Words (BoW) and TF-IDF support
- **RESTful API**: Built with FastAPI for high performance
- **Interactive Documentation**: Automatic Swagger UI generation
- **Model Persistence**: Serialized models and vectorizers using joblib
- **Modular Architecture**: Clean, maintainable Python codebase
- **Version Control**: Git-tracked development

## 📁 Project Structure

```
Sentiment_Analysis/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── models_util.py       # Model loading and prediction utilities
│   └── schemas.py           # Pydantic data schemas
├── models/
│   ├── lr_bow_model.pkl      # Logistic Regression + BoW
│   ├── lr_tfidf_model.pkl    # Logistic Regression + TF-IDF
│   ├── svm_bow_model.pkl     # SVM + BoW
│   ├── svm_tfidf_model.pkl   # SVM + TF-IDF
│   ├── mnb_bow_model.pkl     # Naive Bayes + BoW
│   ├── mnb_tfidf_model.pkl   # Naive Bayes + TF-IDF
│   ├── bow_vectorizer.pkl               # Bag of Words vectorizer
│   └── tfidf_vectorizer.pkl             # TF-IDF vectorizer
├── Sentiment.ipynb         # Model training and evaluation notebook
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Subhraneel2003/Sentiment_Analysis.git
   cd Sentiment_Analysis
   ```

2. **Create virtual environment** (recommended)
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Application

1. **Start the FastAPI server**
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Access the application**
   - API Documentation: http://127.0.0.1:8000/docs
   - Alternative Documentation: http://127.0.0.1:8000/redoc
   - API Base URL: http://127.0.0.1:8000

3. **Test using Swagger UI**
   Navigate to the `/docs` endpoint to use the interactive API documentation for testing.

## 📡 API Usage

### Prediction Endpoint

**POST** `/predict`

#### Query Parameters
- `model`: Model type (`lr`, `svm`, `mnb`)
- `vectorizer`: Vectorization method (`bow`, `tfidf`)

#### Request Body
```json
{
  "text": "This movie was absolutely amazing! Great acting and storyline."
}
```

#### Response
```json
{
  "sentiment": "positive"
}
```

#### Example Requests

```bash
# Logistic Regression with TF-IDF
curl -X POST "http://127.0.0.1:8000/predict?model=lr&vectorizer=tfidf" \
     -H "Content-Type: application/json" \
     -d '{"text": "I loved this movie!"}'

# SVM with Bag of Words
curl -X POST "http://127.0.0.1:8000/predict?model=svm&vectorizer=bow" \
     -H "Content-Type: application/json" \
     -d '{"text": "Terrible movie, waste of time."}'
```

## 🤖 Supported Models & Vectorizers

### Models
| Code | Algorithm | Description |
|------|-----------|-------------|
| `lr` | Logistic Regression | Linear classification algorithm |
| `svm` | Support Vector Machine | Margin-based classification |
| `mnb` | Multinomial Naive Bayes | Probabilistic classification |

### Vectorizers
| Code | Method | Description |
|------|--------|-------------|
| `bow` | Bag of Words | Word frequency-based vectorization |
| `tfidf` | TF-IDF | Term frequency-inverse document frequency |

## 🛠️ Tech Stack

- **Backend Framework**: FastAPI
- **ASGI Server**: Uvicorn
- **Machine Learning**: scikit-learn
- **Natural Language Processing**: NLTK
- **Model Serialization**: joblib
- **Data Validation**: Pydantic
- **Language**: Python 3.10+

## 🎯 Learning Objectives

This project demonstrates:
- Training and persisting machine learning models
- Building robust text preprocessing pipelines
- Creating RESTful APIs with FastAPI
- Implementing model serving architecture
- Using interactive API documentation
- Applying software engineering best practices

## 🔮 Future Enhancements

- [ ] **Enhanced Responses**: Add confidence scores and probability distributions
- [ ] **Model Management**: `/models` endpoint to list available models and their performance
- [ ] **Frontend Interface**: Streamlit or React-based web interface
- [ ] **Containerization**: Docker support for easy deployment
- [ ] **Cloud Deployment**: Deploy on AWS, GCP, or Azure
- [ ] **Model Monitoring**: Performance tracking and logging
- [ ] **Batch Processing**: Support for multiple text predictions
- [ ] **Real-time Updates**: WebSocket support for live predictions

## 📊 Model Performance

The models were trained on the IMDB Movie Review Dataset with the following preprocessing pipeline:
- HTML tag removal
- Lowercasing
- Stopword removal
- Lemmatization
- Vectorization (BoW/TF-IDF)

*Note: Detailed performance metrics can be found in the `Sentiment.ipynb` notebook.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License © 2025 Subhraneel Das
```

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework for building APIs
- [scikit-learn](https://scikit-learn.org/) - Machine learning library for Python
- [NLTK](https://www.nltk.org/) - Natural Language Toolkit
- [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) - Training data source

---

**Built with ❤️ by [Subhraneel Das](https://github.com/Subhraneel2003)**
