import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess_data

def train():
    X_train, X_test, y_train, y_test = preprocess_data("C:/Users/MONICA PUGAZHENDHI/Desktop/sentimental_analysis/data/sentiment_dataset.csv")

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Save model and vectorizer
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(tfidf, "models/vectorizer.pkl")
    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    train()
