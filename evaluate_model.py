import joblib
from preprocess import preprocess_data
from sklearn.metrics import classification_report, accuracy_score
DATA_PATH = "C:/Users/MONICA PUGAZHENDHI/Desktop/sentimental_analysis/data/sentiment_dataset.csv"

def evaluate():
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)


    model = joblib.load("models/sentiment_model.pkl")
    tfidf = joblib.load("models/vectorizer.pkl")

    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
