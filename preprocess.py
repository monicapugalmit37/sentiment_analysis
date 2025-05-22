import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def preprocess_data(path):
    df = pd.read_csv(path)
    
    # Clean the review column
    df['cleaned_review'] = df['review'].apply(clean_text)

    # Drop rows with missing labels
    df = df.dropna(subset=['rating', 'cleaned_review'])

    # Define sentiment: rating >= 4 is positive, <=2 is negative, discard 3-star ratings
    df = df[df['rating'] != 3]
    df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')

    # Encode sentiment
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['sentiment'])

    X = df['cleaned_review']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)
