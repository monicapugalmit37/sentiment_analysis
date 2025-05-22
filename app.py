import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os

# Add src folder to sys.path so we can import preprocess module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocess import clean_text

st.title("🤖 AI Echo: ChatGPT Review Sentiment Analyzer")

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


uploaded_file = st.file_uploader("Upload your chatgpt_reviews.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['cleaned'] = df['review'].astype(str).apply(clean_text)

    X = vectorizer.transform(df['cleaned'])
    df['sentiment'] = model.predict(X)

    st.subheader("Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())
    # Show unique sentiment labels to help debug label names
    st.write("Unique sentiments predicted:", df['sentiment'].unique())

    # Adjust this label filter based on what you see above
    positive_label = 'positive'  # Change this if your model outputs different labels

    st.subheader("WordCloud - Positive Reviews")
    text_pos = " ".join(df[df['sentiment'] == 'positive']['cleaned'])
    if text_pos.strip():
        wordcloud = WordCloud(width=800, height=400).generate(text_pos)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write("No positive reviews found to generate word cloud.")
