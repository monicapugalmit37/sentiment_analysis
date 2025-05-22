# sentiment_analysis
This project is a Sentiment Analysis web application built using Streamlit and Machine Learning. It classifies customer reviews (especially ChatGPT reviews) as positive or negative using a Logistic Regression model trained on TF-IDF vectorized text.
Here is a complete `README.md` file for your **Sentiment Analysis** project:

---

````markdown
# 📊 Sentiment Analysis

This is a **Sentiment Analysis Web Application** that classifies customer reviews as either **positive** or **negative** using a trained **Logistic Regression** model. The app is built using **Streamlit** and supports user file uploads, sentiment prediction, visualization, and word cloud generation.

## 🔧 Features

- Upload CSV files with reviews
- Preprocess reviews and classify sentiment
- Visualize sentiment distribution using bar charts
- Generate WordCloud for positive reviews
- Interactive web UI using Streamlit

---

## 📁 Dataset Description

The dataset used for this project (`sentiment_dataset.csv`) contains the following columns:

| Column Name         | Description                                 |
|---------------------|---------------------------------------------|
| `date`              | Date of the review                          |
| `title`             | Title of the review                         |
| `review`            | Full review text                            |
| `rating`            | Star rating given by the user               |
| `username`          | Name of the reviewer                        |
| `helpful_votes`     | Number of users who found the review helpful|
| `review_length`     | Length of the review                        |
| `platform`          | Platform used (iOS/Android/Web)             |
| `language`          | Language of the review                      |
| `location`          | Location of the reviewer                    |
| `version`           | App version at time of review               |
| `verified_purchase` | Whether the review is from a verified user  |

---

## 🧠 Machine Learning Pipeline

- **Model Used**: Logistic Regression
- **Text Vectorization**: TF-IDF
- **Preprocessing**:
  - Lowercasing
  - Punctuation removal
  - Stopwords removal
  - Whitespace normalization

---

## 📊 Exploratory Data Analysis

- Sentiment distribution (Bar chart)
- Word frequency analysis
- WordCloud generation for positive reviews

---

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
````

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app/app.py
```

### 4. Upload your `CSV` file and explore the results

---

## 📂 Project Structure

```
sentiment-analysis/
│
├── app/
│   ├── app.py              # Streamlit app script
│   └── templates/
│       └── index.html      # Basic HTML form for user input
│
├── models/
│   ├── sentiment_model.pkl # Trained Logistic Regression model
│   └── vectorizer.pkl      # Trained TF-IDF vectorizer
│
├── dataset/
│   └── sentiment_dataset.csv # Sample dataset
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ✅ Conclusion

This project demonstrates how machine learning and natural language processing can be used to build an effective sentiment analysis tool. The combination of a clean frontend (Streamlit), efficient backend (Logistic Regression with TF-IDF), and data visualization makes it useful for analyzing customer feedback at scale.

---

## 🌐 References

* [Streamlit Documentation](https://docs.streamlit.io/)
* [scikit-learn Documentation](https://scikit-learn.org/stable/)
* [WordCloud Python](https://amueller.github.io/word_cloud/)
* [Pandas Library](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)

---

```

You can copy this file directly as your `README.md` and update the GitHub link and any optional sections like licensing or authorship.
```
