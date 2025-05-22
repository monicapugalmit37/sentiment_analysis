import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(path):
    df = pd.read_csv(path)
    
    print("Shape of dataset:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Null values:\n", df.isnull().sum())
    print("Ratings distribution:\n", df['rating'].value_counts())

    # Review length distribution
    df['review_length'] = df['review'].astype(str).apply(len)
    sns.histplot(df['review_length'], bins=50, kde=True)
    plt.title("Distribution of Review Lengths")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()

    # Rating vs sentiment
    df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
    sns.countplot(x='sentiment', data=df)
    plt.title("Sentiment Distribution Based on Rating")
    plt.show()

if __name__ == "__main__":
    perform_eda(r"C:\Users\MONICA PUGAZHENDHI\Desktop\sentimental_analysis\data\sentiment_dataset.csv")
