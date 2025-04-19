import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import joblib

from .preprocess import clean_text


def main():
    # Load dataset
    df = pd.read_csv('data/IMDB Dataset.csv')
    df['clean_review'] = df['review'].apply(clean_text)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    preds = model.predict(X_test_vec)
    print('Accuracy:', accuracy_score(y_test, preds))
    print('F1 Score:', f1_score(y_test, preds, pos_label='positive'))

    # Save artifacts
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')


if __name__ == '__main__':
    main()