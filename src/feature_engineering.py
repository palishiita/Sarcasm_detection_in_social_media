import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_vectorize(train_texts, test_texts, ngram_range=(1, 2), max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def save_vectorizer(vectorizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectoriser saved: {path}")


def load_vectorizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_top_features(vectorizer, n=20):
    return vectorizer.get_feature_names_out()[:n]