from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer