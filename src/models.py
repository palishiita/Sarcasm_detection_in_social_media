from sklearn.svm import SVC
import pickle
import os
from sklearn.svm import SVC, LinearSVC

def train_svm(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

# LinearSVC is much faster than SVC on large data
# def train_svm(X_train, y_train, kernel="linear"):
#     model = LinearSVC(max_iter=2000)
#     model.fit(X_train, y_train)
#     return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {path}")

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)