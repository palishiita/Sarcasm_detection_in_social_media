from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def evaluate(y_test, y_pred, dataset_name="", model_name=""):
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{model_name} on {dataset_name}")
    print(classification_report(y_test, y_pred))
    return {
        "dataset":   dataset_name,
        "model":     model_name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1_macro":  f1_score(y_test, y_pred, average="macro"),
        "f1_sarcastic": report["1"]["f1-score"],
        "precision": report["1"]["precision"],
        "recall":    report["1"]["recall"],
    }

def confusion(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)