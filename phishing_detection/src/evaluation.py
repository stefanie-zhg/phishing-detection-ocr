import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model():
    data_dir = Path(__file__).resolve().parent.parent / "processed_data"
    try:
        model = joblib.load(data_dir / "model.joblib")
        vectorizer = joblib.load(data_dir / "vectorizer.joblib")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None, None, None

    try:
        X_test = pd.read_csv(data_dir / "X_test.csv")['text']
        y_test = pd.read_csv(data_dir / "y_test.csv")['label']
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return None, None, None, None, None

    X_test = X_test.fillna('')
    X_test_vec = vectorizer.transform(X_test)
    threshold = 0.6 #higher threshold, less false positive, only very confident phishing texts are flagged
    y_proba = model.predict_proba(X_test_vec)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = (y_proba >= threshold).astype(int) if y_proba is not None else model.predict(X_test_vec)


    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "report": classification_report(y_test, y_pred, zero_division=0)
    }

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        if k != "report":
            print(f"{k.capitalize()}: {v:.2f}")
    print("\nClassification Report:")
    print(metrics["report"])
    return metrics, y_test, y_pred, y_proba, X_test


def identify_misclassifications(y_true, y_pred, test_texts):
    if y_true is None or y_pred is None or test_texts is None:
        return []

    misclassified = [
        (i, test_texts.iloc[i], true, pred)
        for i, (true, pred) in enumerate(zip(y_true, y_pred))
        if true != pred
    ]

    total = len(y_true)
    print(f"Total misclassifications: {len(misclassified)} out of {total} samples")

    return misclassified


def explain_verdict(y_proba, y_pred):
    if y_proba is None or y_pred is None:
        return ["Confidence scores not available for this model"]

    verdicts = []
    for conf, pred in zip(y_proba, y_pred):
        if pred == 1:
            verdicts.append(f"The model is {conf* 100:.2f}% confident that this message is a phishing message.\n\n️❗️Do not interact❗")
        else:
            verdicts.append(f"The model is {(1 - conf) * 100:.2f}% confident that this message is not a phishing message.")
    return verdicts


