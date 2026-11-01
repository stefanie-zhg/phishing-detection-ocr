import joblib
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path
from ocr import extract_text_from_image
from text_model import predict_text, load_model
from evaluation import evaluate_model, explain_verdict, identify_misclassifications

try:
    BASE_PATH = Path(__file__).resolve()
except NameError:
    BASE_PATH = Path.cwd()

DATA_DIR = BASE_PATH.parent.parent / "processed_data"
LOG_FILE = BASE_PATH.parent.parent / "data" / "prediction_log.csv"
ARCHIVE_FILE = BASE_PATH.parent.parent / "data" / "prediction_log_used.csv"


def log_prediction(text, predicted, confidence):
    log = {
        "text": text,
        "predicted": predicted,
        "confidence": f"{confidence * 100:.2f}%",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "correct_label": "" 
    }

    df = pd.DataFrame([log])
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
        
def process_image_and_predict(image=None, text=None):
    model, vectorizer = load_model()

    # Extract text
    if image:
        extracted_text = extract_text_from_image(image)
    elif text:
        extracted_text = text
    else:
        raise ValueError("Either an image or text must be provided.")

    # Vectorize and predict
    X_vec = vectorizer.transform([extracted_text])

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_vec)[:, 1]
        threshold = 0.5
        y_pred = (y_proba >= threshold).astype(int)
        prediction_label = "phishing" if y_pred[0] == 1 else "not phishing"
        explanation = explain_verdict(y_proba, y_pred)[0]
        confidence = y_proba[0]
    else:
        y_pred = model.predict(X_vec)
        prediction_label = "phishing" if y_pred[0] == 1 else "not phishing"
        explanation = "Confidence score not available for this model."
        confidence = 1.0

    # Log prediction
    log_prediction(extracted_text, y_pred[0], confidence)

    return extracted_text, prediction_label, explanation


def retrain_from_flagged():

    if not log_file.exists():
        print("No flagged prediction log found.")
        return

    df = pd.read_csv(log_file)

    # Keep only rows with proper corrections
    df_flagged = df[df["correct_label"].isin([0, 1])]
    if df_flagged.empty:
        print("No valid flagged corrections found.")
        return

    X = df_flagged["text"].fillna("")
    y = df_flagged["correct_label"].astype(int)

    # Load model and vectorizer
    vectorizer = joblib.load(data_dir / "vectorizer.joblib")
    model = joblib.load(data_dir / "model.joblib")

    X_vec = vectorizer.transform(X)

    # Retrain the model and save it
    model.fit(X_vec, y)
    joblib.dump(model, data_dir / "model.joblib")
    print(f"Retrained model on {len(X)} flagged samples.")

    # Archive used rows
    if archive_file.exists():
        df_flagged.to_csv(archive_file, mode='a', header=False, index=False)
    else:
        df_flagged.to_csv(archive_file, index=False)

    # Keep only unflagged entries in the current log
    df_remaining = df[~df.index.isin(df_flagged.index)]
    df_remaining.to_csv(log_file, index=False)


metrics, y_true, y_pred, y_proba, X_test = evaluate_model()

    
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "retrain":
        retrain_from_flagged()


misclassified = identify_misclassifications(y_true, y_pred, X_test)

if not misclassified:
    print("No misclassifications found!")
else:
    for idx, text, true, pred in misclassified[:10]: 
        print(f"[{idx}] Label: {true}, Predicted: {pred} — \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
