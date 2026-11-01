import pandas as pd
import joblib
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE

# Set base path
try:
    BASE_PATH = Path(__file__).resolve()
except NameError:
    BASE_PATH = Path.cwd()

def train_model():
    data_dir = BASE_PATH.parent.parent / "processed_data"

    # Load preprocessed text and labels
    X_train = pd.read_csv(data_dir / "X_train.csv")['text']
    y_train = pd.read_csv(data_dir / "y_train.csv")['label']

    # Handle missing text values
    X_train = X_train.fillna('')

    # Load pre-fitted vectorizer (from preprocessing)
    vectorizer = joblib.load(data_dir / "vectorizer.joblib")

    # Transform training text to vector form
    X_train_vec = vectorizer.transform(X_train)

    # Apply SMOTE to balance dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train)

    # Train the Naive Bayes model
    model = MultinomialNB().fit(X_resampled, y_resampled)

    # Save the trained model
    joblib.dump(model, data_dir / "model.joblib")
    print("Model trained and saved successfully.")

def predict_text(texts, model, vectorizer):
    return model.predict(vectorizer.transform(texts))

def load_model():
    data_dir = BASE_PATH.parent.parent / "processed_data"
    model = joblib.load(data_dir / "model.joblib")
    vectorizer = joblib.load(data_dir / "vectorizer.joblib")
    print("Load successful...")
    return model, vectorizer

if __name__ == "__main__":
    train_model()
