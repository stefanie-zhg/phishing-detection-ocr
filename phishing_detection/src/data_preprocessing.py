import kagglehub
import os
import pandas as pd
import sys
from pathlib import Path
utils_path = Path(__file__).resolve().parents[1] / "utils"
sys.path.append(str(utils_path))
from text_helper import clean_email_text, clean_sms_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#Download and Load Email Dataset
email_path = kagglehub.dataset_download("bayes2003/emails-for-spam-or-ham-classification-spamassassin")
print("Email dataset downloaded to:", email_path)

email_csv = os.path.join(email_path, "email_origin.csv")
df_email = pd.read_csv(email_csv, encoding='latin1')
df_email = df_email[['label', 'origin']].rename(columns={'origin': 'text'})
df_email['label_num'] = df_email['label'].astype(int)
df_email = df_email[['text', 'label_num']].dropna().drop_duplicates()
df_email['text'] = df_email['text'].apply(clean_email_text)
print(f"Loaded email dataset: {len(df_email)} rows")

#Download and Load SMS Dataset
sms_path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("SMS dataset downloaded to:", sms_path)

sms_csv = os.path.join(sms_path, "spam.csv")
df_sms = pd.read_csv(sms_csv, encoding='latin1')[['v1', 'v2']]
df_sms.columns = ['label', 'text']
df_sms['label_num'] = df_sms['label'].map({'ham': 0, 'spam': 1})
df_sms = df_sms[['text', 'label_num']].dropna().drop_duplicates() #preprocess text (functions in text_helper.py) 
df_sms['text'] = df_sms['text'].apply(clean_sms_text)
print(f"Loaded SMS dataset: {len(df_sms)} rows")

#Combine Both Datasets
df_combined = pd.concat([df_email, df_sms], ignore_index=True).dropna().drop_duplicates()
print(f"Combined dataset: {len(df_combined)} rows")

#Train-Test Split
X = df_combined['text']
y = df_combined['label_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train label distribution:\n", y_train.value_counts())
print("Test label distribution:\n", y_test.value_counts())

#TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Save Preprocessed Files
# Define path to save preprocessed files one level above this script
output_dir = Path(__file__).resolve().parent.parent / "processed_data"
output_dir.mkdir(parents=True, exist_ok=True)

#Save files using that path
X_train.to_frame(name='text').to_csv(output_dir / "X_train.csv", index=False)
X_test.to_frame(name='text').to_csv(output_dir / "X_test.csv", index=False)
y_train.to_frame(name='label').to_csv(output_dir / "y_train.csv", index=False)
y_test.to_frame(name='label').to_csv(output_dir / "y_test.csv", index=False)
joblib.dump(vectorizer, output_dir / "vectorizer.joblib")

print(f"Preprocessing complete. Files saved to {output_dir}")

