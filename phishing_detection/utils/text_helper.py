import re
import string
import nltk
from nltk.tokenize import word_tokenize
# Ensure necessary NLTK resources are downloaded
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')    
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def clean_email_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', ' NUM ', text)  # Replace numbers
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    tokens = nltk.word_tokenize(text)  # Tokenize text
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(lemmatized)  # Join tokens back to string
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_sms_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', ' NUM ', text)  # Replace numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize text
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(lemmatized)  # Join tokens back to string
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
