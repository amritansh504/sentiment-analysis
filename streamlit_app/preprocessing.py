import re
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    doc = nlp(text)
    cleaned = ' '.join([
        token.lemma_ for token in doc
        if token.lemma_ not in stop_words and token.is_alpha
    ])
    
    return cleaned