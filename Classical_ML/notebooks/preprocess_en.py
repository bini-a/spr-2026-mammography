import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator

#lemmatization
import spacy
nlp = spacy.blank("en")
nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
nlp.initialize()

# NLTK built-in stopwords and stemmer
import nltk
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

from tqdm import tqdm
tqdm.pandas()



def translate_portuguese(text):
    translator = GoogleTranslator(source='pt', target='en')
    #text = translator.translate(text)

    return translator.translate(text)

def preprocess_text(text, remove_stops=True, stem=True, apply_lemma=False):
    """Full preprocessing for English mammography reports."""
    text = str(text).lower().replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(\d+)\s*cm\b", r"\1cm", text)
    text = re.sub(r"(\d+)\s*mm\b", r"\1mm", text)
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    if remove_stops:
        tokens = [t for t in tokens if t not in stop_words]
    if apply_lemma:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]

    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)

# Word-level TF-IDF: captures medical terms and bigrams
tfidf_word = TfidfVectorizer(
    max_features=15000,       # keep top 15k word features
    ngram_range=(1, 2),       # unigrams + bigrams (e.g., "calcificações benignas")
    min_df=3,                 # ignore very rare terms
    max_df=0.95,              # ignore terms in >95% of docs
    sublinear_tf=True,        # apply log(1 + tf) — dampens very frequent terms
)

# Character-level TF-IDF: captures morphological patterns in English
tfidf_char = TfidfVectorizer(
    analyzer="char_wb",       # character n-grams within word boundaries
    max_features=15000,
    ngram_range=(3, 5),       # 3 to 5 character sequences
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
)




# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")
# print(f"  Train: {train.shape[0]:,} samples, Test: {test.shape[0]} samples")

# # Translate 

# print(f"\n  Translating {len(train):,} train reports...")
# train["report_en"] = train["report"].progress_apply(translate_portuguese)
# print(f"  Translating {len(test)} test reports...")
# test["report_en"] = test["report"].progress_apply(translate_portuguese)



# # Save translated datasets
# train.to_csv("train_translated.csv", index=False)
# test.to_csv("test_translated.csv", index=False)