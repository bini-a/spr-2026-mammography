import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

#lemmatization
# spaCy for lemmatization
import spacy
nlp = spacy.blank("pt")
nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
nlp.initialize()

# NLTK built-in stopwords and stemmer
import nltk
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop_words = set(stopwords.words("portuguese"))
stemmer = SnowballStemmer("portuguese")
 
def preprocess_text(text, remove_stopwords=True, apply_stemming=True, apply_lemma=False):
    """
    Clean Portuguese mammography reports:
    - Lowercase everything
    - Remove carriage returns / newlines
    - Remove anonymization tokens like <DATA>
    - Remove punctuation (keep hyphens for medical terms)
    - Collapse whitespace
    - Optionally remove stopwords, apply stemming, and/or lemmatization
    """
    text = text.lower()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<[^>]+>", "", text)           # remove <DATA>, <D...> etc.
    
    #Normalize measurements (e.g., "2 cm" → "2cm")
    text = re.sub(r"(\d+)\s*cm", r"\1cm", text)
    text = re.sub(r"(\d+)\s*mm", r"\1mm", text)

    text = re.sub(r"[^\w\s\-]", " ", text)         # remove punctuation except hyphens


    #text = re.sub(r"\s+", " ", text).strip()        # collapse spaces

    #tokenize
    tokens = text.split()

    if remove_stopwords:
        #text = " ".join([word for word in tokens if word not in stop_words])
        tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatization (before stemming — lemma produces proper words, stem truncates)
    if apply_lemma:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]

    if apply_stemming:
        #text = " ".join([stemmer.stem(word) for word in tokens])
        tokens = [stemmer.stem(word) for word in tokens]
    
    #remove single character tokens (often noise after stemming)
    #text = " ".join([word for word in tokens if len(word) > 1])
    tokens = [word for word in tokens if len(word) > 1]

    return " ".join(tokens)


# Word-level TF-IDF: captures medical terms and bigrams
tfidf_word = TfidfVectorizer(
    max_features=15000,       # keep top 15k word features
    ngram_range=(1, 2),       # unigrams + bigrams (e.g., "calcificações benignas")
    min_df=3,                 # ignore very rare terms
    max_df=0.95,              # ignore terms in >95% of docs
    sublinear_tf=True,        # apply log(1 + tf) — dampens very frequent terms
)

# Character-level TF-IDF: captures morphological patterns in Portuguese
tfidf_char = TfidfVectorizer(
    analyzer="char_wb",       # character n-grams within word boundaries
    max_features=15000,
    ngram_range=(3, 5),       # 3 to 5 character sequences
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
)