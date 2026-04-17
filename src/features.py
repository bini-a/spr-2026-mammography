"""
Text preprocessing and TF-IDF feature extraction for Portuguese medical reports.
"""
import pickle
import re
import os
from typing import List, Tuple, Optional

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """
    Minimal cleaning that preserves medically meaningful tokens.
    - Keeps <DATA> placeholder (correlates with comparative exams → BI-RADS 2/3)
    - Normalises whitespace; lowercases
    - Does NOT strip accents — Portuguese accents carry lexical meaning
    """
    if not isinstance(text, str):
        text = str(text)
    # Collapse newlines / tabs to a single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    text = text.lower()
    return text


def _make_vectorizer(cfg: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=cfg.get("analyzer", "word"),
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        max_features=cfg.get("max_features", 150_000),
        sublinear_tf=cfg.get("sublinear_tf", True),
        min_df=cfg.get("min_df", 2),
        strip_accents=None,  # preserve Portuguese characters
    )


def build_features(
    texts_train: List[str],
    texts_val: Optional[List[str]] = None,
    config: Optional[dict] = None,
) -> Tuple:
    """
    Fit TF-IDF (word + char) on texts_train.
    Returns (X_train, vectorizers) or (X_train, X_val, vectorizers).
    """
    cfg = config or {}
    word_cfg = cfg.get("word_tfidf", {})
    char_cfg  = cfg.get("char_tfidf", {})

    # Default char config if not provided
    if "analyzer" not in char_cfg:
        char_cfg = {"analyzer": "char_wb", "ngram_range": [2, 5],
                    "max_features": 150_000, "sublinear_tf": True, "min_df": 3}

    word_vec = _make_vectorizer(word_cfg)
    char_vec  = _make_vectorizer(char_cfg)

    X_word_train = word_vec.fit_transform(texts_train)
    X_char_train  = char_vec.fit_transform(texts_train)
    X_train = hstack([X_word_train, X_char_train], format="csr")

    vectorizers = (word_vec, char_vec)

    if texts_val is not None:
        X_word_val = word_vec.transform(texts_val)
        X_char_val  = char_vec.transform(texts_val)
        X_val = hstack([X_word_val, X_char_val], format="csr")
        return X_train, X_val, vectorizers

    return X_train, vectorizers


def transform_features(texts: List[str], vectorizers: Tuple) -> csr_matrix:
    word_vec, char_vec = vectorizers
    X_word = word_vec.transform(texts)
    X_char  = char_vec.transform(texts)
    return hstack([X_word, X_char], format="csr")


def save_vectorizers(vectorizers: Tuple, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vectorizers.pkl"), "wb") as f:
        pickle.dump(vectorizers, f)


def load_vectorizers(out_dir: str) -> Tuple:
    with open(os.path.join(out_dir, "vectorizers.pkl"), "rb") as f:
        return pickle.load(f)
