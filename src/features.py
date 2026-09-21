from pathlib import Path
import math
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FEATURE_NAMES = [
    "tfidf_similarity",
    "title_similarity",
    "query_term_coverage",
    "title_term_coverage",
    "exact_phrase_match",
    "query_length",
    "document_length",
    "title_length",
    "query_document_length_ratio",
]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _coverage(query: str, document: str) -> float:
    q = set(_tokens(query))
    d = set(_tokens(document))
    return len(q & d) / len(q) if q else 0.0


def build_features(df: pd.DataFrame, vectorizer=None, fit_vectorizer=False):
    """
    Builds query-document features used by LambdaMART.
    Returns (features_dataframe, fitted_vectorizer).
    """
    data = df.copy()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

    corpus = pd.concat(
        [data["query_clean"], data["document_text"]], ignore_index=True
    )

    if fit_vectorizer:
        vectorizer.fit(corpus)

    q_matrix = vectorizer.transform(data["query_clean"])
    d_matrix = vectorizer.transform(data["document_text"])
    tfidf_sim = cosine_similarity(q_matrix, d_matrix).diagonal()

    title_matrix = vectorizer.transform(data["title_clean"])
    title_sim = cosine_similarity(q_matrix, title_matrix).diagonal()

    rows = []
    for i, row in data.reset_index(drop=True).iterrows():
        query = row["query_clean"]
        title = row["title_clean"]
        document = row["document_text"]

        q_len = len(_tokens(query))
        d_len = len(_tokens(document))
        title_len = len(_tokens(title))

        rows.append({
            "tfidf_similarity": float(tfidf_sim[i]),
            "title_similarity": float(title_sim[i]),
            "query_term_coverage": _coverage(query, document),
            "title_term_coverage": _coverage(query, title),
            "exact_phrase_match": float(query in document and bool(query)),
            "query_length": q_len,
            "document_length": d_len,
            "title_length": title_len,
            "query_document_length_ratio": q_len / max(d_len, 1),
        })

    features = pd.DataFrame(rows, index=data.index)
    return features[FEATURE_NAMES], vectorizer


def save_vectorizer(vectorizer, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str | Path):
    return joblib.load(path)
