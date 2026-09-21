import re
import pandas as pd


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["query_clean"] = out["query"].map(normalize_text)
    out["title_clean"] = out["title"].map(normalize_text)
    out["content_clean"] = out["content"].map(normalize_text)
    out["document_text"] = (
        out["title_clean"] + " " + out["content_clean"]
    ).str.strip()
    return out
