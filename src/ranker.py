from pathlib import Path
import lightgbm as lgb
import pandas as pd

from src.preprocessing import preprocess_dataset
from src.features import build_features, load_vectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ranking_model.txt"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "search_dataset.csv"


class SearchRanker:
    def __init__(self):
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "Model files not found. Run `python -m src.train` first."
            )

        self.model = lgb.Booster(model_file=str(MODEL_PATH))
        self.vectorizer = load_vectorizer(VECTORIZER_PATH)
        self.documents = pd.read_csv(DATASET_PATH)
        self.documents = preprocess_dataset(self.documents)

    def search(self, query: str, top_k: int = 5):
        query = query.strip()
        if not query:
            return []

        candidates = self.documents.copy()
        candidates["query"] = query
        candidates["query_clean"] = query.lower()
        candidates = preprocess_dataset(candidates)

        X, _ = build_features(
            candidates,
            vectorizer=self.vectorizer,
            fit_vectorizer=False,
        )

        candidates["score"] = self.model.predict(X)
        candidates = candidates.sort_values("score", ascending=False).head(top_k)

        results = []
        for _, row in candidates.iterrows():
            results.append({
                "doc_id": row["doc_id"],
                "title": row["title"],
                "content": row["content"],
                "score": round(float(row["score"]), 4),
                "relevance": int(row["relevance"]) if "relevance" in row else None,
            })

        return results
