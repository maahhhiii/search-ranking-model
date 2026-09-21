from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset
from src.features import build_features, load_vectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ranking_model.txt"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"


def evaluate():
    test_path = PROJECT_ROOT / "data" / "processed" / "test_features.csv"
    if not test_path.exists():
        raise FileNotFoundError("Run `python -m src.train` first.")

    test = pd.read_csv(test_path)

    feature_names = [
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

    X = test[feature_names]
    model = lgb.Booster(model_file=str(MODEL_PATH))
    test["predicted_score"] = model.predict(X)

    scores = {}
    for k in [1, 3, 5, 10]:
        values = []
        for _, group in test.groupby("query_id"):
            true = group["relevance"].to_numpy(dtype=float)
            pred = group["predicted_score"].to_numpy(dtype=float)

            # ndcg_score expects shape (1, n_documents).
            values.append(ndcg_score([true], [pred], k=k))

        scores[f"NDCG@{k}"] = float(np.mean(values))

    print("\nRanking evaluation")
    print("------------------")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

    test.to_csv(PROJECT_ROOT / "data" / "processed" / "test_predictions.csv", index=False)
    return scores


if __name__ == "__main__":
    evaluate()
