from pathlib import Path

import lightgbm as lgb
import pandas as pd
import numpy as np

from src.features import build_features, load_vectorizer


class SearchRanker:
    """
    Search ranking system using LambdaMART.

    LambdaMART provides the primary ranking score.
    Feature-based relevance is used only to break ties
    and create a readable 0-100 display score.
    """

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[1]

        self.data_path = (
            self.project_root
            / "data"
            / "raw"
            / "search_dataset.csv"
        )

        self.model_path = (
            self.project_root
            / "models"
            / "ranking_model.txt"
        )

        self.vectorizer_path = (
            self.project_root
            / "models"
            / "tfidf_vectorizer.joblib"
        )

        # Load trained LambdaMART model
        self.model = lgb.Booster(
            model_file=str(self.model_path)
        )

        # Load trained TF-IDF vectorizer
        self.vectorizer = load_vectorizer(
            self.vectorizer_path
        )

        # Load dataset
        self.data = pd.read_csv(
            self.data_path
        )

        # Detect document column
        if "content" in self.data.columns:
            self.content_column = "content"

        elif "text" in self.data.columns:
            self.content_column = "text"

        elif "document_text" in self.data.columns:
            self.content_column = "document_text"

        else:
            raise ValueError(
                "No document content column found. "
                f"Available columns: {list(self.data.columns)}"
            )

    def search(self, query, top_k=5):

        query = str(query).strip()

        if not query:
            return []

        # -----------------------------------------
        # Prepare dataset
        # -----------------------------------------

        df = self.data.copy()

        df["query_clean"] = query

        df["document_text"] = (
            df[self.content_column]
            .fillna("")
            .astype(str)
        )

        df["title_clean"] = (
            df["title"]
            .fillna("")
            .astype(str)
        )

        # -----------------------------------------
        # Build features
        # -----------------------------------------

        features, _ = build_features(
            df,
            vectorizer=self.vectorizer,
            fit_vectorizer=False
        )

        # -----------------------------------------
        # LambdaMART prediction
        # -----------------------------------------

        lambda_scores = self.model.predict(
            features
        )

        df["lambda_score"] = lambda_scores

        # -----------------------------------------
        # Feature-based relevance signal
        # -----------------------------------------

        feature_score = (
            0.40 * features["tfidf_similarity"]
            +
            0.30 * features["title_similarity"]
            +
            0.15 * features["query_term_coverage"]
            +
            0.10 * features["title_term_coverage"]
            +
            0.05 * features["exact_phrase_match"]
        )

        df["feature_score"] = feature_score

        # -----------------------------------------
        # Primary ranking = LambdaMART
        # Secondary ranking = feature score
        # -----------------------------------------

        df = df.sort_values(
            by=[
                "lambda_score",
                "feature_score"
            ],
            ascending=[
                False,
                False
            ]
        )

        df = df.head(top_k).copy()

        # -----------------------------------------
        # Normalize display score
        # -----------------------------------------

        selected_lambda = (
            df["lambda_score"]
            .to_numpy(dtype=float)
        )

        selected_features = (
            df["feature_score"]
            .to_numpy(dtype=float)
        )

        # Normalize LambdaMART scores
        if (
            len(selected_lambda) > 1
            and selected_lambda.max() != selected_lambda.min()
        ):
            lambda_norm = (
                (selected_lambda - selected_lambda.min())
                /
                (
                    selected_lambda.max()
                    - selected_lambda.min()
                )
            )
        else:
            lambda_norm = np.ones(
                len(selected_lambda)
            )

        # Normalize feature score
        if (
            len(selected_features) > 1
            and selected_features.max()
            != selected_features.min()
        ):
            feature_norm = (
                (selected_features - selected_features.min())
                /
                (
                    selected_features.max()
                    - selected_features.min()
                )
            )
        else:
            feature_norm = selected_features

        # Combined display relevance
        display_scores = (
            0.70 * lambda_norm
            +
            0.30 * feature_norm
        ) * 100

        df["display_score"] = display_scores

        # -----------------------------------------
        # Build results
        # -----------------------------------------

        results = []

        for index, (_, row) in enumerate(
            df.iterrows()
        ):

            results.append(
                {
                    "doc_id": str(
                        row["doc_id"]
                    ),

                    "title": str(
                        row["title"]
                    ),

                    "content": str(
                        row[self.content_column]
                    ),

                    "text": str(
                        row[self.content_column]
                    ),

                    # Raw LambdaMART score
                    "score": float(
                        row["lambda_score"]
                    ),

                    # Readable relevance score
                    "relevance_score": float(
                        row["display_score"]
                    ),
                }
            )

        return results