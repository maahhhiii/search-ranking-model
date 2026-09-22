from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset
from src.features import (
    build_features,
    save_vectorizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = MODEL_DIR / "ranking_model.txt"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

TRAIN_DATA_PATH = PROCESSED_DIR / "train_features.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test_features.csv"


def train():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Load and preprocess dataset
    # -------------------------------------------------

    df = preprocess_dataset(
        load_dataset()
    )

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Total queries: {df['query_id'].nunique()}"
    )

    # -------------------------------------------------
    # Split by query
    # -------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["query_id"]
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # -------------------------------------------------
    # Build features
    # -------------------------------------------------

    X_train, vectorizer = build_features(
        train_df,
        fit_vectorizer=True
    )

    X_test, _ = build_features(
        test_df,
        vectorizer=vectorizer,
        fit_vectorizer=False
    )

    y_train = train_df["relevance"]
    y_test = test_df["relevance"]

    # -------------------------------------------------
    # Ranking groups
    # -------------------------------------------------

    train_groups = (
        train_df
        .groupby("query_id", sort=False)
        .size()
        .tolist()
    )

    test_groups = (
        test_df
        .groupby("query_id", sort=False)
        .size()
        .tolist()
    )

    print(
        f"Train rows: {len(train_df)}"
    )

    print(
        f"Test rows: {len(test_df)}"
    )

    print(
        f"Train groups: {train_groups}"
    )

    print(
        f"Test groups: {test_groups}"
    )

    # -------------------------------------------------
    # LambdaMART
    # -------------------------------------------------
    #
    # Important:
    # min_child_samples is reduced because this
    # demonstration dataset is intentionally small.
    #

    model = lgb.LGBMRanker(

        objective="lambdarank",

        metric="ndcg",

        ndcg_at=[
            1,
            3,
            5,
            10
        ],

        learning_rate=0.05,

        n_estimators=200,

        num_leaves=15,

        max_depth=-1,

        min_child_samples=3,

        min_split_gain=0,

        reg_alpha=0.0,

        reg_lambda=0.0,

        random_state=42,

        verbosity=-1
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    model.fit(

        X_train,

        y_train,

        group=train_groups,

        eval_set=[
            (X_test, y_test)
        ],

        eval_group=[
            test_groups
        ],

        callbacks=[
            lgb.early_stopping(
                30,
                verbose=False
            )
        ]
    )

    # -------------------------------------------------
    # Save model
    # -------------------------------------------------

    model.booster_.save_model(
        str(MODEL_PATH)
    )

    save_vectorizer(
        vectorizer,
        VECTORIZER_PATH
    )

    # -------------------------------------------------
    # Save processed feature datasets
    # -------------------------------------------------

    train_output = train_df[
        [
            "query_id",
            "query",
            "doc_id",
            "title",
            "content",
            "relevance"
        ]
    ].copy()

    test_output = test_df[
        [
            "query_id",
            "query",
            "doc_id",
            "title",
            "content",
            "relevance"
        ]
    ].copy()

    train_features = pd.concat(
        [
            train_output.reset_index(drop=True),
            X_train.reset_index(drop=True)
        ],
        axis=1
    )

    test_features = pd.concat(
        [
            test_output.reset_index(drop=True),
            X_test.reset_index(drop=True)
        ],
        axis=1
    )

    train_features.to_csv(
        TRAIN_DATA_PATH,
        index=False
    )

    test_features.to_csv(
        TEST_DATA_PATH,
        index=False
    )

    # -------------------------------------------------
    # Show feature importance
    # -------------------------------------------------

    print("\nFeature importance:")

    importance = model.feature_importances_

    for name, value in zip(
        X_train.columns,
        importance
    ):
        print(
            f"{name}: {value}"
        )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        f"Vectorizer saved to: {VECTORIZER_PATH}"
    )

    print(
        "\nTraining complete."
    )


if __name__ == "__main__":
    train()