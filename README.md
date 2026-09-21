# Search Ranking Model

A complete Learning-to-Rank search engine built with Python, TF-IDF features, LambdaMART/LightGBM, NDCG evaluation, and FastAPI.

## Project pipeline

```text
Search Query
     |
     v
Candidate Documents
     |
     v
Feature Engineering
     |
     v
LambdaMART
     |
     v
Predicted Relevance Score
     |
     v
Ranked Results
```

## Features

The model uses:

- TF-IDF query-document similarity
- TF-IDF query-title similarity
- Query term coverage
- Title term coverage
- Exact phrase match
- Query length
- Document length
- Title length
- Query/document length ratio

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Or Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Recommended Python version: 3.11.

## 3. Generate the starter dataset

```bash
python -m src.data_loader
```

This creates:

```text
data/raw/search_dataset.csv
```

The included dataset is a small demo dataset so the entire project can run immediately.

For a serious portfolio experiment, replace it with a larger labeled search dataset while keeping the same columns:

```text
query_id,query,doc_id,title,content,relevance
```

Relevance should be an ordinal label such as 0, 1, 2, 3.

## 4. Train LambdaMART

```bash
python -m src.train
```

This creates:

```text
models/ranking_model.txt
models/tfidf_vectorizer.joblib
data/processed/train_features.csv
data/processed/test_features.csv
```

Important: the train/test split is performed by `query_id`, preventing documents from the same query from leaking into both sets.

## 5. Evaluate

```bash
python -m src.evaluate
```

You will get:

```text
Ranking evaluation
------------------
NDCG@1: ...
NDCG@3: ...
NDCG@5: ...
NDCG@10: ...
```

The numbers are calculated from the actual trained model and test set.

## 6. Run terminal search

```bash
python -m src.search
```

Try:

```text
machine learning python
deep learning
natural language processing
fastapi python
sql database
computer networks
```

Type `exit` to stop.

## 7. Run the FastAPI application

```bash
uvicorn api.main:app --reload
```

Open the application in your browser at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Search API example:

```text
http://127.0.0.1:8000/search?q=machine%20learning%20python
```

## 8. How the model works

LambdaMART is trained with groups. Every `query_id` represents one ranking problem:

```text
Query 1
  document A -> relevance 3
  document B -> relevance 2
  document C -> relevance 0

Query 2
  document D -> relevance 3
  document E -> relevance 1
  document F -> relevance 0
```

The model learns which feature patterns correspond to higher relevance.

At inference time:

```text
query
  -> feature extraction
  -> LambdaMART score
  -> sort descending
  -> top K results
```

## 9. Portfolio upgrade

After the demo works, replace the small CSV with a larger labeled dataset such as a public Learning-to-Rank dataset. Keep the same train/evaluation pipeline, document the dataset size, and report your actual NDCG improvement over a baseline.

Do not put an invented accuracy or improvement percentage on your resume. Report the value measured by your experiment.

## Resume bullet template

After running the final experiment, you can use:

- Built a Learning-to-Rank search engine using LambdaMART and engineered query-document relevance features for ranking candidate documents.
- Evaluated ranking quality using NDCG@1/3/5/10 and compared the ML ranker against a TF-IDF baseline.
- Deployed the trained ranking model through a FastAPI search API with an interactive web interface.

Replace generic wording with your actual measured results.
