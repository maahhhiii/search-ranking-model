from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

COLUMNS = ["query_id", "query", "doc_id", "title", "content", "relevance"]


def create_demo_dataset(path: Path | None = None) -> Path:
    """
    Creates a small, deterministic learning-to-rank dataset so the project
    can be run immediately. Replace this CSV with a larger real dataset later.
    Relevance: 0 = irrelevant, 1 = somewhat relevant, 2 = relevant, 3 = highly relevant.
    """
    path = path or RAW_DIR / "search_dataset.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    topics = [
        ("python machine learning", [
            ("Python Machine Learning Tutorial", "Learn machine learning in Python using pandas, scikit-learn and practical examples.", 3),
            ("Introduction to Machine Learning", "A beginner guide to machine learning concepts, algorithms and model evaluation.", 2),
            ("Python Programming Guide", "Learn Python syntax, functions, classes and useful programming techniques.", 1),
            ("Web Development with HTML", "Build web pages using HTML, CSS and JavaScript.", 0),
            ("Database SQL Basics", "Learn SQL queries, tables, joins and relational databases.", 0),
        ]),
        ("deep learning", [
            ("Deep Learning with PyTorch", "Build neural networks and deep learning models using PyTorch.", 3),
            ("Neural Networks Explained", "An introduction to neural networks, backpropagation and deep learning.", 2),
            ("Machine Learning Basics", "Overview of supervised and unsupervised machine learning algorithms.", 1),
            ("Python Functions", "A guide to writing reusable functions in Python.", 0),
            ("SQL Database Tutorial", "Learn relational database design and SQL commands.", 0),
        ]),
        ("natural language processing", [
            ("NLP with Python", "Process text using tokenization, TF-IDF, named entity recognition and Python NLP libraries.", 3),
            ("Natural Language Processing Guide", "Learn NLP concepts including text classification, embeddings and language models.", 3),
            ("Machine Learning for Text", "Use machine learning algorithms to classify and analyze text documents.", 2),
            ("Computer Networks", "Learn TCP, UDP, HTTP and network communication.", 0),
            ("Operating Systems", "Processes, threads, memory management and scheduling.", 0),
        ]),
        ("fastapi python", [
            ("FastAPI Python Tutorial", "Create fast Python APIs with FastAPI, Pydantic and automatic OpenAPI documentation.", 3),
            ("Building REST APIs with Python", "Learn REST API development in Python with routing, validation and JSON.", 2),
            ("Python Backend Development", "Develop backend applications using Python frameworks and databases.", 2),
            ("Java Spring Boot API", "Build REST APIs using Java and Spring Boot.", 0),
            ("HTML CSS Web Design", "Design responsive websites with HTML and CSS.", 0),
        ]),
        ("sql database", [
            ("SQL Database Tutorial", "Learn SQL queries, joins, indexes, tables and relational database concepts.", 3),
            ("Database Management Systems", "Study database design, normalization, transactions and SQL.", 3),
            ("PostgreSQL Guide", "Use PostgreSQL for relational data storage and SQL queries.", 2),
            ("Python Programming", "Learn Python programming from beginner to advanced topics.", 0),
            ("Machine Learning Models", "Train classification and regression models with machine learning.", 0),
        ]),
        ("computer networks", [
            ("Computer Networks Fundamentals", "Learn TCP IP, routing, switching, HTTP and network protocols.", 3),
            ("Socket Programming in C", "Build TCP client server applications using sockets in C.", 3),
            ("Networking Protocols Guide", "Understand common computer networking protocols and communication.", 2),
            ("Database Systems", "Learn SQL, database normalization and transactions.", 0),
            ("Deep Learning", "Build neural networks with deep learning frameworks.", 0),
        ]),
        ("operating system", [
            ("Operating Systems Concepts", "Learn processes, threads, CPU scheduling, synchronization and memory management.", 3),
            ("Operating System Scheduling", "Study process scheduling, semaphores, deadlocks and synchronization.", 3),
            ("Linux System Programming", "Explore Linux processes, files, permissions and system calls.", 2),
            ("SQL Database Basics", "Learn SQL and relational database concepts.", 0),
            ("HTML Web Development", "Create websites with HTML and CSS.", 0),
        ]),
        ("data science pandas", [
            ("Pandas Data Analysis", "Use pandas to clean, transform and analyze tabular data in Python.", 3),
            ("Data Science with Python", "Learn NumPy, pandas, visualization and machine learning for data science.", 3),
            ("Python Data Analysis", "Analyze datasets using Python and pandas.", 2),
            ("Java Programming", "Learn Java object oriented programming and collections.", 0),
            ("Computer Networks", "Learn network protocols and communication.", 0),
        ]),
    ]

    rows = []
    doc_counter = 1
    for qid, (query, docs) in enumerate(topics, start=1):
        for title, content, relevance in docs:
            rows.append({
                "query_id": qid,
                "query": query,
                "doc_id": f"D{doc_counter:03d}",
                "title": title,
                "content": content,
                "relevance": relevance,
            })
            doc_counter += 1

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else RAW_DIR / "search_dataset.csv"
    if not path.exists():
        create_demo_dataset(path)

    df = pd.read_csv(path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df["query_id"] = df["query_id"].astype(str)
    df["relevance"] = df["relevance"].astype(int)
    df["query"] = df["query"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    return df


if __name__ == "__main__":
    path = create_demo_dataset()
    print(f"Dataset created: {path}")
    print(load_dataset().head(10).to_string(index=False))
