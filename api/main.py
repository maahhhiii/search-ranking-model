from pathlib import Path
import sys

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from the project root
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import create_demo_dataset
from src.ranker import SearchRanker


# Make sure the demo dataset exists
create_demo_dataset()


app = FastAPI(
    title="Search Ranking Model API",
    description="Learning-to-Rank search engine using LambdaMART.",
    version="1.0.0",
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "static"),
    name="static",
)


# Jinja2 templates
templates = Jinja2Templates(
    directory=PROJECT_ROOT / "templates"
)


_ranker = None


def get_ranker():
    global _ranker

    if _ranker is None:
        _ranker = SearchRanker()

    return _ranker


# -----------------------------------------
# Home page
# -----------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "results": [],
            "query": "",
        },
    )


# -----------------------------------------
# Search API
# -----------------------------------------

@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
):

    results = get_ranker().search(
        q,
        top_k=top_k
    )

    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


# -----------------------------------------
# Search UI
# -----------------------------------------

@app.get("/search-ui", response_class=HTMLResponse)
def search_ui(
    request: Request,
    q: str = "",
    top_k: int = 5,
):

    if q.strip():
        results = get_ranker().search(
            q,
            top_k=top_k
        )
    else:
        results = []

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "results": results,
            "query": q,
        },
    )