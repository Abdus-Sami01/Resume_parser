from fastapi import FastAPI

from app.api.routes import jobs, resumes, search

app = FastAPI(
    title="Resume Parser & Semantic Matching",
    description="Structured resume/JD extraction with hybrid retrieval + reranking.",
    version="0.1.0",
)

app.include_router(resumes.router)
app.include_router(jobs.router)
app.include_router(search.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
