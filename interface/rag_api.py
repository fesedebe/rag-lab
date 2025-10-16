# FastAPI backend for RAG querying

from fastapi import FastAPI
from pydantic import BaseModel
from core.rag_baseline import build_rag

app = FastAPI(title="Local RAG API", version="1.0")
rag_chain = build_rag() # Initialize RAG pipeline

class QueryRequest(BaseModel):
    query: str

# API endpoint to handle queries
@app.post("/query")
def query_rag(request: QueryRequest):
    result = rag_chain({"query": request.query})
    return {
        "answer": result["result"],
        "sources": [
            {"source": doc.metadata.get("source"), "content": doc.page_content[:300]}
            for doc in result.get("source_documents", [])
        ]
    }
