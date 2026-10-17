"""FastAPI service: /ingest documents, then ask the guarded agent via /agent."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent
from .llm import OllamaLLM
from .store import ChromaStore, load_docs
from .tools import build_tools

app = FastAPI(title="Guarded RAG Agent")
_state: dict = {}


def get_agent() -> Agent:
    """Lazily build the agent so importing the app needs no Ollama/Chroma."""
    if "agent" not in _state:
        store = ChromaStore()
        _state["store"] = store
        _state["agent"] = Agent(OllamaLLM(), build_tools(store))
    return _state["agent"]


class IngestRequest(BaseModel):
    folder: str = "sample_docs"


class AgentRequest(BaseModel):
    question: str = Field(max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    get_agent()
    docs = load_docs(req.folder)
    if not docs:
        raise HTTPException(status_code=404, detail="No .md files found in that folder.")
    return {"chunks_indexed": _state["store"].add_documents(docs), "documents": [n for n, _ in docs]}


@app.post("/agent")
def ask(req: AgentRequest):
    result = get_agent().run(req.question)
    return {
        "answer": result.answer, "blocked": result.blocked, "block_code": result.block_code,
        "steps": result.steps, "notes": result.notes,
    }
