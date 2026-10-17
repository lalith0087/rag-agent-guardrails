"""Document stores: Chroma (Ollama embeddings) for real use, in-memory for tests."""
import math
import re
from pathlib import Path


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Paragraph-aware chunking: pack whole paragraphs up to max_chars."""
    chunks, current = [], ""
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_docs(folder: str) -> list[tuple[str, str]]:
    return [(p.name, p.read_text()) for p in sorted(Path(folder).glob("*.md"))]


class MemoryStore:
    """Cosine-similarity store with a pluggable embedding function."""

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.rows: list[dict] = []

    def add_documents(self, docs: list[tuple[str, str]]) -> int:
        n = 0
        for name, text in docs:
            for chunk in chunk_text(text):
                self.rows.append({"source": name, "text": chunk, "vec": self.embed_fn(chunk)})
                n += 1
        return n

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.embed_fn(query)

        def cos(a, b):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0

        scored = sorted(self.rows, key=lambda r: cos(q, r["vec"]), reverse=True)[:k]
        return [{"source": r["source"], "text": r["text"], "score": round(cos(q, r["vec"]), 3)} for r in scored]

    def sources(self) -> list[str]:
        return sorted({r["source"] for r in self.rows})


class ChromaStore:
    def __init__(self, path: str = "./chroma_db", name: str = "docs", embed_model: str = "nomic-embed-text"):
        import chromadb
        self.embed_model = embed_model
        self.client = chromadb.PersistentClient(path=path)
        self.name = name
        self.col = self.client.get_or_create_collection(name)

    def _embed(self, text: str) -> list[float]:
        import ollama
        return ollama.embeddings(model=self.embed_model, prompt=text)["embedding"]

    def add_documents(self, docs: list[tuple[str, str]], reset: bool = True) -> int:
        if reset:
            try:
                self.client.delete_collection(self.name)
            except Exception:
                pass
            self.col = self.client.get_or_create_collection(self.name)
        ids, texts, metas = [], [], []
        for name, text in docs:
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{name}-{i}")
                texts.append(chunk)
                metas.append({"source": name})
        if not ids:
            return 0
        self.col.add(ids=ids, documents=texts, metadatas=metas, embeddings=[self._embed(t) for t in texts])
        return len(ids)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(query_embeddings=[self._embed(query)], n_results=min(k, self.col.count()))
        return [
            {"source": m["source"], "text": d, "score": round(1 - dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def sources(self) -> list[str]:
        return sorted({m["source"] for m in self.col.get()["metadatas"]})
