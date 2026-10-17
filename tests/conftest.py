import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_app.agent import Agent  # noqa: E402
from agent_app.store import MemoryStore, load_docs  # noqa: E402
from agent_app.tools import build_tools  # noqa: E402

DIM = 64


def fake_embed(text: str) -> list[float]:
    """Hashed bag-of-words: deterministic, no Ollama needed."""
    vec = [0.0] * DIM
    for w in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        vec[sum(map(ord, w)) % DIM] += 1.0
    return vec


class ScriptedLLM:
    """Replays a fixed list of replies and records the messages it was shown."""

    def __init__(self, replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.seen = []

    def chat(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def tool(name, **args):
    return {"action": "tool", "tool": name, "args": args}


def final(answer):
    return {"action": "final", "answer": answer}


@pytest.fixture
def store():
    s = MemoryStore(fake_embed)
    s.add_documents(load_docs(str(Path(__file__).resolve().parent.parent / "sample_docs")))
    return s


@pytest.fixture
def make_agent(store):
    def _make(replies, **kw):
        llm = ScriptedLLM(replies)
        return Agent(llm, build_tools(store), **kw), llm
    return _make
