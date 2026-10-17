# Guarded RAG Agent

A small, framework-free **tool-calling agent** over a local document collection, with explicit **guardrails**
around every step. Runs fully locally (Ollama + ChromaDB) and is served through a validated FastAPI endpoint.

I built the agent loop and guardrail layer by hand, rather than using CrewAI/LangChain, to understand
what those frameworks abstract: tool schemas, prompt/tool protocols, budgets, and failure handling.
The LLM is behind a one-method interface (`chat(messages) -> str`), so any model or framework can sit behind it.

## How it works

```
question -> input guardrails -> loop { LLM -> JSON action -> validate -> run tool -> observation } -> grounding check -> answer
```

Tools (each with a pydantic argument schema): `search_docs` (RAG over ChromaDB), `calculate` (AST-based safe
arithmetic, no `eval`), `list_docs`.

| Guardrail | What it does |
|---|---|
| Input screening | Blocks empty/oversized input and prompt-injection phrasing; redacts emails and phone numbers |
| Tool allowlist + schema validation | Unknown tools and bad arguments are rejected and fed back to the model |
| Step budget + loop detection | Caps steps; a repeated identical call is rejected once, then the run is stopped |
| Indirect-injection filtering | Strips instruction-like sentences from retrieved text, keeping the legitimate rest |
| Evidence-first | The model must search before answering; one nudge, then the answer is withheld |
| Grounding check | Answer must cite a document retrieved in this run **and** only state numbers present in the evidence |
| Safe "not found" path | `found: false` is only accepted after a search, and the model's text is never echoed on it |

Every response includes the step trace and guardrail notes.

## Run it

```bash
pip install -r requirements.txt
ollama pull llama3.2:1b && ollama pull nomic-embed-text
uvicorn agent_app.main:app
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"folder":"sample_docs"}'
curl -X POST localhost:8000/agent  -H 'content-type: application/json' -d '{"question":"How many days of annual leave do I get?"}'
```

`sample_docs/vendor_notes.md` deliberately contains an injected instruction ("ignore all previous
instructions...") to demonstrate the indirect-injection filter.

## Tests and evaluation

* **26 offline tests** (`pytest`): a scripted fake LLM and in-memory store exercise every guardrail and the API, with no
  Ollama needed. They also run in GitHub Actions CI.
* **Live eval** (`python eval/run_eval.py`, 11 questions, `llama3.2:1b`, temperature 0): **6/11 pass**.

### Honest findings

* The 1B model is the bottleneck. Failures are wrong or missing tool use, not guardrail bugs: it sometimes
  skips searching, repeats a search, invents a number, or answers `False` to a yes/no question.
* The guardrails turn most bad generations into safe refusals instead of wrong answers: invented numbers
  (`unsupported_numbers`), made-up citations and un-searched answers are withheld. The CEO-salary question,
  which has no answer in the docs, is correctly refused.
* **Known weakness:** the grounding check verifies citations and numbers, not meaning. A wrong but
  well-cited non-numeric answer (e.g. the `False` above) still passes. Fixing that needs an entailment/LLM-judge check.
* Bugs found by running the eval and fixed along the way: a "not found" shortcut that bypassed grounding, chunk-level
  injection filtering that discarded legitimate content, and a hard abort on the first repeated call.
* `qwen2.5-coder:1.5b` scored 2/11 with the same prompt (it answered "not found" for everything).

## Next steps

Try a larger local model, add an LLM-judge entailment check, and a framework adapter (CrewAI/LangChain) behind the same interface.
