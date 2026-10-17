"""Guardrails around the agent loop: input screening, tool-call validation,
loop/step budgets, untrusted-content filtering and output grounding."""
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

MAX_QUESTION_CHARS = 2000

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+.{0,40}(instructions|rules|guidelines)",
    r"(reveal|show|print|repeat)\s+(your\s+|the\s+)?(system\s+)?(prompt|instructions)",
    r"you\s+are\s+now\s+(a|an|in)\b",
    r"override\s+(your\s+)?(safety|rules|guardrails)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)")


class GuardrailViolation(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def redact_pii(text: str) -> tuple[str, int]:
    text, n_email = _EMAIL_RE.subn("[REDACTED_EMAIL]", text)
    text, n_phone = _PHONE_RE.subn("[REDACTED_PHONE]", text)
    return text, n_email + n_phone


def check_input(question: str) -> tuple[str, list[str]]:
    """Return (cleaned question, notes). Raises GuardrailViolation to block."""
    question = question.strip()
    if not question:
        raise GuardrailViolation("empty_input", "Question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise GuardrailViolation("input_too_long", f"Question exceeds {MAX_QUESTION_CHARS} characters.")
    if looks_like_injection(question):
        raise GuardrailViolation("prompt_injection", "Request looks like a prompt-injection attempt and was blocked.")
    cleaned, redacted = redact_pii(question)
    return cleaned, ([f"redacted {redacted} PII item(s) from the question"] if redacted else [])


def filter_untrusted_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """Strip sentences that try to instruct the model (indirect prompt injection).

    Only the offending sentences are removed, so the legitimate content of a
    partly-poisoned chunk stays usable. Returns (clean chunks, sentences removed).
    """
    clean, removed = [], 0
    for c in chunks:
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", c["text"]):
            if looks_like_injection(sentence):
                removed += 1
            elif sentence.strip():
                kept.append(sentence.strip())
        if kept:
            clean.append({**c, "text": " ".join(kept)})
    return clean, removed


def validate_tool_call(tools: dict, name: str, args: dict) -> BaseModel:
    """Allowlist the tool name and validate arguments against its pydantic schema."""
    if name not in tools:
        raise GuardrailViolation("unknown_tool", f"Tool '{name}' is not allowed. Allowed: {sorted(tools)}.")
    try:
        return tools[name].args_model(**args)
    except (ValidationError, TypeError) as exc:
        raise GuardrailViolation("bad_arguments", f"Invalid arguments for '{name}': {exc}") from exc


@dataclass
class Budget:
    """Caps the number of steps and detects the agent repeating the same call."""
    max_steps: int = 6
    steps: int = 0
    seen_calls: set = field(default_factory=set)
    repeat_strikes: int = 0

    def next_step(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise GuardrailViolation("max_steps", f"Stopped after {self.max_steps} steps without a final answer.")

    def register_call(self, name: str, args: dict) -> None:
        key = (name, repr(sorted(args.items())))
        if key in self.seen_calls:
            self.repeat_strikes += 1
            raise GuardrailViolation("repeated_call", "Identical tool call repeated; stopping to avoid a loop.")
        self.seen_calls.add(key)


_CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)


def cited_sources(answer: str) -> set[str]:
    return {m.strip() for m in _CITATION_RE.findall(answer)}


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER_RE.findall(text)}


def check_grounding(answer: str, retrieved_sources: set[str], used_calculator: bool,
                    evidence: str = "") -> None:
    """Final answers must (1) cite a document retrieved in this run and (2) only state
    numbers that appear in the retrieved text, calculator output or the question."""
    cites = cited_sources(answer)
    if not (cites & retrieved_sources) and not (used_calculator and not retrieved_sources):
        raise GuardrailViolation(
            "ungrounded_answer",
            "Answer did not cite any document retrieved in this run, so it was withheld.",
        )
    body = _CITATION_RE.sub("", answer)
    unsupported = _numbers(body) - _numbers(evidence)
    if unsupported:
        raise GuardrailViolation(
            "unsupported_numbers",
            f"Answer states number(s) {sorted(unsupported)} not found in the retrieved evidence.",
        )
