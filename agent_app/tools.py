"""Tools the agent may call. Each tool declares a pydantic schema for its arguments."""
import ast
import operator
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field

from .guardrails import filter_untrusted_chunks


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    top_k: int = Field(default=3, ge=1, le=5)


class CalcArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class NoArgs(BaseModel):
    pass


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod,
}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic only (no names, calls or attributes)."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(walk(node.right)) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("only basic arithmetic is allowed")
    return walk(ast.parse(expression.replace(",", ""), mode="eval"))


def build_tools(store) -> dict[str, Tool]:
    def search_docs(args: SearchArgs) -> dict:
        hits = store.search(args.query, args.top_k)
        safe, dropped = filter_untrusted_chunks(hits)
        return {"results": safe, "dropped_untrusted": dropped}

    def calculate(args: CalcArgs) -> dict:
        return {"result": safe_eval(args.expression)}

    def list_docs(_: NoArgs) -> dict:
        return {"documents": store.sources()}

    tools = [
        Tool("search_docs", "Search the document collection. Args: query (string), top_k (1-5). Returns passages with their source file.", SearchArgs, search_docs),
        Tool("calculate", "Evaluate an arithmetic expression, e.g. '1500 * 12'. Args: expression (string).", CalcArgs, calculate),
        Tool("list_docs", "List the available document names. No args.", NoArgs, list_docs),
    ]
    return {t.name: t for t in tools}
