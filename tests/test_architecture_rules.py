"""The no-LangGraph rule is a hard architectural decision, not a preference. Nothing in the codebase
enforces it, so a future dependency or import could reintroduce LangGraph silently. These tests are the
guard rail: they fail if the rule is broken anywhere in the shipped source."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "ba_agent"

#: LangGraph is the runtime `create_agent` compiles to. Importing it, even transitively through a
# convenience wrapper, would hand control flow back to a graph engine — the exact thing Part 6.1 rejects.
FORBIDDEN_MODULES = {"langgraph", "deepagents"}
FORBIDDEN_NAMES = {"create_agent", "StateGraph", "interrupt", "Command"}

#: Also forbidden as *identifiers*: `create_agent` is legal-looking LangChain API but compiles to LangGraph.
ALLOWED_NAME_EXCEPTIONS: set[str] = set()


def _python_files() -> list[Path]:
    return list(SRC.rglob("*.py"))


def test_no_langgraph_or_agent_factory_is_imported():
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in FORBIDDEN_MODULES:
                        offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root_module = (node.module or "").split(".")[0]
                if root_module in FORBIDDEN_MODULES:
                    offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno} from {node.module}")
                for alias in node.names:
                    if alias.name in FORBIDDEN_NAMES:
                        offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno} {alias.name}")
    assert not offenders, f"forbidden orchestration imports found: {offenders}"


def test_langgraph_is_not_an_installed_dependency_of_the_project():
    """A package can be imported only if it is a dependency; pin the dependency list instead of trusting CI."""
    manifest = (SRC.parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = re.search(r"dependencies\s*=\s*\[(.*?)\]", manifest, re.S)
    assert dependencies, "pyproject.toml has no dependencies block"
    block = dependencies.group(1).lower()
    assert "langgraph" not in block, "langgraph must not be a dependency"
    assert '"langchain"' not in block, "the umbrella `langchain` package pulls in LangGraph; use langchain-core"


def test_orchestration_is_a_deterministic_state_machine_not_an_agent_loop():
    """Control flow must live in workflow.py. If the agent loop ever grows a graph, the rule is broken."""
    loop = (SRC / "agent_loop.py").read_text(encoding="utf-8")
    assert "Runnable" in loop, "the agent loop should be composed from langchain-core Runnables"
    assert not re.search(r"\b(CompiledGraph|StateGraph|\.invoke\(\s*state\s*\))", loop)


if __name__ == "__main__":
    test_no_langgraph_or_agent_factory_is_imported()
    test_langgraph_is_not_an_installed_dependency_of_the_project()
    test_orchestration_is_a_deterministic_state_machine_not_an_agent_loop()
    print("LangGraph guard rails: clean")
