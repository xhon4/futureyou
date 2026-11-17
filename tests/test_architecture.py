"""Static checks for invariants that are easy to bypass by accident."""

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "futureyou"
_SUBPROCESS_FUNCS = {"run", "Popen", "call", "check_call", "check_output"}


def _subprocess_calls(tree: ast.AST) -> list[tuple[ast.Call, str]]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _SUBPROCESS_FUNCS:
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "subprocess":
            found.append((node, node.func.attr))
    return found


def test_every_git_subprocess_goes_through_one_helper():
    """The only subprocess launch in the package is Popen inside _git."""
    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text())
        calls = _subprocess_calls(tree)
        if path.name != "tools.py":
            assert calls == [], path.name
            continue
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        inside = _subprocess_calls(functions["_git"])
        assert [name for _, name in inside] == ["Popen"]
        assert len(calls) == 1
