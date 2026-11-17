"""Static checks for invariants that are easy to bypass by accident."""

import ast
import socket
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "futureyou"
_MODULES = {"subprocess", "os", "asyncio"}
_EXACT = {
    "subprocess": {
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "getoutput",
        "getstatusoutput",
    },
    "os": {"system", "popen"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
}


def _is_launch(module: str, name: str) -> bool:
    if name in _EXACT.get(module, ()):
        return True
    return module == "os" and (
        name.startswith(("exec", "spawn")) or name.startswith("posix_spawn")
    )


def _bindings(tree: ast.AST) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    modules: dict[str, str] = {}
    functions: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _MODULES:
                    modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module in _MODULES:
            for alias in node.names:
                functions[alias.asname or alias.name] = (node.module, alias.name)
    return modules, functions


def _launches(tree: ast.AST) -> list[tuple[int, str]]:
    """Process launches found in one module, as ``(lineno, module.name)``."""
    modules, functions = _bindings(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _MODULES:
            if any(alias.name == "*" for alias in node.names):
                found.append((node.lineno, f"{node.module}.*"))
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            pair = functions.get(func.id)
            if pair and (pair[1] == "*" or _is_launch(*pair)):
                found.append((node.lineno, f"{pair[0]}.{pair[1]}"))
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = modules.get(func.value.id)
            if module and _is_launch(module, func.attr):
                found.append((node.lineno, f"{module}.{func.attr}"))
    return found


def _span(tree: ast.AST, name: str) -> tuple[int, int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node.lineno, node.end_lineno or node.lineno
    raise AssertionError(f"{name} is not defined")


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ("import os\ndef f():\n    os.system('x')\n", "os.system"),
        ("import os\ndef f():\n    os.popen('x')\n", "os.popen"),
        ("import os\ndef f():\n    os.execv('/bin/true', ['true'])\n", "os.execv"),
        ("import os\ndef f():\n    os.spawnv(0, '/bin/true', ['true'])\n", "os.spawnv"),
        (
            "import os\ndef f():\n    os.posix_spawn(b'/bin/true', [], [])\n",
            "os.posix_spawn",
        ),
        (
            "import asyncio\ndef f():\n    asyncio.create_subprocess_exec('x')\n",
            "asyncio.create_subprocess_exec",
        ),
        (
            "import asyncio\ndef f():\n    asyncio.create_subprocess_shell('x')\n",
            "asyncio.create_subprocess_shell",
        ),
        (
            "from subprocess import getoutput\ndef f():\n    getoutput('x')\n",
            "subprocess.getoutput",
        ),
        (
            "from subprocess import getstatusoutput as status\ndef f():\n    status('x')\n",
            "subprocess.getstatusoutput",
        ),
        ("import subprocess as sp\ndef f():\n    sp.run(['x'])\n", "subprocess.run"),
        ("from subprocess import *\n", "subprocess.*"),
    ],
)
def test_process_launch_forms_are_visible(source, label):
    found = [name for _, name in _launches(ast.parse(source))]
    assert label in found, found


def test_sendto_is_refused():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with pytest.raises(AssertionError):
        sock.sendto(b"x", ("127.0.0.1", 9))


def test_every_git_subprocess_goes_through_one_helper():
    """The only process launch in the package is Popen inside _git."""
    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text())
        found = _launches(tree)
        if path.name != "tools.py":
            assert found == [], path.name
            continue
        start, end = _span(tree, "_git")
        assert [label for _, label in found] == ["subprocess.Popen"]
        line = found[0][0]
        assert start <= line <= end
