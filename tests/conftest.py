import shutil
import socket
import subprocess
from pathlib import Path

import pytest

# The suite must stay runnable without a provider account.
_PROVIDER_ENV = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "COHERE_API_KEY",
    "DEEPSEEK_API_KEY",
    "FIREWORKS_API_KEY",
    "FUTUREYOU_MODEL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GROQ_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "MISTRAL_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "TOGETHER_API_KEY",
    "XAI_API_KEY",
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=FutureYou Tests",
            "-c",
            "user.email=tests@example.com",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture(autouse=True)
def _no_provider_and_no_network(monkeypatch):
    """Drop provider credentials and fail if a test opens a TCP connection."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)

    def refuse_connect(self, address):
        raise AssertionError(f"tests must not connect to {address}")

    def refuse_sendto(self, *args, **kwargs):
        target = kwargs.get("address")
        if target is None and args:
            target = args[-1]
        raise AssertionError(f"tests must not connect to {target}")

    def refuse_connection(*args, **kwargs):
        target = args[0] if args else kwargs.get("address")
        raise AssertionError(f"tests must not connect to {target}")

    monkeypatch.setattr(socket.socket, "connect", refuse_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_connect)
    monkeypatch.setattr(socket.socket, "sendto", refuse_sendto)
    monkeypatch.setattr(socket, "create_connection", refuse_connection)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Copy of examples/sample-project as a Git repo with one base commit."""
    repo = tmp_path / "sample-project"
    shutil.copytree(EXAMPLES / "sample-project", repo)
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    return repo


@pytest.fixture
def contradicting_repo(sample_repo: Path) -> Path:
    """sample_repo with the change that contradicts ADR-003 applied, unstaged."""
    shutil.copytree(EXAMPLES / "sample-change", sample_repo, dirs_exist_ok=True)
    return sample_repo
