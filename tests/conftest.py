import shutil
import subprocess
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=FutureYou Tests", "-c", "user.email=tests@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


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
