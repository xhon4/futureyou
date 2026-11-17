"""Run the real CLI against temporary Git repositories.

The model is a stub. These tests check the paths a user hits: nothing to
review, a tracked edit, an untracked file, a path Git cannot review, and
that the repository bytes (including `.git` and the index) do not change.
"""

import os
from pathlib import Path

from conftest import git
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from futureyou import agent, cli, report


class StubModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda messages: report.Report(findings=[]))


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("base\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def _snapshot(repo: Path) -> dict[str, tuple[bytes, int]]:
    files = {}
    for path in sorted(repo.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        stat_result = path.stat()
        files[path.relative_to(repo).as_posix()] = (
            path.read_bytes(),
            stat_result.st_mtime_ns,
        )
    return files


def _git_dir_mtimes(repo: Path) -> dict[str, int]:
    git_path = repo / ".git"
    if git_path.is_file():
        line = git_path.read_text().strip()
        raw = Path(line.split(":", 1)[1].strip())
        git_dir = raw if raw.is_absolute() else (repo / raw).resolve()
    else:
        git_dir = git_path
    mtimes = {}
    for path in sorted(git_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            mtimes[str(path)] = path.stat().st_mtime_ns
    return mtimes


def _use_stub(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        agent,
        "load_model",
        lambda: StubModel(messages=iter([AIMessage("reviewed")])),
    )


def test_cli_review_with_no_changes(monkeypatch, tmp_path, capsys):
    _use_stub(monkeypatch, tmp_path)
    repo = _init_repo(tmp_path)
    assert cli.main(["review", str(repo), "--thread", "clean"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "No concerns" in out
    assert "Review scope: no changes against HEAD; ADRs not searched." in out


def test_cli_review_tracked_change(monkeypatch, tmp_path, capsys):
    _use_stub(monkeypatch, tmp_path)
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("base\nchanged\n")
    assert cli.main(["review", str(repo), "--thread", "tracked"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "FutureYou Report" in out
    assert "changes against HEAD, untracked files included; no truncation" in out


def test_cli_review_untracked_file(monkeypatch, tmp_path, capsys):
    _use_stub(monkeypatch, tmp_path)
    repo = _init_repo(tmp_path)
    (repo / "new.txt").write_text("untracked-body\n")
    assert cli.main(["review", str(repo), "--thread", "untracked"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "changes against HEAD, untracked files included; no truncation" in out


def test_cli_review_rejects_a_non_repo(monkeypatch, tmp_path, capsys):
    _use_stub(monkeypatch, tmp_path)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert cli.main(["review", str(plain)]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "not a Git repository" in err
    assert "model call failed" not in err


def test_cli_review_rejects_a_repo_without_commits(monkeypatch, tmp_path, capsys):
    _use_stub(monkeypatch, tmp_path)
    repo = tmp_path / "empty"
    repo.mkdir()
    git(repo, "init", "-q")
    assert cli.main(["review", str(repo)]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "no commits" in err
    assert "model call failed" not in err


def test_cli_review_leaves_the_repository_byte_identical(monkeypatch, tmp_path, capsys):
    _use_stub(monkeypatch, tmp_path)
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("base\nchanged\n")
    (repo / "new.txt").write_text("untracked-body\n")
    os.utime(repo / "README.md", None)
    before = _snapshot(repo)
    assert cli.main(["review", str(repo), "--thread", "readonly"]) == 0
    assert capsys.readouterr().err == ""
    assert _snapshot(repo) == before
    assert before[".git/index"][0] == (repo / ".git" / "index").read_bytes()


def test_cli_review_of_a_linked_worktree_does_not_touch_git_mtimes(
    monkeypatch, tmp_path, capsys
):
    _use_stub(monkeypatch, tmp_path)
    repo = _init_repo(tmp_path)
    worktree = tmp_path / "worktree"
    git(repo, "worktree", "add", "-q", str(worktree), "HEAD")
    (worktree / "README.md").write_text("base\nchanged\n")
    before_main = _git_dir_mtimes(repo)
    before_worktree = _snapshot(worktree)
    assert cli.main(["review", str(worktree), "--thread", "worktree"]) == 0
    assert capsys.readouterr().err == ""
    assert _git_dir_mtimes(repo) == before_main
    assert _snapshot(worktree) == before_worktree
