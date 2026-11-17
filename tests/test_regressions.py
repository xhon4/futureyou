"""Regressions for the Git read path and the review graph.

Each test was written against the unmodified T4 behavior and fails there.
"""

import json
import subprocess
from pathlib import Path

from conftest import git
from langchain_core.messages import ToolMessage

from futureyou import cli, report, tools


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("base\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def _regret(path: str, start: int, end: int) -> report.Finding:
    return report.Finding(
        type="Potential regret",
        title="t",
        why="w",
        suggestion="s",
        change_path=path,
        change_start=start,
        change_end=end,
    )


def _messages(diff: str) -> list[ToolMessage]:
    payload = {"diff": diff, "status": "ok"}
    return [ToolMessage(json.dumps(payload), name="get_git_diff", tool_call_id="1")]


def _scope(result: dict) -> str:
    return cli.review_scope(
        [ToolMessage(json.dumps(result), name="get_git_diff", tool_call_id="1")]
    )


def test_repo_quotepath_config_does_not_mangle_the_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.quotePath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    repo = _repo(tmp_path)
    target = repo / "café.py"
    target.write_text("one\n")
    git(repo, "add", "--", "café.py")
    git(repo, "commit", "-q", "-m", "add")
    target.write_text("one\ntwo changed\n")
    git(repo, "config", "core.quotepath", "true")
    diff = tools.git_diff(repo)["diff"]
    assert r"\303" not in diff
    assert report.diff_ranges(diff).get("café.py"), report.diff_ranges(diff)


def test_mnemonic_prefix_config_does_not_change_the_diff(tmp_path):
    repo = _repo(tmp_path)
    target = repo / "auth.py"
    target.write_text("one\ntwo\n")
    git(repo, "add", "auth.py")
    git(repo, "commit", "-q", "-m", "add")
    target.write_text("one\ntwo changed\n")
    git(repo, "config", "diff.mnemonicPrefix", "true")
    diff = tools.git_diff(repo)["diff"]
    assert "w/auth.py" not in diff
    assert report.diff_ranges(diff).get("auth.py"), report.diff_ranges(diff)


def test_non_utf8_text_does_not_crash(tmp_path):
    repo = _repo(tmp_path)
    (repo / "latin.txt").write_bytes("café\n".encode("latin-1"))
    git(repo, "add", "latin.txt")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "latin.txt").write_bytes("café changed\n".encode("latin-1"))
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert "changed" in result["diff"]


def test_index_only_staged_edit_is_part_of_the_review(tmp_path):
    repo = _repo(tmp_path)
    target = repo / "auth.py"
    target.write_text("safe\n")
    git(repo, "add", "auth.py")
    git(repo, "commit", "-q", "-m", "base")
    target.write_text("bug\n")
    git(repo, "add", "auth.py")
    target.write_text("safe\n")
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert "+bug" in result["diff"]
    assert "+safe" in result["diff"]


def test_clean_filter_does_not_rewrite_the_diff_or_write_git(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    script = tmp_path / "clean.sh"
    script.write_text(
        "#!/bin/sh\nmkdir -p .git/filter-ran\necho ran >> .git/filter-ran/marker\nprintf 'POINTER\\n'\n"
    )
    script.chmod(0o755)
    git(repo, "config", "filter.evil.clean", str(script))
    git(repo, "config", "filter.evil.required", "true")
    (repo / ".gitattributes").write_text("* filter=evil\n")
    (repo / "tracked.txt").write_text("v2\n")
    (repo / "new.txt").write_text("new-body\n")
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert "v2" in result["diff"]
    assert "new-body" in result["diff"]
    assert "POINTER" not in result["diff"]
    assert not (repo / ".git" / "filter-ran").exists()


def test_fsmonitor_hook_does_not_run(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    marker = tmp_path / "fsm-marker"
    hook = tmp_path / "fsm.sh"
    hook.write_text(f"#!/bin/sh\necho ran >> {marker}\nprintf '\\0'\n")
    hook.chmod(0o755)
    git(repo, "config", "core.fsmonitor", str(hook))
    (repo / "tracked.txt").write_text("v2\n")
    result = tools.git_diff(repo)
    assert "v2" in result["diff"]
    assert not marker.exists()


def test_scope_does_not_claim_omitted_untracked_files(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "big.py").write_text("seed\n")
    git(repo, "add", "big.py")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "big.py").write_text("\n".join(f"line{i}" for i in range(80)) + "\n")
    (repo / "new.txt").write_text("UNTRACKED_MARKER\n")
    monkeypatch.setattr(tools, "MAX_DIFF_CHARS", 500)
    result = tools.git_diff(repo)
    assert result["truncated"] is True
    assert "UNTRACKED_MARKER" not in result["diff"]
    scope = _scope(result)
    assert "untracked files included" not in scope
    assert "truncat" in scope


def test_untracked_nested_repo_is_disclosed_and_not_counted(tmp_path):
    repo = _repo(tmp_path)
    nested = repo / "vendor"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "lib.txt").write_text("library\n")
    git(nested, "add", "lib.txt")
    git(nested, "commit", "-q", "-m", "lib")
    result = tools.git_diff(repo)
    assert "library" not in result["diff"]
    assert result["untracked"] == 0
    assert result["nested_skipped"] == 1
    scope = _scope(result)
    assert "nested" in scope
    assert "untracked files included" not in scope


def test_dirty_submodule_is_disclosed_without_its_contents(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    git(sub, "init", "-q")
    (sub / "inner.py").write_text("version1\n")
    git(sub, "add", "inner.py")
    git(sub, "commit", "-q", "-m", "inner")
    parent = tmp_path / "parent"
    parent.mkdir()
    git(parent, "init", "-q")
    (parent / "README.md").write_text("parent\n")
    git(parent, "add", "README.md")
    git(parent, "commit", "-q", "-m", "base")
    git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--",
        str(sub),
        "sub",
    )
    git(parent, "commit", "-q", "-m", "add sub")
    (parent / "sub" / "inner.py").write_text("version2 CHANGED\n")
    (parent / "sub" / "extra.txt").write_text("untracked-inner\n")
    git(parent, "config", "diff.submodule", "diff")
    result = tools.git_diff(parent)
    assert "version2" not in result["diff"]
    assert "untracked-inner" not in result["diff"]
    assert "Subproject commit" in result["diff"]
    assert result["submodules"] >= 1
    assert "submodule" in _scope(result)


def test_oversized_untracked_file_is_not_fully_buffered(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "big.txt").write_text("x" * 200_000 + "\n")
    sizes: list[int] = []
    real = subprocess.run

    def spy(cmd, **kwargs):
        completed = real(cmd, **kwargs)
        sizes.append(len(completed.stdout or ""))
        return completed

    monkeypatch.setattr(tools.subprocess, "run", spy)
    result = tools.git_diff(repo)
    assert max(sizes) <= tools.MAX_DIFF_CHARS
    assert "x" * 1000 not in result["diff"]
    assert result["oversized_untracked"] == 1


def test_missing_git_is_not_called_a_model_failure(tmp_path, monkeypatch, capsys):
    empty = tmp_path / "bin"
    empty.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert cli.main(["review", str(repo)]) == 1
    err = capsys.readouterr().err
    assert "model call failed" not in err
    assert "git" in err.lower()


def test_decode_error_is_not_called_a_model_failure(tmp_path, monkeypatch, capsys):
    def boom(path):
        raise UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "invalid continuation byte")

    monkeypatch.setattr(cli.tools, "resolve_repo", boom)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert cli.main(["review", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "model call failed" not in err
    assert "decode" in err.lower()


def test_git_commands_pin_safe_config(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("z\n")
    calls = []
    real = subprocess.run

    def spy(cmd, **kwargs):
        calls.append((cmd, kwargs.get("env") or {}))
        return real(cmd, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", spy)
    tools.git_diff(repo)
    assert calls
    for cmd, env in calls:
        assert "core.quotepath=false" in cmd
        assert "diff.mnemonicPrefix=false" in cmd
        assert "diff.noprefix=false" in cmd
        assert "core.fsmonitor=false" in cmd
        assert "core.hooksPath=/dev/null" in cmd
        assert env.get("GIT_OPTIONAL_LOCKS") == "0"
