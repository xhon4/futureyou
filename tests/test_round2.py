"""Round-2 review regressions. Each behavior test failed on c8231f6."""

import json
import time
from pathlib import Path

import pytest
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


def _writer(path: Path, marker: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"mkdir -p {marker.parent}\n"
        f"echo ran > {marker}\n"
        "printf 'POINTER\\n'\n"
    )
    path.chmod(0o755)


def _assert_worktree_bytes(result: dict, marker: Path, body: str) -> None:
    assert result["status"] == "ok", result
    assert body in result["diff"]
    assert "POINTER" not in result["diff"]
    assert not marker.exists()


def test_filter_driver_with_equals_does_not_run(tmp_path, monkeypatch):
    """GIT_CONFIG_COUNT already set must stay in force while the driver is disabled."""
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "diff.context")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "0")
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("a\nb\nc\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    script = tmp_path / "clean.sh"
    marker = repo / ".git" / "evil" / "ran"
    _writer(script, marker)
    git(repo, "config", "filter.foo=bar.clean", str(script))
    git(repo, "config", "filter.foo=bar.required", "true")
    (repo / ".gitattributes").write_text("* filter=foo=bar\n")
    (repo / "tracked.txt").write_text("a\nB\nc\n")
    result = tools.git_diff(repo)
    _assert_worktree_bytes(result, marker, "+B\n")
    # diff.context=0 was already in the environment. Replacing GIT_CONFIG_COUNT
    # would drop it and the hunk would grow back to three lines.
    assert "@@ -2 +2 @@" in result["diff"]


def test_filter_driver_with_a_non_utf8_name_does_not_run(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    script = tmp_path / "clean.sh"
    marker = repo / ".git" / "evil" / "ran"
    _writer(script, marker)
    driver = b"foo" + bytes([0xFF]) + b"bar"
    config = (repo / ".git" / "config").read_bytes()
    (repo / ".git" / "config").write_bytes(
        config
        + b'\n[filter "'
        + driver
        + b'"]\n\tclean = '
        + str(script).encode()
        + b"\n\trequired = true\n"
    )
    (repo / ".gitattributes").write_bytes(b"* filter=" + driver + b"\n")
    (repo / "tracked.txt").write_text("v2\n")
    result = tools.git_diff(repo)
    _assert_worktree_bytes(result, marker, "+v2\n")


def test_setsid_child_does_not_extend_the_git_timeout(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "leaked"
    program = bindir / "git"
    program.write_text(
        "#!/bin/sh\n"
        f"/usr/bin/setsid /bin/sh -c '/bin/sleep 3; echo leaked > {marker}' &\n"
        "/bin/sleep 30\n"
    )
    program.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(tools, "GIT_TIMEOUT_SECONDS", 0.4)
    repo = tmp_path / "repo"
    repo.mkdir()
    started = time.monotonic()
    with pytest.raises(tools.RepoError, match="timed out"):
        tools.resolve_repo(repo)
    elapsed = time.monotonic() - started
    assert elapsed < 1.5, elapsed
    assert not marker.exists()


def _assert_full_empty_path(diff: str) -> None:
    ranges = report.diff_ranges(diff)
    assert set(ranges) == {"foo b/empty.txt"}, ranges
    assert report.validate(_regret("foo b/empty.txt", 1, 1), _messages(diff))[0] is None
    assert report.validate(_regret("empty.txt", 1, 1), _messages(diff))[0] is not None


def test_empty_path_containing_b_slash_is_citable_only_under_the_full_name(tmp_path):
    repo = _repo(tmp_path)
    folder = repo / "foo b"
    folder.mkdir()
    (folder / "empty.txt").write_text("")
    untracked = tools.git_diff(repo)
    assert untracked["status"] == "ok"
    _assert_full_empty_path(untracked["diff"])
    git(repo, "add", "--", "foo b/empty.txt")
    staged = tools.git_diff(repo)
    assert staged["status"] == "ok"
    _assert_full_empty_path(staged["diff"])


def test_unequal_diff_header_without_a_rename_is_not_citable():
    diff = "diff --git a/foo b/bar\nnew file mode 100644\nindex 0000000..e69de29\n"
    assert report.diff_ranges(diff) == {}


def test_copy_to_path_containing_b_slash_is_the_citable_name():
    diff = (
        "diff --git a/orig.txt b/new b/copy.txt\n"
        "similarity index 100%\n"
        "copy from orig.txt\n"
        "copy to new b/copy.txt\n"
    )
    ranges = report.diff_ranges(diff)
    assert "new b/copy.txt" in ranges, ranges
    assert "copy.txt" not in ranges


def test_new_binary_dev_null_notice_keeps_the_section_path(tmp_path):
    repo = _repo(tmp_path)
    folder = repo / "foo and b"
    folder.mkdir()
    (folder / "bar.txt").write_bytes(b"\0new")
    git(repo, "add", "--", "foo and b/bar.txt")
    result = tools.git_diff(repo)
    assert "Binary files /dev/null and b/foo and b/bar.txt differ" in result["diff"]
    ranges = report.diff_ranges(result["diff"])
    assert "foo and b/bar.txt" in ranges, ranges
    assert "bar.txt" not in ranges


def test_modified_binary_path_containing_and_b_keeps_the_section_path(tmp_path):
    repo = _repo(tmp_path)
    folder = repo / "foo and b"
    folder.mkdir()
    target = folder / "bar.txt"
    target.write_bytes(b"\0v1")
    git(repo, "add", "--", "foo and b/bar.txt")
    git(repo, "commit", "-q", "-m", "add")
    target.write_bytes(b"\0v2")
    diff = tools.git_diff(repo)["diff"]
    assert "Binary files a/foo and b/bar.txt and b/foo and b/bar.txt differ" in diff
    ranges = report.diff_ranges(diff)
    assert "foo and b/bar.txt" in ranges, ranges
    assert "bar.txt" not in ranges
    assert "bar.txt and b/foo and b/bar.txt" not in ranges


def test_binary_scope_counts_each_path_once(tmp_path):
    repo = _repo(tmp_path)
    (repo / "file.bin").write_bytes(b"base")
    (repo / "other.bin").write_bytes(b"base")
    git(repo, "add", "file.bin", "other.bin")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "file.bin").write_bytes(b"\0staged")
    (repo / "other.bin").write_bytes(b"\0staged")
    git(repo, "add", "file.bin", "other.bin")
    (repo / "file.bin").write_bytes(b"\0unstaged")
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert result["binary_files"] == 2
    scope = _scope(result)
    assert "2 files shown as binary, not reviewed" in scope
