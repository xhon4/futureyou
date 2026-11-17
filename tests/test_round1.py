"""Round-1 review regressions. Each test failed on aa75c61 before its fix."""

import json
import os
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


def test_cr_inside_a_line_does_not_forge_a_hunk():
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+real\r@@ -1,1 +80,1 @@\n"
        "+still\n"
    )
    ranges = report.diff_ranges(diff)
    assert report.validate(_regret("f.py", 80, 80), _messages(diff))[0] is not None
    assert not any(start <= 80 <= end for start, end in ranges.get("f.py", []))
    assert report.validate(_regret("f.py", 1, 1), _messages(diff))[0] is None


def test_formfeed_does_not_retarget_a_later_hunk(tmp_path):
    repo = _repo(tmp_path)
    lines = [f"line{i}" for i in range(1, 31)]
    target = repo / "real.py"
    target.write_text("\n".join(lines) + "\n")
    git(repo, "add", "real.py")
    git(repo, "commit", "-q", "-m", "add")
    edited = lines.copy()
    edited[1] = "\f"
    edited.insert(2, "++ b/innocent.py")
    edited[25] = "changed-line-25"
    target.write_text("\n".join(edited) + "\n")
    diff = tools.git_diff(repo)["diff"]
    ranges = report.diff_ranges(diff)
    assert "innocent.py" not in ranges, ranges
    assert ranges.get("real.py"), ranges


def test_blank_context_does_not_end_the_hunk():
    diff = (
        "diff --git a/blank.py b/blank.py\n"
        "--- a/blank.py\n"
        "+++ b/blank.py\n"
        "@@ -1,5 +1,5 @@\n"
        " a\n"
        "\n"
        "+++ b/innocent.py\n"
        "@@ -10 +10 @@\n"
        "-old\n"
        "+new\n"
    )
    ranges = report.diff_ranges(diff)
    assert "innocent.py" not in ranges, ranges
    assert any(start <= 10 <= end for start, end in ranges.get("blank.py", [])), ranges


def test_suppress_blank_empty_config_does_not_retarget(tmp_path):
    repo = _repo(tmp_path)
    target = repo / "blank.py"
    target.write_text("a\n\nb\n")
    git(repo, "add", "blank.py")
    git(repo, "commit", "-q", "-m", "add")
    target.write_text("a\n\n++ b/innocent.py\nb changed\n")
    git(repo, "config", "diff.suppressBlankEmpty", "true")
    diff = tools.git_diff(repo)["diff"]
    ranges = report.diff_ranges(diff)
    assert "innocent.py" not in ranges, ranges
    assert "b changed" in diff


def test_trailing_space_in_a_path_is_not_stripped(tmp_path):
    repo = _repo(tmp_path)
    (repo / "trail").write_text("a\n")
    (repo / "trail ").write_text("a\n")
    git(repo, "add", "--", "trail", "trail ")
    git(repo, "commit", "-q", "-m", "trails")
    (repo / "trail ").write_text("b\n")
    diff = tools.git_diff(repo)["diff"]
    ranges = report.diff_ranges(diff)
    assert ranges.get("trail "), ranges
    assert "b" in diff
    cited = report.validate(_regret("trail ", 1, 1), _messages(diff))[0]
    assert cited is None
    assert report.validate(_regret("trail", 1, 1), _messages(diff))[0] is not None


def test_binary_path_containing_b_slash_keeps_the_repo_path(tmp_path):
    repo = _repo(tmp_path)
    folder = repo / "foo b"
    folder.mkdir()
    target = folder / "bar.txt"
    target.write_bytes(b"a\n")
    git(repo, "add", "--", "foo b/bar.txt")
    git(repo, "commit", "-q", "-m", "add")
    target.write_bytes(b"\0bin")
    diff = tools.git_diff(repo)["diff"]
    ranges = report.diff_ranges(diff)
    assert "foo b/bar.txt" in ranges, ranges
    assert "bar.txt" not in ranges


def test_rename_path_containing_b_slash_uses_rename_to(tmp_path):
    repo = _repo(tmp_path)
    folder = repo / "foo b"
    folder.mkdir()
    target = folder / "bar.txt"
    target.write_text("a\n")
    git(repo, "add", "--", "foo b/bar.txt")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "new b").mkdir()
    git(repo, "mv", "--", "foo b/bar.txt", "new b/name.txt")
    diff = tools.git_diff(repo)["diff"]
    ranges = report.diff_ranges(diff)
    assert "new b/name.txt" in ranges, ranges
    assert "name.txt" not in ranges


def test_binary_attribute_is_disclosed_and_not_forced_to_text(tmp_path):
    repo = _repo(tmp_path)
    (repo / "note.txt").write_text("hello\n")
    git(repo, "add", "note.txt")
    git(repo, "commit", "-q", "-m", "add")
    (repo / ".gitattributes").write_text("*.txt binary\n")
    (repo / "note.txt").write_text("hello changed\n")
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert "hello changed" not in result["diff"]
    assert "Binary files" in result["diff"]
    assert result["binary_files"] >= 1
    scope = cli.review_scope(
        [ToolMessage(json.dumps(result), name="get_git_diff", tool_call_id="1")]
    )
    assert "shown as binary, not reviewed" in scope


def _writer(path: Path, marker: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"mkdir -p .git/evil && echo x > {marker}\n"
        "printf 'POINTER\\n'\n"
    )
    path.chmod(0o755)


def test_dotted_filter_driver_does_not_run(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    script = tmp_path / "clean.sh"
    marker = repo / ".git" / "evil" / "ran"
    _writer(script, marker)
    git(repo, "config", "filter.foo.bar.clean", str(script))
    git(repo, "config", "filter.foo.bar.required", "true")
    (repo / ".gitattributes").write_text("* filter=foo.bar\n")
    (repo / "tracked.txt").write_text("v2\n")
    result = tools.git_diff(repo)
    assert "v2" in result["diff"]
    assert "POINTER" not in result["diff"]
    assert not marker.exists()


def _other_repo(tmp_path: Path) -> Path:
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    (other / "secret.txt").write_text("SECRET_TOKEN\n")
    git(other, "add", "secret.txt")
    git(other, "commit", "-q", "-m", "secret")
    return other


def test_inherited_git_dir_does_not_read_another_repo(tmp_path, monkeypatch):
    reviewed = _repo(tmp_path)
    (reviewed / "tracked.txt").write_text("visible\n")
    git(reviewed, "add", "tracked.txt")
    git(reviewed, "commit", "-q", "-m", "add")
    (reviewed / "tracked.txt").write_text("visible changed\n")
    other = _other_repo(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    result = tools.git_diff(reviewed)
    assert "SECRET_TOKEN" not in result.get("diff", "")
    assert result["status"] == "ok"
    assert "visible changed" in result["diff"]


def test_git_dir_and_work_tree_do_not_move_resolve_repo(tmp_path, monkeypatch):
    reviewed = _repo(tmp_path)
    other = _other_repo(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert tools.resolve_repo(reviewed) == reviewed.resolve()


def test_symlinked_docs_parent_is_not_searched(tmp_path):
    repo = _repo(tmp_path)
    outside = tmp_path / "outside" / "adr"
    outside.mkdir(parents=True)
    (outside / "001.md").write_text("Use the OUTSIDE_TOKEN for sessions.\n")
    (repo / "docs").symlink_to(tmp_path / "outside")
    result = tools.search_decisions(repo, "OUTSIDE_TOKEN")
    assert result["status"] == "no_adrs"
    assert result["fragments"] == []
    assert "OUTSIDE_TOKEN" not in str(result)


def test_undecodable_untracked_name_does_not_drop_the_tracked_edit(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("safe\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "tracked.txt").write_text("safe changed\n")
    raw_name = bytes([0xFF]) + b"name.txt"
    fd = os.open(repo / os.fsdecode(raw_name), os.O_CREAT | os.O_WRONLY, 0o644)
    os.write(fd, b"secret\n")
    os.close(fd)
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert "safe changed" in result["diff"]


def test_short_abbrev_still_recognizes_an_empty_blob(tmp_path):
    repo = _repo(tmp_path)
    git(repo, "config", "core.abbrev", "4")
    (repo / "empty.txt").write_text("")
    diff = tools.git_diff(repo)["diff"]
    assert "empty.txt" in diff
    assert report.validate(_regret("empty.txt", 1, 1), _messages(diff))[0] is None
