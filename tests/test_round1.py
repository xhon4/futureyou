"""Round-1 review regressions. Each test failed on aa75c61 before its fix."""

import json
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
