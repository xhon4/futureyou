"""Citation parsing regressions.

Body lines that look like diff headers, Git C quoting, deletion line numbers,
empty files, and truncated hunks.
"""

import json
from pathlib import Path

from conftest import git
from langchain_core.messages import ToolMessage

from futureyou import report, tools


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


def test_mnemonic_prefix_paths_parse_to_the_repo_path():
    diff = (
        "diff --git c/auth.py w/auth.py\n"
        "--- c/auth.py\n"
        "+++ w/auth.py\n"
        "@@ -1 +1,2 @@\n"
        " one\n"
        "+changed\n"
    )
    assert report.diff_ranges(diff).get("auth.py"), report.diff_ranges(diff)


def test_added_diff_marker_lines_do_not_retarget_hunks(tmp_path):
    repo = _repo(tmp_path)
    lines = [f"line{i}" for i in range(1, 31)]
    target = repo / "real.py"
    target.write_text("\n".join(lines) + "\n")
    git(repo, "add", "real.py")
    git(repo, "commit", "-q", "-m", "add")
    edited = lines.copy()
    edited[1] = "++ b/innocent.py"
    edited[24] = "changed-line-25"
    target.write_text("\n".join(edited) + "\n")
    diff = tools.git_diff(repo)["diff"]
    ranges = report.diff_ranges(diff)
    assert "innocent.py" not in ranges, ranges
    assert len(ranges.get("real.py", [])) >= 2, ranges
    stolen = ranges.get("innocent.py", [(22, 28)])[0]
    rendered = report.render([_regret("innocent.py", *stolen)], _messages(diff))
    assert "[Unverified]" in rendered


def test_added_dev_null_marker_does_not_drop_ranges(tmp_path):
    repo = _repo(tmp_path)
    target = repo / "wipe.py"
    target.write_text("one\ntwo\nthree\nfour\n")
    git(repo, "add", "wipe.py")
    git(repo, "commit", "-q", "-m", "add")
    target.write_text("one\ntwo\n++ /dev/null\nfour changed\n")
    diff = tools.git_diff(repo)["diff"]
    ranges = report.diff_ranges(diff)
    assert ranges.get("wipe.py"), ranges
    assert report.validate(_regret("wipe.py", 4, 4), _messages(diff))[0] is None


def test_c_quoted_paths_cite_the_repo_path():
    diff = (
        'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n'
        '--- "a/caf\\303\\251.py"\n'
        '+++ "b/caf\\303\\251.py"\n'
        "@@ -1 +1,2 @@\n"
        " one\n"
        "+changed\n"
    )
    ranges = report.diff_ranges(diff)
    assert list(ranges) == ["café.py"], ranges
    start, end = ranges["café.py"][0]
    assert report.validate(_regret("café.py", start, end), _messages(diff))[0] is None


def test_c_quoted_embedded_quote_cites_the_repo_path():
    diff = (
        'diff --git "a/a\\"b.py" "b/a\\"b.py"\n'
        '--- "a/a\\"b.py"\n'
        '+++ "b/a\\"b.py"\n'
        "@@ -1 +1,2 @@\n"
        " x\n"
        "+y\n"
    )
    ranges = report.diff_ranges(diff)
    assert list(ranges) == ['a"b.py'], ranges
    assert report.validate(_regret('a"b.py', 1, 2), _messages(diff))[0] is None


def test_deleted_lines_inside_a_surviving_file_are_citable(tmp_path):
    repo = _repo(tmp_path)
    lines = [f"l{i}" for i in range(1, 11)]
    target = repo / "base.txt"
    target.write_text("\n".join(lines) + "\n")
    git(repo, "add", "base.txt")
    git(repo, "commit", "-q", "-m", "add")
    target.write_text("\n".join(lines[:8]) + "\n")
    diff = tools.git_diff(repo)["diff"]
    assert report.validate(_regret("base.txt", 9, 10), _messages(diff))[0] is None
    assert report.validate(_regret("base.txt", 11, 11), _messages(diff))[0] is not None


def test_emptied_file_lines_remain_citable(tmp_path):
    repo = _repo(tmp_path)
    (repo / "notes.txt").write_text("one\ntwo\nthree\n")
    git(repo, "add", "notes.txt")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "notes.txt").write_text("")
    diff = tools.git_diff(repo)["diff"]
    assert report.diff_ranges(diff).get("notes.txt"), report.diff_ranges(diff)
    assert report.validate(_regret("notes.txt", 1, 3), _messages(diff))[0] is None


def test_empty_new_file_is_citable(tmp_path):
    repo = _repo(tmp_path)
    (repo / "empty.txt").write_text("")
    diff = tools.git_diff(repo)["diff"]
    assert "empty.txt" in diff
    assert report.validate(_regret("empty.txt", 1, 1), _messages(diff))[0] is None


def test_truncated_hunk_does_not_verify_lines_absent_from_the_body(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path)
    (repo / "big.py").write_text("\n".join(f"old{i}" for i in range(80)) + "\n")
    git(repo, "add", "big.py")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "big.py").write_text("\n".join(f"new{i}" for i in range(80)) + "\n")
    monkeypatch.setattr(tools, "MAX_DIFF_CHARS", 400)
    diff = tools.git_diff(repo)["diff"]
    assert "new79" not in diff
    ranges = report.diff_ranges(diff).get("big.py", [])
    assert not any(start <= 1 <= 80 <= end for start, end in ranges), ranges
    assert report.validate(_regret("big.py", 80, 80), _messages(diff))[0] is not None
