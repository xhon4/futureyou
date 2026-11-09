import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import git
from futureyou import tools


def test_git_diff_includes_staged_unstaged_and_untracked(sample_repo: Path):
    (sample_repo / "src" / "api.ts").write_text("// staged change\n")
    git(sample_repo, "add", "src/api.ts")
    (sample_repo / "src" / "auth.ts").write_text("// unstaged change\n")
    (sample_repo / "src" / "new.ts").write_text("// untracked file\n")

    before = subprocess.run(["git", "status", "--porcelain"], cwd=sample_repo, capture_output=True, check=True).stdout
    result = tools.git_diff(sample_repo)
    after = subprocess.run(["git", "status", "--porcelain"], cwd=sample_repo, capture_output=True, check=True).stdout

    assert before == after
    assert result["status"] == "ok"
    assert result["truncated"] is False
    assert "+// staged change" in result["diff"]
    assert "+// unstaged change" in result["diff"]
    assert "new file mode" in result["diff"]
    assert "+// untracked file" in result["diff"]
    assert result["untracked"] == 1
    assert result["untracked_skipped"] == 0


def test_git_diff_reports_no_changes(sample_repo: Path):
    assert tools.git_diff(sample_repo) == {"status": "no_changes", "diff": "", "truncated": False, "untracked": 0, "untracked_skipped": 0}


def test_git_diff_untracked_only_and_ignored(sample_repo: Path):
    (sample_repo / ".gitignore").write_text("ignored.txt\n")
    (sample_repo / "ignored.txt").write_text("ignored content\n")
    (sample_repo / "new.txt").write_text("new content\n")
    result = tools.git_diff(sample_repo)
    assert result["status"] == "ok"
    assert "+new content" in result["diff"]
    assert "ignored content" not in result["diff"]
    assert result["untracked"] == 2


def test_git_diff_untracked_binary_and_symlink(sample_repo: Path, tmp_path: Path):
    (sample_repo / "binary.dat").write_bytes(b"\x00binary-secret\xff")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret\n")
    (sample_repo / "link").symlink_to(outside)
    result = tools.git_diff(sample_repo)
    assert "Binary files /dev/null and b/binary.dat differ" in result["diff"]
    assert "binary-secret" not in result["diff"]
    assert "outside secret" not in result["diff"]
    assert result["untracked"] == 1
    assert result["untracked_skipped"] == 0


def test_git_diff_untracked_cap(sample_repo: Path, monkeypatch):
    monkeypatch.setattr(tools, "MAX_UNTRACKED_FILES", 1, raising=False)
    (sample_repo / "a.txt").write_text("first\n")
    (sample_repo / "b.txt").write_text("second\n")
    result = tools.git_diff(sample_repo)
    assert result["untracked"] == 1
    assert result["untracked_skipped"] == 1
    assert "+first" in result["diff"]
    assert "+second" not in result["diff"]


def test_git_diff_untracked_content_is_truncated(sample_repo: Path, monkeypatch):
    monkeypatch.setattr(tools, "MAX_DIFF_CHARS", 200)
    (sample_repo / "new.txt").write_text("content\n" * 100)
    result = tools.git_diff(sample_repo)
    assert result["status"] == "ok"
    assert result["truncated"] is True
    assert len(result["diff"]) <= 200


def test_git_diff_truncates_at_a_line_boundary(sample_repo: Path):
    (sample_repo / "src" / "auth.ts").write_text("x" * 99 + "\n" * 1 + ("y" * 99 + "\n") * 1000)

    result = tools.git_diff(sample_repo)

    assert result["truncated"] is True
    assert len(result["diff"]) <= tools.MAX_DIFF_CHARS
    assert result["diff"].endswith("\n")


def test_resolve_repo_returns_root_from_subdirectory(sample_repo: Path):
    assert tools.resolve_repo(sample_repo / "src") == sample_repo.resolve()


def test_resolve_repo_rejects_non_repo_and_repo_without_commits(tmp_path: Path):
    with pytest.raises(tools.RepoError, match="not a Git repository"):
        tools.resolve_repo(tmp_path)
    git(tmp_path, "init", "-q")
    with pytest.raises(tools.RepoError, match="no commits"):
        tools.resolve_repo(tmp_path)


def test_search_decisions_returns_stable_fragments_with_status(sample_repo: Path):
    result = tools.search_decisions(sample_repo, "session token localStorage")

    assert result["status"] == "ok"
    assert result == tools.search_decisions(sample_repo, "session token localStorage")
    top = result["fragments"][0]
    assert top["path"] == "docs/adr/003-authentication.md"
    assert top["adr_status"] == "Accepted"
    assert top["end_line"] - top["start_line"] < tools.MAX_FRAGMENT_LINES
    lines = (sample_repo / top["path"]).read_text().splitlines()
    assert top["excerpt"] == "\n".join(lines[top["start_line"] - 1 : top["end_line"]])
    assert "localStorage" in top["excerpt"]
    assert len(result["fragments"]) <= tools.MAX_FRAGMENTS


def test_search_decisions_without_matches(sample_repo: Path):
    result = tools.search_decisions(sample_repo, "kubernetes")
    assert result == {"status": "no_matches", "fragments": [], "skipped": []}


def test_search_decisions_without_adrs(sample_repo: Path):
    shutil.rmtree(sample_repo / "docs")
    result = tools.search_decisions(sample_repo, "session")
    assert result == {"status": "no_adrs", "fragments": [], "skipped": []}


def test_search_decisions_skips_oversized_files_and_symlinks(sample_repo: Path, tmp_path: Path):
    adr_dir = sample_repo / "docs" / "adr"
    (adr_dir / "004-huge.md").write_text("session\n" * (tools.MAX_ADR_BYTES // 8 + 1))
    outside = tmp_path / "outside.md"
    outside.write_text("session secret outside the repo\n")
    (adr_dir / "005-link.md").symlink_to(outside)

    result = tools.search_decisions(sample_repo, "session")

    assert result["status"] == "partial"
    assert result["skipped"] == [{"path": "docs/adr/004-huge.md", "reason": "larger than 64 KiB"}]
    assert all("outside" not in f["excerpt"] for f in result["fragments"])
