import shutil
from pathlib import Path

import pytest

from conftest import git
from futureyou import tools


def test_git_diff_includes_staged_and_unstaged_but_not_untracked(sample_repo: Path):
    (sample_repo / "src" / "api.ts").write_text("// staged change\n")
    git(sample_repo, "add", "src/api.ts")
    (sample_repo / "src" / "auth.ts").write_text("// unstaged change\n")
    (sample_repo / "src" / "new.ts").write_text("// untracked file\n")

    result = tools.git_diff(sample_repo)

    assert result["status"] == "ok"
    assert result["truncated"] is False
    assert "+// staged change" in result["diff"]
    assert "+// unstaged change" in result["diff"]
    assert "new.ts" not in result["diff"]


def test_git_diff_reports_no_changes(sample_repo: Path):
    assert tools.git_diff(sample_repo) == {"status": "no_changes", "diff": "", "truncated": False}


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
