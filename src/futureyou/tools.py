"""Read-only access to the reviewed repository: the Git diff and its ADRs.

These functions know nothing about the model or the CLI. They return plain
dicts so the agent can pass them to the model and tests can assert on them.
"""

import re
import subprocess
from pathlib import Path

MAX_DIFF_CHARS = 60_000
MAX_UNTRACKED_FILES = 200
MAX_ADR_FILES = 100
MAX_ADR_BYTES = 64 * 1024
MAX_FRAGMENTS = 5
MAX_FRAGMENT_LINES = 12
MIN_TERM_LENGTH = 3

_WORD = re.compile(r"[a-z0-9]+")
_STATUS_LINE = re.compile(r"^\W*status\W*:\s*(.+)$", re.IGNORECASE)
_STATUS_HEADING = re.compile(r"^#+\s*status\s*$", re.IGNORECASE)


class RepoError(Exception):
    """The given path cannot be reviewed."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, shell=False)


def resolve_repo(path: Path) -> Path:
    """Return the root of the Git repository containing `path`, which must have a commit."""
    if not path.is_dir():
        raise RepoError(f"{path} is not a directory")
    top = _git(path, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        raise RepoError(f"{path} is not a Git repository")
    root = Path(top.stdout.strip()).resolve()
    if _git(root, "rev-parse", "--verify", "--quiet", "HEAD").returncode != 0:
        raise RepoError(f"{root} has no commits yet; FutureYou compares changes against HEAD")
    return root


def git_diff(repo_root: Path) -> dict:
    """Changes against HEAD, including non-ignored untracked files, without modifying the index."""
    result = _git(repo_root, "diff", "--no-ext-diff", "--no-textconv", "--no-color", "HEAD")
    if result.returncode != 0:
        return {"status": "error", "message": result.stderr.strip()}
    diff = result.stdout
    listing = _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    if listing.returncode != 0:
        return {"status": "error", "message": listing.stderr.strip()}
    files = [name for name in listing.stdout.split("\0") if name and not (repo_root / name).is_symlink()]
    included = files[:MAX_UNTRACKED_FILES]
    for name in included:
        result = _git(repo_root, "diff", "--no-index", "--no-color", "--no-ext-diff", "--no-textconv", "--", "/dev/null", name)
        if result.returncode not in (0, 1):
            return {"status": "error", "message": result.stderr.strip()}
        diff += result.stdout
    truncated = len(diff) > MAX_DIFF_CHARS
    if truncated:
        cut = diff.rfind("\n", 0, MAX_DIFF_CHARS) + 1
        diff = diff[: cut or MAX_DIFF_CHARS]
    return {"status": "ok" if diff or files else "no_changes", "diff": diff, "truncated": truncated,
            "untracked": len(included), "untracked_skipped": len(files) - len(included)}


def _adr_status(lines: list[str]) -> str:
    for i, line in enumerate(lines):
        if match := _STATUS_LINE.match(line.strip()):
            return match.group(1).strip(" *_")
        if _STATUS_HEADING.match(line.strip()):
            following = next((l.strip() for l in lines[i + 1 :] if l.strip()), "")
            return following.strip(" *_") or "unknown"
    return "unknown"


def _fragments(path: str, lines: list[str], terms: set[str]) -> list[tuple[int, dict]]:
    status = _adr_status(lines)
    found = []
    covered_until = 0
    for i, line in enumerate(lines):
        if i < covered_until or not terms & set(_WORD.findall(line.lower())):
            continue
        start = max(0, i - 2)
        end = min(len(lines), start + MAX_FRAGMENT_LINES)
        covered_until = end
        window = set(_WORD.findall(" ".join(lines[start:end]).lower()))
        found.append((
            len(terms & window),
            {
                "path": path,
                "start_line": start + 1,
                "end_line": end,
                "adr_status": status,
                "excerpt": "\n".join(lines[start:end]),
            },
        ))
    return found


def search_decisions(repo_root: Path, query: str) -> dict:
    """Term search over the Markdown files directly under docs/adr/."""
    adr_dir = repo_root / "docs" / "adr"
    empty = {"fragments": [], "skipped": []}
    if adr_dir.is_symlink() or not adr_dir.is_dir():
        return {"status": "no_adrs", **empty}
    files = sorted(
        p for p in adr_dir.iterdir() if p.suffix == ".md" and not p.is_symlink() and p.is_file()
    )
    if not files:
        return {"status": "no_adrs", **empty}

    skipped = [
        {"path": p.relative_to(repo_root).as_posix(), "reason": f"over the {MAX_ADR_FILES}-file limit"}
        for p in files[MAX_ADR_FILES:]
    ]
    terms = {t for t in _WORD.findall(query.lower()) if len(t) >= MIN_TERM_LENGTH}
    scored = []
    for path in files[:MAX_ADR_FILES]:
        rel = path.relative_to(repo_root).as_posix()
        if path.stat().st_size > MAX_ADR_BYTES:
            skipped.append({"path": rel, "reason": "larger than 64 KiB"})
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            skipped.append({"path": rel, "reason": "unreadable or not UTF-8"})
            continue
        scored += _fragments(rel, text.splitlines(), terms)

    scored.sort(key=lambda item: (-item[0], item[1]["path"], item[1]["start_line"]))
    fragments = [fragment for _, fragment in scored[:MAX_FRAGMENTS]]
    if skipped:
        status = "partial"
    elif fragments:
        status = "ok"
    else:
        status = "no_matches"
    return {"status": status, "fragments": fragments, "skipped": skipped}
