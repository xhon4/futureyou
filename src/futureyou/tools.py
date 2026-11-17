"""Read-only access to the reviewed repository: the Git diff and its ADRs.

These functions know nothing about the model or the CLI. They return plain
dicts so the agent can pass them to the model and tests can assert on them.
"""

import os
import re
import subprocess
from pathlib import Path

MAX_DIFF_CHARS = 60_000
# Separate from MAX_DIFF_CHARS so a test can shrink the combined diff without
# also skipping ordinary untracked files. Files this large are not read at all.
MAX_UNTRACKED_BYTES = MAX_DIFF_CHARS
MAX_UNTRACKED_FILES = 200
MAX_ADR_FILES = 100
MAX_ADR_BYTES = 64 * 1024
MAX_FRAGMENTS = 5
MAX_FRAGMENT_LINES = 12
MIN_TERM_LENGTH = 3
GIT_TIMEOUT_SECONDS = 60

_WORD = re.compile(r"[a-z0-9]+")
_STATUS_LINE = re.compile(r"^\W*status\W*:\s*(.+)$", re.IGNORECASE)
_STATUS_HEADING = re.compile(r"^#+\s*status\s*$", re.IGNORECASE)
_SUBMODULE_LINE = re.compile(r"^[ +-]Subproject commit [0-9a-f]+")

# `-c` outranks repository, environment, and system config. A reviewed repo
# therefore cannot re-enable quoting, mnemonic prefixes, clean/smudge filters,
# fsmonitor, or hooks for the read. `--no-ext-diff` also blocks GIT_EXTERNAL_DIFF.
_GIT_CONFIG = (
    "-c",
    "core.quotepath=false",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.autocrlf=false",
    "-c",
    "diff.mnemonicPrefix=false",
    "-c",
    "diff.noprefix=false",
    "-c",
    "diff.submodule=short",
    "-c",
    "diff.suppressBlankEmpty=false",
    "-c",
    "diff.external=",
    "-c",
    "color.ui=false",
    "-c",
    "submodule.recurse=false",
    "-c",
    "advice.detachedHead=false",
)
_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv", "--no-color")


class RepoError(Exception):
    """The given path cannot be reviewed."""


def _decode(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", errors="backslashreplace")


def _git(
    repo: Path, *args: str, config: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Skip optional index refresh so a read does not rewrite `.git/index`.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.pop("GIT_EXTERNAL_DIFF", None)
    try:
        completed = subprocess.run(
            ["git", *_GIT_CONFIG, *config, *args],
            cwd=repo,
            capture_output=True,
            env=env,
            shell=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise RepoError("git was not found on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise RepoError(
            f"git timed out after {GIT_TIMEOUT_SECONDS:g} seconds"
        ) from error
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        _decode(completed.stdout),
        _decode(completed.stderr),
    )


def _filter_overrides(repo: Path) -> tuple[str, ...] | None:
    """Disable every configured clean/smudge filter for one read.

    An empty command is a no-op, so `git diff` hashes worktree bytes instead
    of running `filter.<name>.clean` (which is how Git LFS writes `.git/lfs`).
    """
    listed = _git(repo, "config", "-z", "--name-only", "--get-regexp", r"^filter\.")
    if listed.returncode not in (0, 1):
        return None
    suffixes = (".clean", ".smudge", ".process", ".required")
    names: set[str] = set()
    for key in listed.stdout.split("\0"):
        if not key.startswith("filter."):
            continue
        name = key[len("filter.") :]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        else:
            continue
        if name:
            names.add(name)
    flags: list[str] = []
    for name in sorted(names):
        for suffix in ("clean", "smudge", "process"):
            flags.extend(["-c", f"filter.{name}.{suffix}="])
        flags.extend(["-c", f"filter.{name}.required=false"])
    return tuple(flags)


def resolve_repo(path: Path) -> Path:
    """Return the root of the Git repository containing `path`, which must have a commit."""
    if not path.is_dir():
        raise RepoError(f"{path} is not a directory")
    top = _git(path, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        raise RepoError(f"{path} is not a Git repository")
    root = Path(top.stdout.strip()).resolve()
    if _git(root, "rev-parse", "--verify", "--quiet", "HEAD").returncode != 0:
        raise RepoError(
            f"{root} has no commits yet; FutureYou compares changes against HEAD"
        )
    return root


def _git_error(result: subprocess.CompletedProcess[str]) -> dict:
    return {"status": "error", "message": result.stderr.strip()}


def git_diff(repo_root: Path) -> dict:
    """Staged and unstaged changes against HEAD, plus non-ignored untracked files.

    The index is not modified. Clean/smudge filters and hooks are not run.
    Nested untracked Git repositories and submodule contents are not read;
    the result says so instead of counting them as included.
    """
    overrides = _filter_overrides(repo_root)
    if overrides is None:
        return {"status": "error", "message": "could not read Git filter configuration"}

    staged = _git(repo_root, "diff", "--cached", *_DIFF_FLAGS, config=overrides)
    if staged.returncode != 0:
        return _git_error(staged)
    unstaged = _git(repo_root, "diff", *_DIFF_FLAGS, config=overrides)
    if unstaged.returncode != 0:
        return _git_error(unstaged)
    tracked = staged.stdout + unstaged.stdout

    listing = _git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        config=overrides,
    )
    if listing.returncode != 0:
        return _git_error(listing)

    nested: list[str] = []
    candidates: list[str] = []
    for name in listing.stdout.split("\0"):
        if not name:
            continue
        path = repo_root / name
        if path.is_symlink():
            continue
        if path.is_dir():
            nested.append(name.rstrip("/"))
            continue
        candidates.append(name)

    included = candidates[:MAX_UNTRACKED_FILES]
    untracked_skipped = len(candidates) - len(included)
    parts: list[str] = []
    oversized = 0
    for name in included:
        path = repo_root / name
        try:
            size = path.stat().st_size
        except OSError as error:
            return {"status": "error", "message": f"open({name!r}): {error}"}
        if size >= MAX_UNTRACKED_BYTES:
            oversized += 1
            continue
        result = _git(
            repo_root,
            "diff",
            "--no-index",
            *_DIFF_FLAGS,
            "--",
            "/dev/null",
            name,
            config=overrides,
        )
        if result.returncode not in (0, 1):
            return _git_error(result)
        parts.append(result.stdout)

    diff = tracked + "".join(parts)
    truncated = len(diff) > MAX_DIFF_CHARS
    untracked_cut = 0
    retained = False
    if truncated:
        cut = diff.rfind("\n", 0, MAX_DIFF_CHARS) + 1
        kept = cut or MAX_DIFF_CHARS
        retained = kept > len(tracked) and bool(parts)
        position = len(tracked)
        for part in parts:
            if position >= kept or position + len(part) > kept:
                untracked_cut += 1
            position += len(part)
        diff = diff[:kept]
    submodules = sum(1 for line in diff.split("\n") if _SUBMODULE_LINE.match(line))
    binary_files = sum(
        1
        for line in diff.split("\n")
        if line.startswith("Binary files ") and line.endswith(" differ")
    )
    return {
        "status": "ok" if diff else "no_changes",
        "diff": diff,
        "truncated": truncated,
        "untracked": len(parts),
        "untracked_skipped": untracked_skipped,
        "nested_skipped": len(nested),
        "oversized_untracked": oversized,
        "untracked_cut": untracked_cut,
        "untracked_retained": retained,
        "submodules": submodules,
        "binary_files": binary_files,
    }


def _adr_status(lines: list[str]) -> str:
    for i, line in enumerate(lines):
        if match := _STATUS_LINE.match(line.strip()):
            return match.group(1).strip(" *_")
        if _STATUS_HEADING.match(line.strip()):
            following = next(
                (item.strip() for item in lines[i + 1 :] if item.strip()), ""
            )
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
        found.append(
            (
                len(terms & window),
                {
                    "path": path,
                    "start_line": start + 1,
                    "end_line": end,
                    "adr_status": status,
                    "excerpt": "\n".join(lines[start:end]),
                },
            )
        )
    return found


def search_decisions(repo_root: Path, query: str) -> dict:
    """Term search over the Markdown files directly under docs/adr/."""
    adr_dir = repo_root / "docs" / "adr"
    empty = {"fragments": [], "skipped": []}
    if adr_dir.is_symlink() or not adr_dir.is_dir():
        return {"status": "no_adrs", **empty}
    files = sorted(
        p
        for p in adr_dir.iterdir()
        if p.suffix == ".md" and not p.is_symlink() and p.is_file()
    )
    if not files:
        return {"status": "no_adrs", **empty}

    skipped = [
        {
            "path": p.relative_to(repo_root).as_posix(),
            "reason": f"over the {MAX_ADR_FILES}-file limit",
        }
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
