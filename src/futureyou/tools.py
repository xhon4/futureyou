"""Read-only access to the reviewed repository: the Git diff and its ADRs.

These functions know nothing about the model or the CLI. They return plain
dicts so the agent can pass them to the model and tests can assert on them.
"""

import os
import re
import signal
import subprocess
from pathlib import Path

from futureyou.report import binary_paths

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
# A grandchild that called setsid is outside the process group and can hold
# the stdout pipe. The read after killpg stops on its own deadline.
_KILL_DRAIN_SECONDS = 1

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
# These choose which repository Git reads. The caller passes the work tree
# as cwd; an inherited value would review a different repo. Config isolation
# (GIT_CONFIG_GLOBAL, GIT_CONFIG_NOSYSTEM) is left in place.
_REPO_LOCATING_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
)


class RepoError(Exception):
    """The given path cannot be reviewed."""


def _decode(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", errors="backslashreplace")


def _append_config(
    env: dict[bytes, bytes], pairs: tuple[tuple[bytes, bytes], ...]
) -> None:
    """Add config pairs after any GIT_CONFIG_COUNT entries already in the environment.

    ``git -c key=value`` splits on the first ``=``, so a driver named ``foo=bar``
    cannot be disabled that way. ``GIT_CONFIG_KEY_n`` carries the key bytes
    unchanged, including ``=`` and bytes that are not UTF-8. Later entries
    override earlier ones, and the entries already in the environment stay.
    """
    if not pairs:
        return
    raw = env.get(b"GIT_CONFIG_COUNT", b"0").strip()
    try:
        count = int(raw) if raw else 0
    except ValueError:
        count = 0
    if count < 0:
        count = 0
    for key, value in pairs:
        env[f"GIT_CONFIG_KEY_{count}".encode("ascii")] = key
        env[f"GIT_CONFIG_VALUE_{count}".encode("ascii")] = value
        count += 1
    env[b"GIT_CONFIG_COUNT"] = str(count).encode("ascii")


def _reap(process: subprocess.Popen) -> None:
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass
    drain = min(GIT_TIMEOUT_SECONDS, _KILL_DRAIN_SECONDS)
    try:
        process.communicate(timeout=drain)
    except subprocess.TimeoutExpired:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        try:
            process.wait(timeout=drain)
        except subprocess.TimeoutExpired:
            pass


def _git(
    repo: Path,
    *args: str,
    config: tuple[tuple[bytes, bytes], ...] = (),
    binary: bool = False,
) -> subprocess.CompletedProcess:
    env = os.environb.copy()
    # Skip optional index refresh so a read does not rewrite `.git/index`.
    env[b"GIT_OPTIONAL_LOCKS"] = b"0"
    env.pop(b"GIT_EXTERNAL_DIFF", None)
    for name in _REPO_LOCATING_ENV:
        env.pop(name.encode("ascii"), None)
    _append_config(env, config)
    command = ["git", *_GIT_CONFIG, *args]
    try:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            shell=False,
            start_new_session=True,
        )
    except FileNotFoundError as error:
        raise RepoError("git was not found on PATH") from error
    except OSError as error:
        raise RepoError(f"could not run git: {error}") from error
    try:
        stdout, stderr = process.communicate(timeout=GIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        _reap(process)
        raise RepoError(
            f"git timed out after {GIT_TIMEOUT_SECONDS:g} seconds"
        ) from error
    if binary:
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        _decode(stdout),
        _decode(stderr),
    )


_FILTER_SUFFIXES = (b".clean", b".smudge", b".process", b".required")


def _filter_overrides(repo: Path) -> tuple[tuple[bytes, bytes], ...] | None:
    """Disable every configured clean/smudge filter for one read.

    An empty command is a no-op, so `git diff` hashes worktree bytes instead
    of running `filter.<name>.clean` (which is how Git LFS writes `.git/lfs`).
    The driver name is the raw bytes between ``filter.`` and the last suffix,
    so ``foo.bar``, ``foo=bar``, and a name containing ``0xff`` all match.
    """
    listed = _git(
        repo,
        "config",
        "-z",
        "--name-only",
        "--get-regexp",
        r"^filter\.",
        binary=True,
    )
    if listed.returncode not in (0, 1):
        return None
    names: set[bytes] = set()
    for key in (listed.stdout or b"").split(b"\0"):
        if not key.startswith(b"filter."):
            continue
        body = key[len(b"filter.") :]
        for suffix in _FILTER_SUFFIXES:
            if body.endswith(suffix) and len(body) > len(suffix):
                names.add(body[: -len(suffix)])
                break
    pairs: list[tuple[bytes, bytes]] = []
    for name in sorted(names):
        prefix = b"filter." + name + b"."
        for suffix in (b"clean", b"smudge", b"process"):
            pairs.append((prefix + suffix, b""))
        pairs.append((prefix + b"required", b"false"))
    return tuple(pairs)


def resolve_repo(path: Path) -> Path:
    """Return the root of the Git repository containing `path`, which must have a commit."""
    if not path.is_dir():
        raise RepoError(f"{path} is not a directory")
    top = _git(path, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        detail = top.stderr.strip()
        message = f"{path} is not a Git repository"
        raise RepoError(f"{message}: {detail}" if detail else message)
    root = Path(top.stdout.strip()).resolve()
    head = _git(root, "rev-parse", "--verify", "--quiet", "HEAD")
    if head.returncode != 0:
        detail = head.stderr.strip()
        message = f"{root} has no commits yet; FutureYou compares changes against HEAD"
        raise RepoError(f"{message}: {detail}" if detail else message)
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
        binary=True,
    )
    if listing.returncode != 0:
        return {"status": "error", "message": _decode(listing.stderr).strip()}

    nested: list[str] = []
    candidates: list[str] = []
    undecodable = 0
    for raw_name in listing.stdout.split(b"\0"):
        if not raw_name:
            continue
        try:
            name = os.fsdecode(raw_name)
        except UnicodeDecodeError:
            undecodable += 1
            continue
        path = repo_root / name
        try:
            path.lstat()
        except OSError:
            undecodable += 1
            continue
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
    binary_files = len(binary_paths(diff))
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
        "undecodable_untracked": undecodable,
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


def _inside_repo(repo_root: Path, path: Path) -> bool:
    root = repo_root.resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def search_decisions(repo_root: Path, query: str) -> dict:
    """Term search over the Markdown files directly under docs/adr/."""
    adr_dir = repo_root / "docs" / "adr"
    empty = {"fragments": [], "skipped": []}
    if (
        adr_dir.is_symlink()
        or not adr_dir.is_dir()
        or not _inside_repo(repo_root, adr_dir)
    ):
        return {"status": "no_adrs", **empty}
    files = sorted(
        p
        for p in adr_dir.iterdir()
        if p.suffix == ".md"
        and not p.is_symlink()
        and p.is_file()
        and _inside_repo(repo_root, p)
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
