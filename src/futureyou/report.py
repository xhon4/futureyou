"""Structured findings and deterministic checks against the evidence actually read."""

import json
import re
from typing import Literal

from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel, Field


class Finding(BaseModel):
    type: Literal["Contradiction", "Potential regret"] = Field(
        description="Finding category; contradictions require a current-policy ADR"
    )
    title: str = Field(description="Short finding title")
    why: str = Field(
        description="Explanation; identify inferences for potential regrets"
    )
    suggestion: str = Field(description="Suggested next step")
    change_path: str = Field(description="Changed file path shown in the diff")
    change_start: int = Field(description="First cited line in the diff")
    change_end: int = Field(description="Last cited line in the same diff hunk")
    decision_path: str | None = Field(
        default=None, description="Consulted ADR path, required for contradictions"
    )
    decision_start: int | None = Field(
        default=None, description="First line in a consulted ADR excerpt"
    )
    decision_end: int | None = Field(
        default=None, description="Last line in the same consulted ADR excerpt"
    )


class Report(BaseModel):
    findings: list[Finding] = Field(
        description="Up to five findings, or an empty list when there are no concerns"
    )


def tool_results(messages: list[BaseMessage], name: str) -> list[dict]:
    results = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == name:
            try:
                value = json.loads(message.content)
                if isinstance(value, dict):
                    results.append(value)
            except (TypeError, json.JSONDecodeError):
                continue
    return results


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_PREFIXES = ("a/", "b/", "c/", "i/", "w/")
_EMPTY_BLOB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
_OCTAL = set("01234567")
_GIT_ESCAPES = {
    "a": b"\a",
    "b": b"\b",
    "f": b"\f",
    "n": b"\n",
    "r": b"\r",
    "t": b"\t",
    "v": b"\v",
    "\\": b"\\",
    '"': b'"',
}


def _unquote_git(token: str) -> str:
    """Decode a Git C-quoted path, including octal bytes, back to text."""
    if len(token) < 2 or not token.startswith('"'):
        return token
    body = token[1:-1] if token.endswith('"') else token[1:]
    raw = bytearray()
    index = 0
    while index < len(body):
        char = body[index]
        if char != "\\":
            raw.extend(char.encode("utf-8"))
            index += 1
            continue
        index += 1
        if index >= len(body):
            break
        escaped = body[index]
        if escaped in _GIT_ESCAPES:
            raw.extend(_GIT_ESCAPES[escaped])
            index += 1
        elif escaped in _OCTAL:
            end = index
            while end < len(body) and end < index + 3 and body[end] in _OCTAL:
                end += 1
            raw.append(int(body[index:end], 8))
            index = end
        else:
            raw.extend(escaped.encode("utf-8"))
            index += 1
    return raw.decode("utf-8", errors="surrogateescape")


def _strip_prefix(path: str) -> str:
    if path == "/dev/null":
        return path
    for prefix in _DIFF_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _header_path(text: str) -> str:
    # Git separates an optional timestamp with a tab. Trailing spaces are part
    # of the path; strip() would cite the trimmed name instead.
    token = text.split("\t", 1)[0]
    if token.endswith("\r"):
        token = token[:-1]
    if token.startswith('"'):
        return _strip_prefix(_unquote_git(token))
    return _strip_prefix(token)


def _take_quoted(text: str) -> tuple[str | None, str]:
    if not text.startswith('"'):
        return None, text
    index = 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == '"':
            return _unquote_git(text[: index + 1]), text[index + 1 :]
        index += 1
    return _unquote_git(text), ""


def _plain_git_path(token: str) -> str:
    token = token.split("\t", 1)[0]
    if token.endswith("\r"):
        token = token[:-1]
    if token.startswith('"'):
        return _unquote_git(token)
    return token


def _path_body(token: str) -> str:
    token = token.split("\t", 1)[0]
    if token.endswith("\r"):
        token = token[:-1]
    if token.startswith('"'):
        token = _unquote_git(token)
    return _strip_prefix(token)


def _split_git_paths(rest: str) -> tuple[str, str] | None:
    """Split ``diff --git`` into the two path tokens.

    For an ordinary section Git prints ``a/<path> b/<path>`` with the same
    path on both sides. The split is the space where those two paths are
    equal after unquoting. A rename or copy has different paths there; the
    caller uses ``rename to`` / ``copy to`` instead. Two quoted tokens are
    already separated, so they are not a guess. Any other header is rejected.
    """
    if rest.endswith("\r"):
        rest = rest[:-1]
    if rest.startswith('"'):
        first, remainder = _take_quoted(rest)
        second, leftover = _take_quoted(remainder.lstrip(" "))
        if first is None or second is None or leftover or not first or not second:
            return None
        return first, second
    matches: list[tuple[str, str]] = []
    for index, char in enumerate(rest):
        if char != " ":
            continue
        left, right = rest[:index], rest[index + 1 :]
        if left and right and _path_body(left) == _path_body(right):
            matches.append((left, right))
    if len(matches) == 1:
        return matches[0]
    return None


def _header_endpoints(rest: str) -> tuple[str | None, str | None]:
    parsed = _split_git_paths(rest)
    if not parsed:
        return None, None
    return _path_body(parsed[0]), _path_body(parsed[1])


def _diff_lines(diff: str) -> list[str]:
    """Split a diff on LF only.

    ``str.splitlines`` also breaks on CR, VT, FF and Unicode line separators.
    Those bytes can appear inside a changed line, and treating them as line
    breaks lets a forged hunk header or ``+++`` marker retarget the parser.
    A trailing CR is the CRLF separator, not part of the line.
    """
    lines = diff.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def _collapse(numbers: set[int]) -> list[tuple[int, int]]:
    if not numbers:
        return []
    ordered = sorted(numbers)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for number in ordered[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append((start, previous))
        start = previous = number
    ranges.append((start, previous))
    return ranges


def diff_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Line numbers present in the diff body, keyed by repository path.

    Header lines are recognized only before the first hunk, so a body line
    that starts with ``+++`` or ``---`` cannot retarget the file. Paths are
    unquoted and a single ``a/`` ``b/`` ``c/`` ``i/`` or ``w/`` prefix is
    removed. Ranges come from the lines still in the text, not from the hunk
    header's length, so a truncated hunk cannot verify a line that was cut.
    An empty added file has no hunk; it is citable as line 1.
    """
    lines: dict[str, set[int]] = {}
    path: str | None = None
    old_path: str | None = None
    deleted = False
    in_hunk = False
    in_section = False
    old_line = 0
    new_line = 0
    new_file_mode = False
    empty_blob = False

    def use(candidate: str | None) -> None:
        nonlocal path
        if not candidate or candidate == "/dev/null":
            return
        if path and path != candidate and not lines.get(path):
            lines.pop(path, None)
        path = candidate
        lines.setdefault(path, set())

    def close() -> None:
        if path and new_file_mode and empty_blob and not lines.get(path):
            lines[path].add(1)

    for raw in _diff_lines(diff):
        if raw.startswith("diff --git "):
            close()
            old_raw, new_raw = _header_endpoints(raw[len("diff --git ") :])
            in_hunk = False
            in_section = True
            deleted = new_raw == "/dev/null"
            new_file_mode = False
            empty_blob = False
            old_path = None if old_raw in (None, "/dev/null") else old_raw
            path = None
            use(old_path if deleted else new_raw)
            continue
        if not in_section:
            continue
        if not in_hunk and raw.startswith("--- "):
            value = _header_path(raw[4:])
            if value != "/dev/null":
                old_path = value
            continue
        if not in_hunk and raw.startswith("+++ "):
            value = _header_path(raw[4:])
            deleted = value == "/dev/null"
            use(old_path if deleted else value)
            continue
        if not in_hunk and raw.startswith("rename from "):
            old_path = _plain_git_path(raw[len("rename from ") :])
            continue
        if not in_hunk and raw.startswith("rename to "):
            use(_plain_git_path(raw[len("rename to ") :]))
            continue
        if not in_hunk and raw.startswith("copy from "):
            old_path = _plain_git_path(raw[len("copy from ") :])
            continue
        if not in_hunk and raw.startswith("copy to "):
            use(_plain_git_path(raw[len("copy to ") :]))
            continue
        if not in_hunk and raw.startswith("new file mode "):
            new_file_mode = True
            continue
        if not in_hunk and raw.startswith("index "):
            parts = raw.split()
            if len(parts) >= 2 and ".." in parts[1]:
                abbrev = parts[1].split("..", 1)[1]
                empty_blob = len(abbrev) >= 4 and _EMPTY_BLOB.startswith(abbrev)
            continue
        match = _HUNK.match(raw)
        if match and path is not None:
            in_hunk = True
            old_line = int(match.group(1))
            new_line = int(match.group(3))
            continue
        if not in_hunk or path is None:
            continue
        if raw == "":
            # diff.suppressBlankEmpty emits a blank context line as an empty
            # string. It is still context; it must not end the hunk.
            bucket = lines.setdefault(path, set())
            if old_line >= 1:
                bucket.add(old_line)
            if not deleted and new_line >= 1:
                bucket.add(new_line)
            old_line += 1
            new_line += 1
            continue
        if raw.startswith("\\"):
            continue
        bucket = lines.setdefault(path, set())
        if raw.startswith("+"):
            if new_line >= 1:
                bucket.add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            if old_line >= 1:
                bucket.add(old_line)
            old_line += 1
        elif raw.startswith(" "):
            if old_line >= 1:
                bucket.add(old_line)
            if not deleted and new_line >= 1:
                bucket.add(new_line)
            old_line += 1
            new_line += 1
        else:
            in_hunk = False
    close()
    return {name: _collapse(numbers) for name, numbers in lines.items()}


def binary_paths(diff: str) -> set[str]:
    """Paths Git presented as binary, one entry per path.

    The path is the one established by ``diff --git`` (or ``rename to`` /
    ``copy to``). The ``Binary files`` line is only a notice; its own text
    is not parsed, because ``/dev/null`` and a path containing `` and b/``
    do not split on the first `` and ``.
    """
    found: set[str] = set()
    path: str | None = None
    in_section = False
    for raw in _diff_lines(diff):
        if raw.startswith("diff --git "):
            in_section = True
            old_raw, new_raw = _header_endpoints(raw[len("diff --git ") :])
            if new_raw == "/dev/null":
                path = None if old_raw in (None, "/dev/null") else old_raw
            else:
                path = new_raw
            continue
        if not in_section:
            continue
        if raw.startswith("rename to "):
            path = _plain_git_path(raw[len("rename to ") :])
            continue
        if raw.startswith("copy to "):
            path = _plain_git_path(raw[len("copy to ") :])
            continue
        if path and raw.startswith("Binary files ") and raw.endswith(" differ"):
            found.add(path)
    return found


def validate(
    finding: Finding, messages: list[BaseMessage]
) -> tuple[str | None, str | None]:
    diffs = tool_results(messages, "get_git_diff")
    files = diff_ranges(diffs[-1].get("diff", "")) if diffs else {}
    if finding.change_path not in files:
        change_error = "file not in the diff"
    elif not (
        1 <= finding.change_start <= finding.change_end
        and any(
            start <= finding.change_start <= finding.change_end <= end
            for start, end in files[finding.change_path]
        )
    ):
        change_error = "lines not in the diff"
    else:
        change_error = None
    fragments = [
        fragment
        for result in tool_results(messages, "search_decisions")
        for fragment in result.get("fragments", [])
    ]
    if finding.decision_path is None:
        decision_error = (
            "missing decision citation"
            if finding.type == "Contradiction"
            or finding.decision_start is not None
            or finding.decision_end is not None
            else None
        )
    else:
        matching = [f for f in fragments if f.get("path") == finding.decision_path]
        if not matching:
            decision_error = "ADR not consulted"
        elif (
            finding.decision_start is None
            or finding.decision_end is None
            or not any(
                f["start_line"]
                <= finding.decision_start
                <= finding.decision_end
                <= f["end_line"]
                and finding.decision_start >= 1
                for f in matching
            )
        ):
            decision_error = "lines not in the consulted ADR excerpts"
        else:
            decision_error = None
    return change_error, decision_error


def render(findings: list[Finding], messages: list[BaseMessage]) -> str:
    findings = findings[:5]
    if not findings:
        return "No concerns"
    blocks = [f"{len(findings)} concern(s)"]
    for finding in findings:
        change_error, decision_error = validate(finding, messages)
        invalid = bool(change_error or decision_error)

        def mark(error: str | None, invalid: bool = invalid) -> str:
            return f"  ✗ {error}" if error else "  ✓" if invalid else ""

        lines = [
            f"[{'Unverified' if invalid else finding.type}] {finding.title}",
            f"Why: {finding.why}",
            f"Change: {finding.change_path}:{finding.change_start}-{finding.change_end}{mark(change_error)}",
        ]
        if finding.decision_path is not None:
            lines.append(
                f"Decision: {finding.decision_path}:{finding.decision_start}-{finding.decision_end}{mark(decision_error)}"
            )
        elif decision_error:
            lines.append(f"Decision: none{mark(decision_error)}")
        lines.append(f"Suggestion: {finding.suggestion}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
