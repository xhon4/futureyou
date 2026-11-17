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
_EMPTY_BLOB = "e69de29"
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
    token = text.split("\t", 1)[0].strip()
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


def _split_git_paths(rest: str) -> tuple[str, str] | None:
    rest = rest.strip()
    if rest.startswith('"'):
        first, rest = _take_quoted(rest)
        second, _remainder = _take_quoted(rest.lstrip())
        if first is None or second is None:
            return None
        return first, second
    # Unquoted. A space can belong to the path, so accept the split where both
    # sides wear a diff prefix, or where noprefix repeated the same path.
    for index in range(len(rest) - 1, 0, -1):
        if rest[index] != " ":
            continue
        left, right = rest[:index], rest[index + 1 :]
        if left == right and left:
            return left, right
        if _strip_prefix(left) != left and _strip_prefix(right) != right:
            return left, right
    return None


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
            parsed = _split_git_paths(raw[len("diff --git ") :])
            in_hunk = False
            deleted = False
            new_file_mode = False
            empty_blob = False
            old_path = None
            path = None
            if parsed:
                old_raw, new_raw = parsed
                old_path = _strip_prefix(old_raw)
                new_path = _strip_prefix(new_raw)
                deleted = new_path == "/dev/null"
                use(old_path if deleted else new_path)
            continue
        if path is None and old_path is None:
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
        if not in_hunk and raw.startswith("new file mode "):
            new_file_mode = True
            continue
        if not in_hunk and raw.startswith("index "):
            parts = raw.split()
            if len(parts) >= 2 and ".." in parts[1]:
                empty_blob = parts[1].split("..", 1)[1].startswith(_EMPTY_BLOB)
            continue
        match = _HUNK.match(raw)
        if match and path is not None:
            in_hunk = True
            old_line = int(match.group(1))
            new_line = int(match.group(3))
            continue
        if not in_hunk or path is None:
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
