"""Structured findings and deterministic checks against the evidence actually read."""

import json
import re
import shlex
from typing import Literal

from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel, Field


class Finding(BaseModel):
    type: Literal["Contradiction", "Potential regret"] = Field(description="Finding category; contradictions require a current-policy ADR")
    title: str = Field(description="Short finding title")
    why: str = Field(description="Explanation; identify inferences for potential regrets")
    suggestion: str = Field(description="Suggested next step")
    change_path: str = Field(description="Changed file path shown in the diff")
    change_start: int = Field(description="First cited line in the diff")
    change_end: int = Field(description="Last cited line in the same diff hunk")
    decision_path: str | None = Field(default=None, description="Consulted ADR path, required for contradictions")
    decision_start: int | None = Field(default=None, description="First line in a consulted ADR excerpt")
    decision_end: int | None = Field(default=None, description="Last line in the same consulted ADR excerpt")


class Report(BaseModel):
    findings: list[Finding] = Field(description="Up to five findings, or an empty list when there are no concerns")


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


def diff_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Collect ranges per file, using old lines only for deleted files."""
    files: dict[str, list[tuple[int, int]]] = {}
    old = new = None
    deleted = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            old = new = None
            deleted = False
            try:
                parts = shlex.split(line)
                old, new = parts[2].removeprefix("a/"), parts[3].removeprefix("b/")
                files.setdefault(new, [])
            except (ValueError, IndexError):
                pass
        elif line.startswith("--- ") and old is not None:
            value = line[4:].split("\t", 1)[0]
            if value != "/dev/null":
                old = value.removeprefix("a/").strip('"')
        elif line.startswith("+++ ") and new is not None:
            value = line[4:].split("\t", 1)[0]
            deleted = value == "/dev/null"
            if deleted:
                files.pop(new, None)
                files.setdefault(old, [])
            else:
                new = value.removeprefix("b/").strip('"')
                files.setdefault(new, [])
        elif (match := _HUNK.match(line)) and new is not None:
            start = int(match.group(1 if deleted else 3))
            count = int(match.group(2 if deleted else 4) or "1")
            if count:
                files.setdefault(old if deleted else new, []).append((start, start + count - 1))
    return files


def validate(finding: Finding, messages: list[BaseMessage]) -> tuple[str | None, str | None]:
    diffs = tool_results(messages, "get_git_diff")
    files = diff_ranges(diffs[-1].get("diff", "")) if diffs else {}
    if finding.change_path not in files:
        change_error = "file not in the diff"
    elif not (1 <= finding.change_start <= finding.change_end and any(
        start <= finding.change_start <= finding.change_end <= end for start, end in files[finding.change_path]
    )):
        change_error = "lines not in the diff"
    else:
        change_error = None
    fragments = [fragment for result in tool_results(messages, "search_decisions") for fragment in result.get("fragments", [])]
    if finding.decision_path is None:
        decision_error = "missing decision citation" if finding.type == "Contradiction" or finding.decision_start is not None or finding.decision_end is not None else None
    else:
        matching = [f for f in fragments if f.get("path") == finding.decision_path]
        if not matching:
            decision_error = "ADR not consulted"
        elif finding.decision_start is None or finding.decision_end is None or not any(
            f["start_line"] <= finding.decision_start <= finding.decision_end <= f["end_line"] and finding.decision_start >= 1 for f in matching
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
        def mark(error: str | None) -> str:
            return f"  ✗ {error}" if error else "  ✓" if invalid else ""
        lines = [f"[{'Unverified' if invalid else finding.type}] {finding.title}", f"Why: {finding.why}",
                 f"Change: {finding.change_path}:{finding.change_start}-{finding.change_end}{mark(change_error)}"]
        if finding.decision_path is not None:
            lines.append(f"Decision: {finding.decision_path}:{finding.decision_start}-{finding.decision_end}{mark(decision_error)}")
        elif decision_error:
            lines.append(f"Decision: none{mark(decision_error)}")
        lines.append(f"Suggestion: {finding.suggestion}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
