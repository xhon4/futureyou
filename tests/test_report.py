import json

import pytest
from langchain_core.messages import ToolMessage

from futureyou import report, tools


def finding(**overrides):
    data = dict(type="Contradiction", title="title", why="why", suggestion="fix", change_path="x", change_start=1, change_end=1, decision_path="adr", decision_start=3, decision_end=4)
    return report.Finding(**(data | overrides))


def messages(diff):
    return [ToolMessage(json.dumps({"diff": diff, "status": "ok"}), name="get_git_diff", tool_call_id="1"), ToolMessage(json.dumps({"fragments": [{"path": "adr", "start_line": 3, "end_line": 5}]}), name="search_decisions", tool_call_id="2")]


DIFF = """diff --git a/x b/x
--- a/x
+++ b/x
@@ -1,2 +1,2 @@
 a
+b
@@ -8 +8 @@
-old
+new
"""


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"change_start": 3, "change_end": 8}, "lines not in the diff"),
    ({"change_path": "missing"}, "file not in the diff"),
    ({"decision_path": "missing"}, "ADR not consulted"),
    ({"decision_start": 5, "decision_end": 6}, "lines not in the consulted ADR excerpts"),
    ({"decision_path": None, "decision_start": None, "decision_end": None}, "missing decision citation"),
])
def test_invalid_evidence(overrides, reason):
    result = report.render([finding(**overrides)], messages(DIFF))
    assert "[Unverified]" in result
    assert f"✗ {reason}" in result
    assert "  ✓" in result


def test_multi_hunk_and_omitted_count():
    assert "[Contradiction]" in report.render([finding(change_start=8, change_end=8)], messages(DIFF))
    assert "lines not in the diff" in report.render([finding(change_start=8, change_end=9)], messages(DIFF))


def test_deleted_file():
    diff = "diff --git a/x b/x\n--- a/x\n+++ /dev/null\n@@ -3,2 +0,0 @@\n-a\n-b\n"
    assert "[Contradiction]" in report.render([finding(change_start=3, change_end=4)], messages(diff))


def test_untracked_file(sample_repo):
    (sample_repo / "x").write_text("one\ntwo\n")
    diff = tools.git_diff(sample_repo)["diff"]
    assert "[Contradiction]" in report.render([finding(change_start=1, change_end=2)], messages(diff))


def test_empty_and_five_limit():
    assert report.render([], messages(DIFF)) == "No concerns"
    assert report.render([finding()] * 6, messages(DIFF)).count("[Contradiction]") == 5


def test_regret_without_decision():
    text = report.render([finding(type="Potential regret", decision_path=None, decision_start=None, decision_end=None)], messages(DIFF))
    assert "[Potential regret]" in text
    assert "Decision:" not in text
