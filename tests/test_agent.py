from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from futureyou import agent, cli, report

FINDING = dict(type="Contradiction", title="Session token stored on client", why="The change persists the session token in localStorage.", change_path="src/auth.ts", change_start=17, change_end=18, decision_path="docs/adr/003-authentication.md", decision_start=15, decision_end=17, suggestion="Keep server-managed sessions or revise ADR-003.")

REPORT = (
    "1 concern(s)\n\n"
    "[Contradiction] Session token stored on client\n"
    "Why: The change persists the session token in localStorage.\n"
    "Change: src/auth.ts:17-18\n"
    "Decision: docs/adr/003-authentication.md:15-17\n"
    "Suggestion: Keep server-managed sessions or revise ADR-003."
)


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that supports bind_tools and records what it was sent."""

    seen: list = []
    structured: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda
        def respond(messages):
            self.seen.append(list(messages))
            return self.structured.pop(0)
        return RunnableLambda(respond)

    def invoke(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super().invoke(messages, *args, **kwargs)


@pytest.fixture
def models(monkeypatch, tmp_path: Path):
    """Each CLI run gets a fresh model, like a new process; replies are queued per run."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    created: list[ScriptedModel] = []
    queue: list[list[AIMessage]] = []

    def load_model():
        replies = queue.pop(0)
        structured = [item for item in replies if isinstance(item, report.Report)]
        model = ScriptedModel(messages=iter(item for item in replies if isinstance(item, AIMessage)), seen=[],
                              structured=structured or [report.Report(findings=[report.Finding(**FINDING)])])
        created.append(model)
        return model

    monkeypatch.setattr(agent, "load_model", load_model)
    return queue, created


def review_replies() -> list[AIMessage]:
    return [
        AIMessage("", tool_calls=[{"name": "search_decisions", "args": {"query": "session token"}, "id": "call-2"}]),
        AIMessage(REPORT),
    ]


def test_ask_recovers_the_review_after_reopening_sqlite(models, contradicting_repo: Path, tmp_path, capsys):
    queue, created = models
    queue += [review_replies(), [AIMessage("Remove the localStorage token first.")]]

    assert cli.main(["review", str(contradicting_repo), "--thread", "auth-review"]) == 0
    review_out = capsys.readouterr().out
    assert REPORT in review_out
    assert "Thread: auth-review" in review_out

    assert cli.main(["ask", "--thread", "auth-review", "What should I change first?"]) == 0
    assert capsys.readouterr().out == "Remove the localStorage token first.\n"

    ask_input = created[1].seen[0]
    tool_names = [m.name for m in ask_input if isinstance(m, ToolMessage)]
    assert tool_names == ["get_git_diff", "search_decisions"]
    assert ask_input[-1].content == "What should I change first?"
    assert ask_input[-2].content == REPORT
    assert (tmp_path / "data" / "futureyou" / "checkpoints.sqlite").is_file()


def test_review_generates_and_prints_a_thread_id(models, contradicting_repo: Path, capsys):
    queue, _ = models
    queue.append(review_replies())

    assert cli.main(["review", str(contradicting_repo)]) == 0

    thread_lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("Thread: ")]
    assert len(thread_lines) == 1 and len(thread_lines[0].removeprefix("Thread: ")) == 8


def test_review_refuses_an_existing_thread(models, contradicting_repo: Path, capsys):
    queue, created = models
    queue += [review_replies(), review_replies()]
    cli.main(["review", str(contradicting_repo), "--thread", "auth-review"])
    capsys.readouterr()

    assert cli.main(["review", str(contradicting_repo), "--thread", "auth-review"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert err == "futureyou: thread 'auth-review' already exists; use another --thread\n"
    assert len(created) == 1


def test_ask_requires_an_existing_thread(models, capsys):
    queue, created = models
    queue.append([AIMessage("unused")])

    assert cli.main(["ask", "--thread", "missing", "Anything?"]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert err == "futureyou: thread 'missing' does not exist; run futureyou review first\n"
    assert created == []


def _snapshot(repo: Path) -> dict:
    return {p.relative_to(repo).as_posix(): p.read_bytes() for p in sorted(repo.rglob("*")) if p.is_file()}


def test_review_routes_tools_and_reports_the_known_contradiction(models, contradicting_repo: Path, capsys):
    queue, created = models
    queue.append(review_replies())
    before = _snapshot(contradicting_repo)

    assert cli.main(["review", str(contradicting_repo), "--thread", "known"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("FutureYou Report\n\n" + REPORT + "\n\n")
    assert "Review scope: changes against HEAD, untracked files included; no truncation; ADRs searched.\n" in out
    final_input = created[0].seen[-1]
    diff, adrs = [m for m in final_input if isinstance(m, ToolMessage)]
    assert "+  localStorage.setItem(TOKEN_KEY, token);" in diff.content
    assert "docs/adr/003-authentication.md" in adrs.content
    assert _snapshot(contradicting_repo) == before


@pytest.mark.parametrize(("overrides", "reason"), [
    ({"change_start": 999, "change_end": 999}, "lines not in the diff"),
    ({"change_path": "missing"}, "file not in the diff"),
    ({"decision_path": "missing"}, "ADR not consulted"),
    ({"decision_start": 999, "decision_end": 999}, "lines not in the consulted ADR excerpts"),
    ({"decision_path": None, "decision_start": None, "decision_end": None}, "missing decision citation"),
])
def test_review_marks_invalid_citations(models, contradicting_repo, capsys, overrides, reason):
    queue, _ = models
    queue.append(review_replies() + [report.Report(findings=[report.Finding(**(FINDING | overrides))])])
    assert cli.main(["review", str(contradicting_repo), "--thread", "invalid"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "[Unverified]" in out and f"✗ {reason}" in out
    assert "; 1 finding(s) with invalid evidence.\nThread: invalid\n" in out


def test_review_empty_structured_findings(models, contradicting_repo, capsys):
    queue, _ = models
    queue.append(review_replies() + [report.Report(findings=[])])
    assert cli.main(["review", str(contradicting_repo), "--thread", "empty"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert out == "FutureYou Report\n\nNo concerns\n\nReview scope: changes against HEAD, untracked files included; no truncation; ADRs searched.\nThread: empty\n"


def test_review_without_changes_answers_without_the_model(models, sample_repo: Path, capsys):
    queue, created = models
    queue.append([])

    assert cli.main(["review", str(sample_repo), "--thread", "clean"]) == 0

    assert capsys.readouterr().out == (
        "FutureYou Report\n\nNo concerns\n\n"
        "Review scope: no changes against HEAD; ADRs not searched.\nThread: clean\n"
    )
    assert created[0].seen == []


def test_review_reads_the_diff_before_the_model_even_if_it_never_asks(models, contradicting_repo: Path, capsys):
    queue, created = models
    queue.append([AIMessage(REPORT)])

    assert cli.main(["review", str(contradicting_repo)]) == 0

    out = capsys.readouterr().out
    assert "[Unverified]" in out
    assert "ADR not consulted" in out
    assert "Review scope: changes against HEAD, untracked files included; no truncation; ADRs not searched; 1 finding(s) with invalid evidence.\n" in out
    (diff,) = [m for m in created[0].seen[0] if isinstance(m, ToolMessage)]
    assert diff.name == "get_git_diff"
    assert "+  localStorage.setItem(TOKEN_KEY, token);" in diff.content


def test_diff_failure_stops_the_review_without_calling_the_model(models, monkeypatch, contradicting_repo, capsys):
    monkeypatch.setattr(agent.tools, "git_diff", lambda root: {"status": "error", "message": "fatal: bad object HEAD"})
    queue, created = models
    queue.append([AIMessage("No concerns")])

    assert cli.main(["review", str(contradicting_repo)]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert err == "futureyou: could not read the Git diff: fatal: bad object HEAD\n"
    assert created[0].seen == []


def test_review_without_adrs_reports_the_limitation(models, contradicting_repo: Path, capsys):
    for adr in (contradicting_repo / "docs" / "adr").iterdir():
        adr.unlink()
    queue, _ = models
    queue.append(review_replies())

    assert cli.main(["review", str(contradicting_repo)]) == 0

    assert "Review scope: changes against HEAD, untracked files included; no truncation; no ADRs in docs/adr/; 1 finding(s) with invalid evidence.\n" in capsys.readouterr().out


def test_review_scope_reports_truncation(models, monkeypatch, contradicting_repo: Path, capsys):
    monkeypatch.setattr(agent.tools, "MAX_DIFF_CHARS", 200)
    queue, _ = models
    queue.append(review_replies())

    assert cli.main(["review", str(contradicting_repo)]) == 0

    assert "Review scope: changes against HEAD, untracked files included; diff truncated at 200 characters; ADRs searched.\n" in (
        capsys.readouterr().out
    )


def test_review_scope_reports_untracked_cap(models, monkeypatch, sample_repo: Path, capsys):
    monkeypatch.setattr(agent.tools, "MAX_UNTRACKED_FILES", 1, raising=False)
    (sample_repo / "a.txt").write_text("first\n")
    (sample_repo / "b.txt").write_text("second\n")
    queue, _ = models
    queue.append([AIMessage(REPORT)])

    assert cli.main(["review", str(sample_repo), "--thread", "cap"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "Review scope: changes against HEAD, untracked files included; no truncation; 1 untracked files over the 1-file limit not read; ADRs not searched; 1 finding(s) with invalid evidence.\n" in out


def test_review_stops_at_the_recursion_limit(models, contradicting_repo: Path, capsys):
    queue, _ = models
    queue.append([
        AIMessage("", tool_calls=[{"name": "get_git_diff", "args": {}, "id": f"call-{i}"}]) for i in range(20)
    ])

    assert cli.main(["review", str(contradicting_repo)]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert err == f"futureyou: the review stopped after {agent.RECURSION_LIMIT} steps without a report\n"


def test_provider_errors_are_reported_without_traceback(monkeypatch, tmp_path: Path, contradicting_repo: Path, capsys):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    class FailingModel(ScriptedModel):
        def invoke(self, messages, *args, **kwargs):
            raise PermissionError("invalid API key")

    monkeypatch.setattr(agent, "load_model", lambda: FailingModel(messages=iter([])))

    assert cli.main(["review", str(contradicting_repo)]) == 1

    out, err = capsys.readouterr()
    assert out == ""
    assert err == "futureyou: model call failed: PermissionError: invalid API key\n"


def test_opencode_base_url_sends_a_stable_session_header(monkeypatch):
    monkeypatch.setenv("FUTUREYOU_MODEL", "openai:kimi-k3")
    monkeypatch.setenv("OPENAI_API_KEY", "oc_sk_test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://opencode.ai/inference/go/openai/v1")
    first, second = agent.load_model(), agent.load_model()
    session = first.default_headers["x-opencode-session"]
    assert session.startswith("ses_")
    assert second.default_headers["x-opencode-session"] == session


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("openai:gpt-4.1-mini", None),
        ("openai:kimi-k3", "https://opencode.ai.example.com/v1"),
        ("openai:kimi-k3", "https://example.com/opencode.ai/v1"),
        ("anthropic:claude-haiku-5-5", "https://opencode.ai/inference/go/openai/v1"),
    ],
)
def test_other_setups_send_no_session_header(monkeypatch, model, base_url):
    calls = []
    monkeypatch.setattr(agent, "init_chat_model", lambda name, **kwargs: calls.append((name, kwargs)))
    monkeypatch.setenv("FUTUREYOU_MODEL", model)
    if base_url:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent.load_model()
    assert calls == [(model, {})]
