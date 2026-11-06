from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from futureyou import agent, cli

REPORT = (
    "1 concern\n\n"
    "[Contradiction] Session token stored on client\n"
    "Why: The change persists the session token in localStorage.\n"
    "Change: src/auth.ts:8-25\n"
    "Decision: docs/adr/003-authentication.md:13-17\n"
    "Suggestion: Keep server-managed sessions or revise ADR-003."
)


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that supports bind_tools and records what it was sent."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

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
        model = ScriptedModel(messages=iter(queue.pop(0)), seen=[])
        created.append(model)
        return model

    monkeypatch.setattr(agent, "load_model", load_model)
    return queue, created


def review_replies() -> list[AIMessage]:
    return [
        AIMessage("", tool_calls=[{"name": "get_git_diff", "args": {}, "id": "call-1"}]),
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
    assert "Review scope: tracked changes against HEAD; no truncation; ADRs searched.\n" in out
    final_input = created[0].seen[-1]
    diff, adrs = [m for m in final_input if isinstance(m, ToolMessage)]
    assert "+  localStorage.setItem(TOKEN_KEY, token);" in diff.content
    assert "docs/adr/003-authentication.md" in adrs.content
    assert _snapshot(contradicting_repo) == before


def test_review_without_changes_does_not_search_or_invent(models, sample_repo: Path, capsys):
    queue, _ = models
    queue.append([
        AIMessage("", tool_calls=[{"name": "get_git_diff", "args": {}, "id": "call-1"}]),
        AIMessage("No concerns"),
    ])

    assert cli.main(["review", str(sample_repo)]) == 0

    out = capsys.readouterr().out
    assert "No concerns\n" in out
    assert "Review scope: no tracked changes against HEAD; ADRs not searched.\n" in out


def test_review_without_adrs_reports_the_limitation(models, contradicting_repo: Path, capsys):
    for adr in (contradicting_repo / "docs" / "adr").iterdir():
        adr.unlink()
    queue, _ = models
    queue.append(review_replies())

    assert cli.main(["review", str(contradicting_repo)]) == 0

    assert "Review scope: tracked changes against HEAD; no truncation; no ADRs in docs/adr/.\n" in capsys.readouterr().out


def test_review_scope_reports_truncation(models, monkeypatch, contradicting_repo: Path, capsys):
    monkeypatch.setattr(agent.tools, "MAX_DIFF_CHARS", 200)
    queue, _ = models
    queue.append(review_replies())

    assert cli.main(["review", str(contradicting_repo)]) == 0

    assert "Review scope: tracked changes against HEAD; diff truncated at 200 characters; ADRs searched.\n" in (
        capsys.readouterr().out
    )


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
