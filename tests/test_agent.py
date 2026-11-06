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
