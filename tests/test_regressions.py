"""Regressions for the Git read path and the review graph.

Each test was written against the unmodified T4 behavior and fails there.
"""

import json
import subprocess
from pathlib import Path

from conftest import git
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from futureyou import agent, cli, report, tools


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("base\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def _regret(path: str, start: int, end: int) -> report.Finding:
    return report.Finding(
        type="Potential regret",
        title="t",
        why="w",
        suggestion="s",
        change_path=path,
        change_start=start,
        change_end=end,
    )


def _messages(diff: str) -> list[ToolMessage]:
    payload = {"diff": diff, "status": "ok"}
    return [ToolMessage(json.dumps(payload), name="get_git_diff", tool_call_id="1")]


def _scope(result: dict) -> str:
    return cli.review_scope(
        [ToolMessage(json.dumps(result), name="get_git_diff", tool_call_id="1")]
    )


def test_repo_quotepath_config_does_not_mangle_the_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.quotePath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    repo = _repo(tmp_path)
    target = repo / "café.py"
    target.write_text("one\n")
    git(repo, "add", "--", "café.py")
    git(repo, "commit", "-q", "-m", "add")
    target.write_text("one\ntwo changed\n")
    git(repo, "config", "core.quotepath", "true")
    diff = tools.git_diff(repo)["diff"]
    assert r"\303" not in diff
    assert report.diff_ranges(diff).get("café.py"), report.diff_ranges(diff)


def test_mnemonic_prefix_config_does_not_change_the_diff(tmp_path):
    repo = _repo(tmp_path)
    target = repo / "auth.py"
    target.write_text("one\ntwo\n")
    git(repo, "add", "auth.py")
    git(repo, "commit", "-q", "-m", "add")
    target.write_text("one\ntwo changed\n")
    git(repo, "config", "diff.mnemonicPrefix", "true")
    diff = tools.git_diff(repo)["diff"]
    assert "w/auth.py" not in diff
    assert report.diff_ranges(diff).get("auth.py"), report.diff_ranges(diff)


def test_non_utf8_text_does_not_crash(tmp_path):
    repo = _repo(tmp_path)
    (repo / "latin.txt").write_bytes("café\n".encode("latin-1"))
    git(repo, "add", "latin.txt")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "latin.txt").write_bytes("café changed\n".encode("latin-1"))
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert "changed" in result["diff"]


def test_index_only_staged_edit_is_part_of_the_review(tmp_path):
    repo = _repo(tmp_path)
    target = repo / "auth.py"
    target.write_text("safe\n")
    git(repo, "add", "auth.py")
    git(repo, "commit", "-q", "-m", "base")
    target.write_text("bug\n")
    git(repo, "add", "auth.py")
    target.write_text("safe\n")
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert "+bug" in result["diff"]
    assert "+safe" in result["diff"]


def test_clean_filter_does_not_rewrite_the_diff_or_write_git(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    script = tmp_path / "clean.sh"
    script.write_text(
        "#!/bin/sh\nmkdir -p .git/filter-ran\necho ran >> .git/filter-ran/marker\nprintf 'POINTER\\n'\n"
    )
    script.chmod(0o755)
    git(repo, "config", "filter.evil.clean", str(script))
    git(repo, "config", "filter.evil.required", "true")
    (repo / ".gitattributes").write_text("* filter=evil\n")
    (repo / "tracked.txt").write_text("v2\n")
    (repo / "new.txt").write_text("new-body\n")
    result = tools.git_diff(repo)
    assert result["status"] == "ok"
    assert "v2" in result["diff"]
    assert "new-body" in result["diff"]
    assert "POINTER" not in result["diff"]
    assert not (repo / ".git" / "filter-ran").exists()


def test_fsmonitor_hook_does_not_run(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    marker = tmp_path / "fsm-marker"
    hook = tmp_path / "fsm.sh"
    hook.write_text(f"#!/bin/sh\necho ran >> {marker}\nprintf '\\0'\n")
    hook.chmod(0o755)
    git(repo, "config", "core.fsmonitor", str(hook))
    (repo / "tracked.txt").write_text("v2\n")
    result = tools.git_diff(repo)
    assert "v2" in result["diff"]
    assert not marker.exists()


def test_scope_does_not_claim_omitted_untracked_files(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "big.py").write_text("seed\n")
    git(repo, "add", "big.py")
    git(repo, "commit", "-q", "-m", "add")
    (repo / "big.py").write_text("\n".join(f"line{i}" for i in range(80)) + "\n")
    (repo / "new.txt").write_text("UNTRACKED_MARKER\n")
    monkeypatch.setattr(tools, "MAX_DIFF_CHARS", 500)
    result = tools.git_diff(repo)
    assert result["truncated"] is True
    assert "UNTRACKED_MARKER" not in result["diff"]
    scope = _scope(result)
    assert "untracked files included" not in scope
    assert "truncat" in scope


def test_untracked_nested_repo_is_disclosed_and_not_counted(tmp_path):
    repo = _repo(tmp_path)
    nested = repo / "vendor"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "lib.txt").write_text("library\n")
    git(nested, "add", "lib.txt")
    git(nested, "commit", "-q", "-m", "lib")
    result = tools.git_diff(repo)
    assert "library" not in result["diff"]
    assert result["untracked"] == 0
    assert result["nested_skipped"] == 1
    scope = _scope(result)
    assert "nested" in scope
    assert "untracked files included" not in scope


def test_dirty_submodule_is_disclosed_without_its_contents(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    git(sub, "init", "-q")
    (sub / "inner.py").write_text("version1\n")
    git(sub, "add", "inner.py")
    git(sub, "commit", "-q", "-m", "inner")
    parent = tmp_path / "parent"
    parent.mkdir()
    git(parent, "init", "-q")
    (parent / "README.md").write_text("parent\n")
    git(parent, "add", "README.md")
    git(parent, "commit", "-q", "-m", "base")
    git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--",
        str(sub),
        "sub",
    )
    git(parent, "commit", "-q", "-m", "add sub")
    (parent / "sub" / "inner.py").write_text("version2 CHANGED\n")
    (parent / "sub" / "extra.txt").write_text("untracked-inner\n")
    git(parent, "config", "diff.submodule", "diff")
    result = tools.git_diff(parent)
    assert "version2" not in result["diff"]
    assert "untracked-inner" not in result["diff"]
    assert "Subproject commit" in result["diff"]
    assert result["submodules"] >= 1
    assert "submodule" in _scope(result)


def test_oversized_untracked_file_is_not_fully_buffered(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "big.txt").write_text("x" * 200_000 + "\n")
    sizes: list[int] = []
    real = subprocess.run

    def spy(cmd, **kwargs):
        completed = real(cmd, **kwargs)
        sizes.append(len(completed.stdout or ""))
        return completed

    monkeypatch.setattr(tools.subprocess, "run", spy)
    result = tools.git_diff(repo)
    assert max(sizes) <= tools.MAX_DIFF_CHARS
    assert "x" * 1000 not in result["diff"]
    assert result["oversized_untracked"] == 1


def test_missing_git_is_not_called_a_model_failure(tmp_path, monkeypatch, capsys):
    empty = tmp_path / "bin"
    empty.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert cli.main(["review", str(repo)]) == 1
    err = capsys.readouterr().err
    assert "model call failed" not in err
    assert "git" in err.lower()


def test_decode_error_is_not_called_a_model_failure(tmp_path, monkeypatch, capsys):
    def boom(path):
        raise UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "invalid continuation byte")

    monkeypatch.setattr(cli.tools, "resolve_repo", boom)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert cli.main(["review", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "model call failed" not in err
    assert "decode" in err.lower()


def test_ask_mode_does_not_execute_tools_when_the_model_emits_tool_calls(
    tmp_path, monkeypatch
):
    calls = []

    def fake(root, query):
        calls.append(query)
        return {"status": "no_matches", "fragments": [], "skipped": []}

    monkeypatch.setattr(tools, "search_decisions", fake)
    repo = _repo(tmp_path)

    class Noncompliant(GenericFakeChatModel):
        def bind_tools(self, bound, **kwargs):
            return self

    model = Noncompliant(
        messages=iter(
            [
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "search_decisions",
                            "args": {"query": "tokens"},
                            "id": "c1",
                        }
                    ],
                ),
                AIMessage("answer"),
            ]
        )
    )
    graph = agent.build_graph(model, repo)
    graph.invoke(
        {
            "messages": [HumanMessage("What did we decide?")],
            "mode": "ask",
            "repo_root": str(repo),
        },
        {"recursion_limit": 8},
    )
    assert calls == []


def _hallucinating_model(tool_name: str) -> GenericFakeChatModel:
    class Hallucinating(GenericFakeChatModel):
        def bind_tools(self, bound, **kwargs):
            return self

        def with_structured_output(self, schema, **kwargs):
            from langchain_core.runnables import RunnableLambda

            return RunnableLambda(lambda messages: report.Report(findings=[]))

    return Hallucinating(
        messages=iter(
            [
                AIMessage(
                    "",
                    tool_calls=[{"name": tool_name, "args": {}, "id": "hallucinated"}],
                ),
                AIMessage("done"),
            ]
        )
    )


def test_review_does_not_execute_a_hallucinated_git_diff(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "README.md").write_text("base\nchanged\n")
    calls = []
    original = tools.git_diff

    def spy(root):
        calls.append(root)
        return original(root)

    monkeypatch.setattr(tools, "git_diff", spy)
    graph = agent.build_graph(_hallucinating_model("get_git_diff"), repo)
    graph.invoke(
        {
            "messages": [HumanMessage("Review the changes.")],
            "mode": "review",
            "repo_root": str(repo),
        },
        {"recursion_limit": 12},
    )
    assert len(calls) == 1


def test_ask_mode_does_not_execute_a_hallucinated_git_diff(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "git_diff", lambda root: calls.append(root))
    repo = _repo(tmp_path)
    graph = agent.build_graph(_hallucinating_model("get_git_diff"), repo)
    graph.invoke(
        {
            "messages": [HumanMessage("What changed?")],
            "mode": "ask",
            "repo_root": str(repo),
        },
        {"recursion_limit": 8},
    )
    assert calls == []


def test_clean_repo_does_not_require_a_model(
    sample_repo, monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("FUTUREYOU_MODEL", raising=False)
    assert cli.main(["review", str(sample_repo), "--thread", "clean"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "No concerns" in out
    assert "FUTUREYOU_MODEL" not in out


def test_git_commands_pin_safe_config(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("z\n")
    calls = []
    real = subprocess.run

    def spy(cmd, **kwargs):
        calls.append((cmd, kwargs.get("env") or {}))
        return real(cmd, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", spy)
    tools.git_diff(repo)
    assert calls
    for cmd, env in calls:
        assert "core.quotepath=false" in cmd
        assert "diff.mnemonicPrefix=false" in cmd
        assert "diff.noprefix=false" in cmd
        assert "core.fsmonitor=false" in cmd
        assert "core.hooksPath=/dev/null" in cmd
        assert env.get("GIT_OPTIONAL_LOCKS") == "0"


def test_git_subprocess_times_out_after_60_seconds(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("z\n")
    timeouts = []
    real = subprocess.run

    def spy(cmd, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        return real(cmd, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", spy)
    tools.git_diff(repo)
    assert timeouts
    assert timeouts == [60] * len(timeouts)


def test_git_timeout_is_not_called_a_model_failure(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)

    def timed_out(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 60)

    monkeypatch.setattr(tools.subprocess, "run", timed_out)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert cli.main(["review", str(repo), "--thread", "slow"]) == 1
    err = capsys.readouterr().err
    assert "model call failed" not in err
    assert "timed out" in err.lower()
