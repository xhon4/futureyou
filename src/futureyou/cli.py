"""Command line entry point: `futureyou review PATH` and `futureyou ask --thread ID QUESTION`."""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# Restrict checkpoint deserialization to known types (GHSA-g48c-2wqr-h844).
# Must be set before LangGraph is imported.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

from futureyou import agent, tools  # noqa: E402

REVIEW_REQUEST = "Review the local changes of this repository."


class ThreadError(Exception):
    """The requested thread cannot be used for this command."""


def checkpoint_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(data_home).expanduser() / "futureyou" / "checkpoints.sqlite"


def _open_checkpoints():
    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(path))


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": agent.RECURSION_LIMIT}


def _tool_results(messages: list[BaseMessage], name: str) -> list[dict]:
    results = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == name:
            try:
                results.append(json.loads(message.content))
            except (TypeError, json.JSONDecodeError):
                continue
    return results


def review_scope(messages: list[BaseMessage]) -> str:
    """Describe what the review actually covered, from the real tool results."""
    diffs = _tool_results(messages, "get_git_diff")
    if not diffs:
        diff_scope = "diff not read"
    elif diffs[-1].get("status") == "no_changes":
        diff_scope = "no tracked changes against HEAD"
    elif diffs[-1].get("status") != "ok":
        diff_scope = "Git diff failed"
    elif diffs[-1].get("truncated"):
        diff_scope = f"tracked changes against HEAD; diff truncated at {tools.MAX_DIFF_CHARS:,} characters"
    else:
        diff_scope = "tracked changes against HEAD; no truncation"

    statuses = {result.get("status") for result in _tool_results(messages, "search_decisions")}
    if not statuses:
        adr_scope = "ADRs not searched"
    elif "no_adrs" in statuses:
        adr_scope = "no ADRs in docs/adr/"
    elif "partial" in statuses:
        adr_scope = "some ADR files were skipped"
    elif statuses == {"no_matches"}:
        adr_scope = "no matching ADRs"
    else:
        adr_scope = "ADRs searched"
    return f"Review scope: {diff_scope}; {adr_scope}."


def review(path: Path, thread: str | None) -> int:
    repo_root = tools.resolve_repo(path)
    thread = thread or uuid.uuid4().hex[:8]
    with _open_checkpoints() as checkpoints:
        if checkpoints.get_tuple(_config(thread)) is not None:
            raise ThreadError(f"thread '{thread}' already exists; use another --thread")
        graph = agent.build_graph(agent.load_model(), repo_root, checkpoints)
        state = graph.invoke(
            {"messages": [HumanMessage(REVIEW_REQUEST)], "mode": "review", "repo_root": str(repo_root)},
            _config(thread),
        )
    diff = _tool_results(state["messages"], "get_git_diff")[0]
    if diff["status"] == "error":
        raise tools.RepoError(f"could not read the Git diff: {diff['message']}")
    print("FutureYou Report\n")
    print("No concerns" if diff["status"] == "no_changes" else state["messages"][-1].text.strip())
    print(f"\n{review_scope(state['messages'])}")
    print(f"Thread: {thread}")
    return 0


def ask(thread: str, question: str) -> int:
    with _open_checkpoints() as checkpoints:
        saved = checkpoints.get_tuple(_config(thread))
        if saved is None:
            raise ThreadError(f"thread '{thread}' does not exist; run futureyou review first")
        # Tools are never routed in ask mode; the graph only needs the reviewed root to compile.
        repo_root = Path(saved.checkpoint["channel_values"]["repo_root"])
        graph = agent.build_graph(agent.load_model(), repo_root, checkpoints)
        state = graph.invoke({"messages": [HumanMessage(question)], "mode": "ask"}, _config(thread))
    print(state["messages"][-1].text.strip())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="futureyou",
        description="Review local Git changes against the architecture decisions in docs/adr/.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    review_parser = commands.add_parser(
        "review", help="review staged and unstaged changes to tracked files against HEAD"
    )
    review_parser.add_argument("path", type=Path, help="path inside the Git repository to review")
    review_parser.add_argument("--thread", help="name for this review session (default: a generated ID)")
    ask_parser = commands.add_parser(
        "ask",
        help="ask a follow-up question about a saved review",
        description="Answer from the saved review conversation. It does not read the current code; "
        "run a new review to see newer changes.",
    )
    ask_parser.add_argument("--thread", required=True, help="thread of a previous review")
    ask_parser.add_argument("question", help="follow-up question about that review")
    args = parser.parse_args(argv)

    try:
        if args.command == "review":
            return review(args.path, args.thread)
        return ask(args.thread, args.question)
    except (tools.RepoError, agent.ConfigError, ThreadError) as error:
        print(f"futureyou: {error}", file=sys.stderr)
    except GraphRecursionError:
        print(f"futureyou: the review stopped after {agent.RECURSION_LIMIT} steps without a report", file=sys.stderr)
    except Exception as error:  # Provider, authentication and network errors.
        print(f"futureyou: model call failed: {type(error).__name__}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
