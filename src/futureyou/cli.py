"""Command line entry point: `futureyou review PATH`."""

import argparse
import json
import os
import sys
from pathlib import Path

# Restrict checkpoint deserialization to known types (GHSA-g48c-2wqr-h844).
# Must be set before LangGraph is imported.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

from futureyou import agent, tools  # noqa: E402

REVIEW_REQUEST = "Review the local changes of this repository."


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


def review(path: Path) -> int:
    repo_root = tools.resolve_repo(path)
    graph = agent.build_graph(agent.load_model(), repo_root)
    state = graph.invoke(
        {"messages": [HumanMessage(REVIEW_REQUEST)], "mode": "review", "repo_root": str(repo_root)},
        {"recursion_limit": agent.RECURSION_LIMIT},
    )
    print("FutureYou Report\n")
    print(state["messages"][-1].text.strip())
    print(f"\n{review_scope(state['messages'])}")
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
    args = parser.parse_args(argv)

    try:
        return review(args.path)
    except (tools.RepoError, agent.ConfigError) as error:
        print(f"futureyou: {error}", file=sys.stderr)
    except GraphRecursionError:
        print(f"futureyou: the review stopped after {agent.RECURSION_LIMIT} steps without a report", file=sys.stderr)
    except Exception as error:  # Provider, authentication and network errors.
        print(f"futureyou: model call failed: {type(error).__name__}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
