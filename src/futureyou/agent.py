"""The LangGraph review agent: state, tools bound to one repository, and routing."""

import json
import os
import uuid
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from futureyou import prompts, report, tools

RECURSION_LIMIT = 12
# OpenCode Go requires a stable x-opencode-session header; one per process is enough.
OPENCODE_SESSION = f"ses_{uuid.uuid4().hex}"


class State(MessagesState):
    mode: Literal["review", "ask"]
    repo_root: str
    invalid_evidence: int


class ConfigError(Exception):
    """FutureYou is not configured to call a model."""


def load_model() -> BaseChatModel:
    name = os.environ.get("FUTUREYOU_MODEL")
    if not name:
        raise ConfigError(
            "FUTUREYOU_MODEL is not set, e.g. export FUTUREYOU_MODEL=openai:gpt-4.1-mini"
        )
    host = urlparse(os.environ.get("OPENAI_BASE_URL", "")).hostname or ""
    if name.startswith("openai:") and (
        host == "opencode.ai" or host.endswith(".opencode.ai")
    ):
        return init_chat_model(
            name, default_headers={"x-opencode-session": OPENCODE_SESSION}
        )
    return init_chat_model(name)


def build_tools(repo_root: Path) -> list:
    """The two tools, bound to an already validated repository. The model never picks paths."""

    @tool
    def get_git_diff() -> dict:
        """Return changes against HEAD, including non-ignored untracked files."""
        return tools.git_diff(repo_root)

    @tool
    def search_decisions(query: str) -> dict:
        """Search the architecture decision records in docs/adr/ for the terms in `query`."""
        return tools.search_decisions(repo_root, query)

    return [get_git_diff, search_decisions]


def route(state: State) -> Literal["tools", "report", "__end__"]:
    # Ask never calls tools, even if the model emits tool_calls despite tool_choice.
    if state["mode"] == "ask":
        return END
    if state["messages"][-1].tool_calls:
        return "tools"
    return "report"


def route_start(state: State) -> Literal["read_diff", "agent"]:
    return "read_diff" if state["mode"] == "review" else "agent"


def route_diff(state: State) -> Literal["agent", "__end__"]:
    """Only a readable, non-empty diff reaches the model; the CLI reports the other cases."""
    return (
        "agent" if json.loads(state["messages"][-1].content)["status"] == "ok" else END
    )


class DeferredModel:
    """Load the provider model on first use.

    A clean repository and a Git failure never reach the agent node, so they
    do not need FUTUREYOU_MODEL. ``bind_tools`` stays inside that node.
    """

    def __init__(self) -> None:
        self._model: BaseChatModel | None = None

    def _load(self) -> BaseChatModel:
        if self._model is None:
            self._model = load_model()
        return self._model

    def bind_tools(self, tools, **kwargs):
        return self._load().bind_tools(tools, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        return self._load().with_structured_output(schema, **kwargs)

    def invoke(self, messages, **kwargs):
        return self._load().invoke(messages, **kwargs)


def build_graph(model, repo_root: Path, checkpointer=None):
    tool_list = build_tools(repo_root)
    get_git_diff, search_decisions = tool_list

    def agent(state: State) -> dict:
        if state["mode"] == "review":
            # The graph already read the diff, so the review model only searches ADRs.
            chat = model.bind_tools([search_decisions])
            instructions = prompts.REVIEW
        else:
            # tool_choice is a hint. route() is what actually refuses tool calls.
            chat = model.bind_tools(tool_list, tool_choice="none")
            instructions = prompts.ASK
        reply = chat.invoke([SystemMessage(instructions), *state["messages"]])
        if state["mode"] == "ask" and getattr(reply, "tool_calls", None):
            reply = AIMessage(reply.text)
        return {"messages": [reply]}

    def read_diff(state: State) -> dict:
        """Read the diff before the model runs, recorded as a regular get_git_diff call."""
        call = {"name": get_git_diff.name, "args": {}, "id": "read-diff"}
        result = tools.git_diff(repo_root)
        return {
            "messages": [
                AIMessage("", tool_calls=[call]),
                ToolMessage(
                    json.dumps(result), tool_call_id=call["id"], name=call["name"]
                ),
            ]
        }

    def make_report(state: State) -> dict:
        structured = model.with_structured_output(report.Report).invoke(
            [SystemMessage(prompts.REPORT), *state["messages"]]
        )
        findings = structured.findings[:5]
        invalid = sum(
            any(report.validate(finding, state["messages"])) for finding in findings
        )
        return {
            "messages": [AIMessage(report.render(findings, state["messages"]))],
            "invalid_evidence": invalid,
        }

    graph = StateGraph(State)
    graph.add_node("read_diff", read_diff)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tool_list))
    graph.add_node("report", make_report)
    graph.add_edge("report", END)
    graph.add_conditional_edges(START, route_start)
    graph.add_conditional_edges("read_diff", route_diff)
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
