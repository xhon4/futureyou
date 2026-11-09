"""The LangGraph review agent: state, tools bound to one repository, and routing."""

import os
import uuid
from pathlib import Path
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from futureyou import prompts, tools

RECURSION_LIMIT = 12
# OpenCode Go requires a stable x-opencode-session header; one per process is enough.
OPENCODE_SESSION = f"ses_{uuid.uuid4().hex}"


class State(MessagesState):
    mode: Literal["review", "ask"]
    repo_root: str


class ConfigError(Exception):
    """FutureYou is not configured to call a model."""


def load_model() -> BaseChatModel:
    name = os.environ.get("FUTUREYOU_MODEL")
    if not name:
        raise ConfigError("FUTUREYOU_MODEL is not set, e.g. export FUTUREYOU_MODEL=openai:gpt-4.1-mini")
    if "opencode.ai" in os.environ.get("OPENAI_BASE_URL", ""):
        return init_chat_model(name, default_headers={"x-opencode-session": OPENCODE_SESSION})
    return init_chat_model(name)


def build_tools(repo_root: Path) -> list:
    """The two tools, bound to an already validated repository. The model never picks paths."""

    @tool
    def get_git_diff() -> dict:
        """Return the staged and unstaged changes to tracked files against HEAD."""
        return tools.git_diff(repo_root)

    @tool
    def search_decisions(query: str) -> dict:
        """Search the architecture decision records in docs/adr/ for the terms in `query`."""
        return tools.search_decisions(repo_root, query)

    return [get_git_diff, search_decisions]


def route(state: State) -> Literal["tools", "__end__"]:
    return "tools" if state["messages"][-1].tool_calls else END


def build_graph(model: BaseChatModel, repo_root: Path, checkpointer=None):
    tool_list = build_tools(repo_root)
    review_model = model.bind_tools(tool_list)
    # Same tools, but never callable: the saved history keeps valid tool messages.
    ask_model = model.bind_tools(tool_list, tool_choice="none")

    def agent(state: State) -> dict:
        if state["mode"] == "review":
            chat, instructions = review_model, prompts.REVIEW
        else:
            chat, instructions = ask_model, prompts.ASK
        return {"messages": [chat.invoke([SystemMessage(instructions), *state["messages"]])]}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tool_list))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
