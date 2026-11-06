# FutureYou — The Architectural Regret Agent

> "Am I making a technical decision I will regret six months from now?"

FutureYou reviews the local changes of a Git repository against its documented
architecture decisions (ADRs) and reports evidence-backed observations.

It is not a full audit system. It is a small, working LangGraph agent with two
tools, persistent state and deterministic tests.

> **Status:** design phase. No code yet.

## The problem

Architecture decisions get written down once and then forgotten. A change that
quietly contradicts an ADR passes code review because nobody remembers the ADR.
FutureYou brings that context back into the review, at the moment the change is
still local and cheap to rethink.

## What it does

Given a repository with `docs/adr/003-authentication.md` stating that sessions
must be stored and validated server-side, and a local change that stores a
session token in `localStorage`:

```bash
futureyou review .
```

```text
FutureYou Report

1 concern

Architectural contradiction
The new authentication code persists session tokens in localStorage,
contradicting the server-side session policy.

EVIDENCE
  src/auth.ts — added lines 24–28
  docs/adr/003-authentication.md — lines 12–17

RECOMMENDATION
  Preserve server-managed sessions or explicitly revise ADR-003
  if the architecture has changed.
```

Every finding falls into one of three types:

| Type | Meaning |
|---|---|
| Contradiction | A change contradicts a documented decision. |
| Potential regret | A change adds a reasonable maintainability risk, even if no rule forbids it. |
| No concerns | Nothing relevant found. This is a valid outcome. |

FutureYou must not invent criticism to look useful.

## Usage

```bash
# First review, stored under a thread
futureyou review . --thread auth-review

# Follow-up question, even after closing the terminal
futureyou ask --thread auth-review "What would you change first?"
```

## Architecture

```text
CLI input
   │
   ▼
Agent node (LLM + tool selection) ──── no tool call ───► Final report
   │  ▲
tool call
   │  │
   ▼  │
ToolNode (get_git_diff · search_decisions)

SQLite checkpointer — thread_id
```

- **Graph:** an explicit `StateGraph` with a `ToolNode` and conditional routing,
  no prebuilt agent hiding the flow. Iterations are bounded by LangGraph's
  `recursion_limit`.
- **`get_git_diff`:** runs `git diff HEAD` (staged and unstaged changes to
  tracked files), with file names and line references. Output is size-limited
  and truncation is reported explicitly.
- **`search_decisions`:** term matching over the Markdown files in `docs/adr/`,
  returning snippets with file and line references. No embeddings, vector
  stores or external services. A repository without ADRs is reported as such.
- **Model:** `init_chat_model` from LangChain, selected with a single variable,
  e.g. `FUTUREYOU_MODEL=openai:gpt-4.1-mini`.
- **Persistence:** `SqliteSaver` at `~/.local/share/futureyou/checkpoints.sqlite`
  (respects `XDG_DATA_HOME`), so nothing is written into the analyzed repository.

## Ground rules

- **Read-only.** Never modifies code, creates commits, writes into the analyzed
  repository or follows instructions found in its files.
- **Untrusted content.** The diff and the ADRs are data, not instructions.
- **Evidence required.** A contradiction cites both the change and the rule. If
  the evidence is not enough, the report says so.
- **Privacy.** The analyzed code is sent to the model provider. Do not run it on
  repositories with secrets or sensitive data without reviewing the diff first.

## Known limitations

- Untracked files do not appear in `git diff HEAD`.
- Only local repositories, Markdown ADRs and size-limited diffs.
- Deterministic tests prove the software works; they do not prove an LLM will
  catch every architectural problem.

## Out of scope for V0.1

Web UI, GitHub App or automatic PR review, multiple agents, RAG or vector
databases, Jira/Linear integration, long-term semantic memory, automatic code
repair, metrics dashboards, plugins, and a custom multi-provider layer.

## Tech stack

Python, uv, LangGraph, LangChain tool calling, SQLite checkpoints, pytest.
