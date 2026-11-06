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
uv run futureyou review .
```

```text
FutureYou Report

1 concern

[Contradiction] Session token stored on client
Why: The new code persists session tokens in localStorage, contradicting
     the server-side session policy.
Change: src/auth.ts:24-28
Decision: docs/adr/003-authentication.md:12-17
Suggestion: Keep server-managed sessions or explicitly revise ADR-003
            if the architecture has changed.

Review scope: tracked changes against HEAD; no truncation.
```

The report lists at most five observations. If there are no ADRs, no
matches or the diff was truncated, the report says so instead of presenting
the review as complete.

Every finding falls into one of three types:

| Type | Meaning |
|---|---|
| Contradiction | A change contradicts a documented decision. |
| Potential regret | A change adds a reasonable maintainability risk, even if no rule forbids it. |
| No concerns | Nothing relevant found. This is a valid outcome. |

FutureYou must not invent criticism to look useful.

## Usage

```bash
# Review with a generated thread ID (printed on screen)
uv run futureyou review .

# Review under a named thread
uv run futureyou review . --thread auth-review

# Follow-up question, even after closing the terminal
uv run futureyou ask --thread auth-review "What should I change first?"
```

- Every `review` starts a new session. Reusing an existing thread name fails
  instead of mixing two reviews.
- `ask` requires an existing thread. It answers from the saved conversation,
  not from the current code: it calls no tools and does not read the
  repository again. Run a new `review` to see newer changes.

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
- **Local state is sensitive too.** Checkpoints store diffs and ADR excerpts
  in the local SQLite database.
- **Privacy.** The analyzed code is sent to the model provider. Do not run it on
  repositories with secrets or sensitive data without reviewing the diff first.

## Known limitations

- Untracked files do not appear in `git diff HEAD`.
- Only local repositories with at least one commit, Markdown ADRs under
  `docs/adr/` and diffs up to 60,000 characters.
- Evidence pointers are produced by the model. Tests verify the flow, but
  nothing automatically verifies that each citation is semantically correct.
- Reading the diff before concluding is required by the prompt, not enforced
  by the graph: a real model could skip it.
- Deterministic tests prove the software works; they do not prove an LLM will
  catch every architectural problem.

## Out of scope for V0.1

Web UI, GitHub App or automatic PR review, multiple agents, RAG or vector
databases, Jira/Linear integration, long-term semantic memory, automatic code
repair, metrics dashboards, plugins, and a custom multi-provider layer.

## Tech stack

Python, uv, LangGraph, LangChain tool calling, SQLite checkpoints, pytest.

## License

[MIT](LICENSE)
