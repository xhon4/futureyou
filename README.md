# FutureYou — The Architectural Regret Agent

> "Am I making a technical decision I will regret six months from now?"

FutureYou reviews the local changes of a Git repository against its documented
architecture decisions (ADRs) and reports evidence-backed observations.

It is not a full audit system. It is a small, working LangGraph agent with two
tools, persistent state and deterministic tests.

> **Status:** V0.1. Working CLI with deterministic tests; real-model output
> below is illustrative until the first recorded demo run.

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
Change: src/auth.ts:17-18
Decision: docs/adr/003-authentication.md:14-16
Suggestion: Keep server-managed sessions or explicitly revise ADR-003
            if the architecture has changed.

Review scope: tracked changes against HEAD; no truncation; ADRs searched.
Thread: 3f9c2a1b
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

## Quick start

Requirements: Git, [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
git clone <repository-url> futureyou
cd futureyou
uv sync

# FutureYou does not load .env files: export the variables in your shell.
export FUTUREYOU_MODEL=openai:gpt-4.1-mini
export OPENAI_API_KEY=sk-...

scripts/demo.sh
```

`scripts/demo.sh` copies `examples/sample-project` to a temporary directory,
commits it, applies the change in `examples/sample-change` (a session token in
`localStorage`, against ADR-003), runs `review` and a follow-up `ask`, and
deletes the copy. Your own clone is never modified.

`FUTUREYOU_MODEL` takes any `provider:model` string supported by LangChain's
`init_chat_model`; only the OpenAI integration is installed.

Any OpenAI-compatible gateway works through `OPENAI_BASE_URL`. For example,
[OpenCode Go](https://opencode.ai/docs/console/go) (FutureYou sends the required
`x-opencode-session` header automatically when the base URL is on opencode.ai):

```bash
export OPENAI_BASE_URL=https://opencode.ai/inference/go/openai/v1
export OPENAI_API_KEY=oc_sk_...
export FUTUREYOU_MODEL=openai:kimi-k3
```

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

## Tests

```bash
uv run pytest
```

The suite uses simulated models: no network, no API key, no tokens spent.
It runs on every push through GitHub Actions.

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
- Results depend on the model: the same change can produce different wording
  or line ranges across runs.

## Out of scope for V0.1

Web UI, GitHub App or automatic PR review, multiple agents, RAG or vector
databases, Jira/Linear integration, long-term semantic memory, automatic code
repair, metrics dashboards, plugins, and a custom multi-provider layer.

## Tech stack

Python, uv, LangGraph, LangChain tool calling, SQLite checkpoints, pytest.

## License

[MIT](LICENSE)
