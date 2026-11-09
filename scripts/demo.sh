#!/usr/bin/env bash
# Runs FutureYou on a throwaway copy of examples/sample-project with a change
# that contradicts ADR-003. Needs FUTUREYOU_MODEL and the provider API key.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

repo="$workdir/sample-project"
cp -R "$root/examples/sample-project" "$repo"
git -C "$repo" init -q
git -C "$repo" add .
git -C "$repo" -c user.name="FutureYou Demo" -c user.email=demo@example.com commit -q -m "Base project"
cp -R "$root/examples/sample-change/." "$repo"

thread="demo-$(date +%s)"
echo "$ git diff --stat"
git -C "$repo" --no-pager diff --stat
echo
echo "$ futureyou review <sample-project> --thread $thread"
uv run --project "$root" futureyou review "$repo" --thread "$thread"
echo
question="What should I change first?"
echo "$ futureyou ask --thread $thread \"$question\""
uv run --project "$root" futureyou ask --thread "$thread" "$question"
