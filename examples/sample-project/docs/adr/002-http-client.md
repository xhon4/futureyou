# ADR-002: Single HTTP client module

## Status

Accepted

## Context

API calls were spread across components, each handling errors differently.

## Decision

All requests to the backend go through `src/api.ts`. Components never call
`fetch` directly.

## Consequences

Error handling and base URLs live in one place.
