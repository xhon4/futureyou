# ADR-003: Server-side sessions

## Status

Accepted

## Context

The application handles personal data. A token stolen from the browser would
give an attacker full access to an account.

## Decision

Sessions must be stored and validated on the server side. The server sets an
`HttpOnly`, `Secure` session cookie. The client must not store persistent
authentication tokens, including in `localStorage` or `sessionStorage`.

## Consequences

Logging out invalidates the session on the server. JavaScript running in the
page cannot read the session identifier.
