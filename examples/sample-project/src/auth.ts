import { request } from "./api";

export interface Credentials {
  email: string;
  password: string;
}

// The server answers with an HttpOnly session cookie (ADR-003).
export async function login(credentials: Credentials): Promise<void> {
  await request("/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials),
  });
}

export async function logout(): Promise<void> {
  await request("/session", { method: "DELETE" });
}
