import { request } from "./api";

export interface Credentials {
  email: string;
  password: string;
}

const TOKEN_KEY = "session_token";

export async function login(credentials: Credentials): Promise<void> {
  const response = await request("/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials),
  });
  const { token } = await response.json();
  // Keep the user logged in across reloads.
  localStorage.setItem(TOKEN_KEY, token);
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export async function logout(): Promise<void> {
  localStorage.removeItem(TOKEN_KEY);
  await request("/session", { method: "DELETE" });
}
