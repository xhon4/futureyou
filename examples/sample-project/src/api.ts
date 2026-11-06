const BASE_URL = "/api";

export async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(`${BASE_URL}${path}`, { credentials: "include", ...init });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response;
}
