/** Thin typed wrapper over the Flask JSON API. Mirrors the legacy `api()` helper: unwraps `{ok, data}`. */

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

interface Envelope<T> {
  ok?: boolean;
  data?: T;
  error?: string;
}

export async function api<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  const method = init.method ?? 'GET';
  const res = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' }, // the server rejects mutating requests that are not JSON
    body: init.body === undefined ? (method === 'GET' ? undefined : '{}') : JSON.stringify(init.body),
  });
  let payload: Envelope<T> | T | null = null;
  try {
    payload = (await res.json()) as Envelope<T>;
  } catch {
    /* non-JSON body: fall through to the status check */
  }
  const env = (payload ?? {}) as Envelope<T>;
  if (!res.ok || env.ok === false) {
    throw new ApiError(env.error || res.statusText || 'Request failed', res.status);
  }
  return (env.data !== undefined ? env.data : (payload as T)) as T;
}

/** For list endpoints where a failure should degrade to an empty list rather than break the page. */
export async function apiOr<T>(path: string, fallback: T): Promise<T> {
  try {
    return await api<T>(path);
  } catch {
    return fallback;
  }
}

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
