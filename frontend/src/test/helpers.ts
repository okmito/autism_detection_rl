import { vi } from 'vitest';

export type Handler = (body: unknown) => { status?: number; body: unknown };
export const calls: { method: string; path: string; body: unknown }[] = [];

/** Fake Flask API. Unlisted GETs return an empty list, like the endpoints do on a fresh database. */
export function installApi(routes: Record<string, Handler>) {
  calls.length = 0;
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    const method = init?.method ?? 'GET';
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, path, body });
    const h = routes[`${method} ${path}`];
    const out = h ? h(body) : { body: { ok: true, data: path === '/api/research/dashboard' ? { artifacts: {}, tables: {}, limitations: [] } : [] } };
    const status = out.status ?? 200;
    return { ok: status < 400, status, statusText: 'x', json: async () => out.body } as Response;
  }));
}
export const ok = (data: unknown): { body: unknown } => ({ body: { ok: true, data } });

