import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { App } from '../App';
import { ROUTES } from '../routes';
import { calls, installApi, ok } from './helpers';

const QUESTION = { index: 3, code: 'A4', text: 'Does your child look at you when you call their name?',
  options: [{ value: 0, letter: 'a', label: 'Always' }, { value: 1, letter: 'b', label: 'Rarely' }] };
const META = { budget: 6, tau: 0.5, policies_available: ['greedy', 'random', 'beta_greedy'], rl_policy: { available: false, reason: 'no checkpoint' } };

beforeEach(() => { window.location.hash = '#/screening'; });

describe('routing', () => {
  it.each(ROUTES.map((r) => [r.key, r.title] as const))('renders the %s view with its title and no crash', async (key, title) => {
    installApi({ 'GET /api/meta': () => ok(META), 'GET /api/rl/scenarios': () => ok({ scenarios: [], policies: [], budgets: [6] }) });
    window.location.hash = `#/${key}`;
    render(<App />);
    expect(await screen.findByRole('heading', { level: 1, name: title })).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole('status', { name: '' , busy: true })).toBeNull());
    expect(screen.queryByText(/Failed to load module/)).toBeNull();
    expect(screen.getByRole('link', { name: ROUTES.find((r) => r.key === key)!.nav, current: 'page' })).toBeInTheDocument();
  });

  it('an unknown hash falls back to the overview', async () => {
    installApi({});
    window.location.hash = '#/nope';
    render(<App />);
    expect(await screen.findByRole('heading', { level: 1, name: 'Overview Dashboard' })).toBeInTheDocument();
  });
});

describe('guided screening', () => {
  it('runs start -> answer -> result, posting the item index and the scored value', async () => {
    installApi({
      'GET /api/meta': () => ok(META),
      'POST /api/screening/start': () => ok({ session_id: 'ses_1', mode: 'interactive', budget: 6, tau: 0.5, policy: 'greedy', question: QUESTION, belief: 0.4, asked: 0, selection: null }),
      'POST /api/screening/answer': () => ok({ finished: true, result: { decision: 'NO_REFERRAL', p_hat: 0.21, asked: 1, stop_reason: 'budget', trace: [], support_report: { recommendations: [] } } }),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: 'Start new session' }));

    expect(await screen.findByRole('heading', { level: 2, name: /^A4\s*Does your child/ })).toBeInTheDocument();
    // answers are the instrument's own wording, never a literal Yes/No
    const group = screen.getByRole('group', { name: 'Answer options' });
    expect(within(group).getByRole('button', { name: 'Always' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^(yes|no)$/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /prefer not to answer/i })).toBeNull(); // rl_adaptive only

    await userEvent.click(within(group).getByRole('button', { name: 'Rarely' }));
    expect(await screen.findByText('Screening result (research prototype)')).toBeInTheDocument();
    expect(screen.getByText('NO REFERRAL')).toBeInTheDocument();
    expect(screen.getByText(/Not a diagnosis\./)).toBeInTheDocument();

    const post = calls.find((c) => c.path === '/api/screening/answer');
    expect(post?.body).toMatchObject({ session_id: 'ses_1', value: 1, stop: false, item_index: 3 });
  });

  it('shows a warning instead of Yes/No when the backend sends no answer options', async () => {
    installApi({
      'GET /api/meta': () => ok(META),
      'POST /api/screening/start': () => ok({ session_id: 's', mode: 'interactive', budget: 6, tau: 0.5, policy: 'greedy', question: { index: 0, code: 'A1', text: 'q' }, belief: 0.4, asked: 0 }),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: 'Start new session' }));
    expect(await screen.findByText(/Answer options for this question are unavailable/)).toBeInTheDocument();
  });

  it('rl_adaptive offers "Prefer not to answer" and sends skip', async () => {
    installApi({
      'GET /api/meta': () => ok({ ...META, policies_available: ['greedy', 'rl_adaptive'] }),
      'POST /api/screening/start': () => ok({ session_id: 'rl1', mode: 'interactive', budget: 10, tau: 0.5, policy: 'rl_adaptive', question: QUESTION, belief: 0.4, asked: 0 }),
      'POST /api/screening/answer': () => ok({ finished: false, next: { ...QUESTION, index: 5, code: 'A6' }, belief_after: 0.4, asked: 1, trace_step: { step: 1, item: 'A4', skipped: true, belief_before: 0.4, belief_after: 0.4 } }),
    });
    render(<App />);
    await userEvent.selectOptions(await screen.findByLabelText(/Policy \(real/), 'rl_adaptive');
    await userEvent.click(screen.getByRole('button', { name: 'Start new session' }));
    await userEvent.click(await screen.findByRole('button', { name: /prefer not to answer/i }));
    expect(await screen.findByRole('heading', { level: 2, name: /^A6\s*\S/ })).toBeInTheDocument();
    expect(calls.find((c) => c.path === '/api/screening/answer')?.body).toMatchObject({ skip: true, value: null, item_index: 3 });
    expect(screen.getByText('skipped')).toBeInTheDocument();
  });

  it('a double click submits the answer once', async () => {
    let resolve: (v: unknown) => void = () => {};
    installApi({
      'GET /api/meta': () => ok(META),
      'POST /api/screening/start': () => ok({ session_id: 's', mode: 'interactive', budget: 6, tau: 0.5, policy: 'greedy', question: QUESTION, belief: 0.4, asked: 0 }),
    });
    const inner = globalThis.fetch;
    vi.stubGlobal('fetch', vi.fn((i: RequestInfo | URL, init?: RequestInit) =>
      String(i) === '/api/screening/answer'
        ? new Promise<Response>((r) => { calls.push({ method: 'POST', path: String(i), body: init?.body }); resolve = r as (v: unknown) => void; })
        : inner(i, init)));
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: 'Start new session' }));
    const btn = await screen.findByRole('button', { name: 'Always' });
    await userEvent.dblClick(btn);
    expect(calls.filter((c) => c.path === '/api/screening/answer')).toHaveLength(1);
    resolve({ ok: true, status: 200, json: async () => ({ ok: true, data: { finished: true, result: { decision: 'NO_REFERRAL' } } }) });
    await screen.findByText('Screening result (research prototype)');
  });

  it('on a 409 (stale answer) it reloads the current question from the server', async () => {
    installApi({
      'GET /api/meta': () => ok(META),
      'POST /api/screening/start': () => ok({ session_id: 's9', mode: 'interactive', budget: 6, tau: 0.5, policy: 'greedy', question: QUESTION, belief: 0.4, asked: 0 }),
      'POST /api/screening/answer': () => ({ status: 409, body: { ok: false, error: 'already answered' } }),
      'GET /api/screening/sessions/s9': () => ok({ id: 's9', status: 'active', policy: 'greedy', mode: 'interactive', budget: 6, tau: 0.5, asked_count: 1, belief: 0.5, pending_question: { ...QUESTION, index: 4, code: 'A5' }, trace: [] }),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: 'Start new session' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Always' }));
    expect(await screen.findByRole('heading', { level: 2, name: /^A5\s*\S/ })).toBeInTheDocument();
    expect(await screen.findByRole('alert')).toHaveTextContent('already answered');
  });
});

describe('destructive privacy actions', () => {
  it('asks before deleting: cancel sends nothing, confirm posts', async () => {
    window.location.hash = '#/privacy';
    installApi({ 'POST /api/privacy/delete_all': () => ok({ deleted: true }) });
    render(<App />);
    const del = await screen.findByRole('button', { name: 'Delete all app data' });

    await userEvent.click(del);
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel' }));
    expect(calls.some((c) => c.path === '/api/privacy/delete_all')).toBe(false);

    await userEvent.click(del);
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/api/privacy/delete_all')).toBe(true));
  });

  it('sends JSON content type on mutating requests (the server rejects anything else)', async () => {
    window.location.hash = '#/privacy';
    installApi({ 'POST /api/consent': () => ok({}) });
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: 'Save consent record' }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/consent')).toBe(true));
    const init = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find((c) => c[0] === '/api/consent')![1] as RequestInit;
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');
  });
});

describe('error handling', () => {
  it('a view that throws while rendering shows the error card, and navigating away recovers', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    // a malformed payload (object instead of a list) makes the overview throw inside render
    installApi({ 'GET /api/screening/sessions': () => ok({ unexpected: 'shape' }) });
    window.location.hash = '#/overview';
    render(<App />);
    expect(await screen.findByText(/Failed to load module/)).toBeInTheDocument();
    window.location.hash = '#/timeline';
    expect(await screen.findByRole('heading', { level: 1, name: 'Observation Timeline' })).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText(/Failed to load module/)).toBeNull());
  });

  it('degrades list endpoints to empty states when the network is down', async () => {
    window.location.hash = '#/context';
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('network down'); }));
    render(<App />);
    expect(await screen.findByText(/No observations match/)).toBeInTheDocument();
  });
});
