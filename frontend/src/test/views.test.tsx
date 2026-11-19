import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from '../App';
import { calls, installApi, ok } from './helpers';

const OBS = [
  { id: 'obs_1', context: 'Classroom environments', what_happened: 'Covered ears during the fire drill', frequency: 'once', consistency: 'Occasional', insufficient_info: false, created_at: '2026-11-01T10:00:00Z' },
  { id: 'obs_2', context: 'Daily routines', what_happened: 'Struggled with breakfast change', difficulty: 'transition', insufficient_info: true, created_at: '2026-11-02T10:00:00Z' },
];

describe('Context Explorer', () => {
  it('lists observations, filters by text and context, and saves a new one', async () => {
    window.location.hash = '#/context';
    installApi({ 'GET /api/observations': () => ok(OBS), 'POST /api/observations': () => ok({ id: 'obs_3' }) });
    render(<App />);
    expect(await screen.findByText('Covered ears during the fire drill')).toBeInTheDocument();
    expect(screen.getByText('Insufficient info')).toBeInTheDocument();

    await userEvent.type(screen.getByRole('searchbox', { name: 'Search observations' }), 'breakfast');
    expect(screen.queryByText('Covered ears during the fire drill')).toBeNull();
    expect(screen.getByText('Struggled with breakfast change')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /clear search/i }));

    await userEvent.selectOptions(screen.getByLabelText('Filter by context'), 'Daily routines');
    expect(screen.queryByText('Covered ears during the fire drill')).toBeNull();
    await userEvent.selectOptions(screen.getByLabelText('Filter by context'), '');

    await userEvent.type(screen.getByLabelText('What happened?'), 'Lined up toys');
    await userEvent.click(screen.getByLabelText('Mark as insufficient information'));
    await userEvent.click(screen.getByRole('button', { name: 'Save observation' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/api/observations')).toBe(true));
    const body = calls.find((c) => c.method === 'POST' && c.path === '/api/observations')!.body as Record<string, unknown>;
    expect(body).toMatchObject({ context: 'Classroom environments', what_happened: 'Lined up toys', insufficient_info: true });
    expect(body).not.toHaveProperty('id');
  });

  it('editing sends PUT with the id in the path, and removal asks first', async () => {
    window.location.hash = '#/context';
    installApi({ 'GET /api/observations': () => ok(OBS), 'PUT /api/observations/obs_1': () => ok({}), 'DELETE /api/observations/obs_2': () => ok({}) });
    render(<App />);
    await screen.findByText('Covered ears during the fire drill');
    await userEvent.click(screen.getAllByRole('button', { name: 'Edit' })[0]!);
    expect(screen.getByLabelText('What happened?')).toHaveValue('Covered ears during the fire drill');
    await userEvent.click(screen.getByRole('button', { name: 'Save observation' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'PUT' && c.path === '/api/observations/obs_1')).toBe(true));

    await userEvent.click(screen.getAllByRole('button', { name: 'Remove' })[1]!);
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel' }));
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false);
    await userEvent.click(screen.getAllByRole('button', { name: 'Remove' })[1]!);
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Remove' }));
    await waitFor(() => expect(calls.some((c) => c.method === 'DELETE' && c.path === '/api/observations/obs_2')).toBe(true));
  });
});

describe('Support Studio', () => {
  it('generates suggestions for an observation, loads one into the form, and saves it as a strategy', async () => {
    window.location.hash = '#/support';
    installApi({
      'GET /api/observations': () => ok(OBS),
      'POST /api/support/suggest': () => ok([{ title: 'Quiet corner', domain: 'Sensory', triggered_observation_id: 'obs_1', matched_terms: [{ term: 'ears', field: 'what_happened' }], why_useful: 'Reduces noise', how_to_try: 'Offer headphones', how_to_record: 'Note distress 1-5' }]),
      'POST /api/support/strategies': () => ok({ id: 'st_1' }),
    });
    render(<App />);
    await userEvent.selectOptions(await screen.findByLabelText('Observation'), 'obs_1');
    await userEvent.click(screen.getByRole('button', { name: 'Generate suggestions' }));
    expect(await screen.findByText('Quiet corner')).toBeInTheDocument();
    expect(screen.getByText(/“ears” \(what_happened\)/)).toBeInTheDocument();
    expect(calls.find((c) => c.path === '/api/support/suggest')?.body).toEqual({ observation_id: 'obs_1' });

    await userEvent.click(screen.getByRole('button', { name: 'Use this' }));
    expect(screen.getByLabelText('Title')).toHaveValue('Quiet corner');
    await userEvent.click(screen.getByRole('button', { name: 'Save strategy' }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/support/strategies' && c.method === 'POST')).toBe(true));
    expect(calls.find((c) => c.path === '/api/support/strategies' && c.method === 'POST')!.body).toMatchObject({
      title: 'Quiet corner', triggered_observation_id: 'obs_1', status: 'saved', source: 'observation-grounded rule suggestion (user-saved)',
    });
  });

  it('warns instead of calling the API when no observation is selected', async () => {
    window.location.hash = '#/support';
    installApi({ 'GET /api/observations': () => ok(OBS) });
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: 'Generate suggestions' }));
    expect(await within(screen.getByRole('region', { name: 'Notifications' })).findByText('Select an observation')).toBeInTheDocument();
    expect(calls.some((c) => c.path === '/api/support/suggest')).toBe(false);
  });

  it('"Mark tried/helped" uses the in-app prompt and records the attempt', async () => {
    window.location.hash = '#/support';
    installApi({
      'GET /api/support/strategies': () => ok([{ id: 'st_9', title: 'Visual schedule', status: 'saved' }]),
      'PUT /api/support/strategies/st_9': () => ok({}), 'POST /api/support/strategies/st_9/attempts': () => ok({}),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: 'Mark tried/helped' }));
    const dialog = screen.getByRole('dialog');
    await userEvent.clear(within(dialog).getByRole('textbox'));
    await userEvent.type(within(dialog).getByRole('textbox'), 'mixed');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(calls.some((c) => c.path === '/api/support/strategies/st_9/attempts')).toBe(true));
    expect(calls.find((c) => c.method === 'PUT' && c.path === '/api/support/strategies/st_9')!.body).toEqual({ status: 'tried', outcome: 'mixed' });
  });
});

describe('Timeline', () => {
  it('filters by kind and by search text', async () => {
    window.location.hash = '#/timeline';
    installApi({ 'GET /api/timeline': () => ok([
      { kind: 'observation', title: 'Fire drill', context: 'Classroom environments', event_time: '2026-11-01' },
      { kind: 'screening', title: 'Session finished', event_time: '2026-11-02' },
    ]) });
    render(<App />);
    expect(await screen.findByText('Fire drill')).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText('Filter by kind'), 'screening');
    expect(screen.queryByText('Fire drill')).toBeNull();
    expect(screen.getByText('Session finished')).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText('Filter by kind'), '');
    await userEvent.type(screen.getByRole('searchbox', { name: 'Search timeline' }), 'nothing-matches');
    expect(screen.getByText(/No timeline events yet/)).toBeInTheDocument();
  });
});

describe('RL Intelligence Lab', () => {
  it('replays a scenario and steps through it', async () => {
    window.location.hash = '#/rl-lab';
    installApi({
      'GET /api/rl/scenarios': () => ok({ scenarios: [{ id: 'syn_typical', name: 'Synthetic: typical' }], policies: [{ name: 'Greedy', kind: 'heuristic', status: 'available', reason: 'ok' }, { name: 'DQN', kind: 'rl', status: 'unavailable', reason: 'no checkpoint' }], budgets: [4, 6] }),
      'POST /api/rl/replay': () => ok({ ok: true, scenario: 'syn_typical', policy: 'greedy', budget: 6, steps: [{ n: 1 }, { n: 2 }, { n: 3 }], summary: { questions_asked: 3 }, trace: [{ item: 'A1' }] }),
    });
    render(<App />);
    expect(await screen.findByText('unavailable')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Step$/ })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Replay' }));
    expect(await screen.findByText('Step 1/3')).toBeInTheDocument();
    expect(calls.find((c) => c.path === '/api/rl/replay')!.body).toEqual({ scenario: 'syn_typical', policy: 'greedy', budget: 6 });
    await userEvent.click(screen.getByRole('button', { name: /^Step$/ }));
    expect(screen.getByText('Step 2/3')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Restart/ }));
    expect(screen.getByText('Step 1/3')).toBeInTheDocument();
  });

  it('shows a failed replay message rather than stale data', async () => {
    window.location.hash = '#/rl-lab';
    installApi({
      'GET /api/rl/scenarios': () => ok({ scenarios: [{ id: 's', name: 'S' }], policies: [], budgets: [6] }),
      'POST /api/rl/replay': () => ok({ ok: false, error: 'policy unavailable' }),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole('button', { name: 'Replay' }));
    expect(await screen.findByText('policy unavailable')).toBeInTheDocument();
  });
});

describe('Explainable AI', () => {
  it('loads the explanation of a completed session', async () => {
    window.location.hash = '#/explain';
    installApi({
      'GET /api/screening/sessions': () => ok([{ id: 'ses_a', status: 'completed', policy: 'greedy', mode: 'interactive' }, { id: 'ses_b', status: 'active', policy: 'greedy', mode: 'interactive' }]),
      'GET /api/screening/sessions/ses_a': () => ok({ id: 'ses_a', status: 'completed', policy: 'greedy', mode: 'interactive', result: { trace: [{ step: 1, item: 'A9', value: 1, belief_before: 0.5, belief_after: 0.8 }], explanation: { contributions: [{ item: 'A9', rank: 1, phi: 0.2, direction: 'raises' }], attribution: { baseline_p: 0.5, full_p: 0.8, method: 'shapley' } }, counterfactual: null } }),
    });
    render(<App />);
    const select = await screen.findByLabelText('Completed session');
    expect(within(select).queryByText(/ses_b/)).toBeNull(); // active sessions are not explainable
    expect(screen.getByRole('button', { name: 'Load explanation' })).toBeDisabled();
    await userEvent.selectOptions(select, 'ses_a');
    await userEvent.click(screen.getByRole('button', { name: 'Load explanation' }));
    expect(await screen.findByText('Group-Shapley contributions (observed responses)')).toBeInTheDocument();
    expect(screen.getByText('0.200000')).toBeInTheDocument();
    expect(screen.getByText(/No single-flip decision change/)).toBeInTheDocument();
  });
});

describe('Research & Analytics', () => {
  it('summarises known artifacts, says so when an H1 verdict is missing, and lists tables in a tab', async () => {
    window.location.hash = '#/research';
    installApi({ 'GET /api/research/dashboard': () => ok({
      limitations: ['Labels are questionnaire-derived'],
      artifacts: { 'predictor_saudi_metrics.json': { test_brier: 0.0123, test_auroc: 0.98, test_ece: 0.01 }, 'step5_policy_benchmark_saudi.json': { policies: ['greedy'], per_budget: [{ B: 3, policy: 'greedy', brier: 0.05, auroc: 0.9, items_asked_mean: 3 }] }, 'evoi_scale_saudi.json': null },
      tables: { 'results.csv': [{ a: 1, b: 'x' }] },
    }) });
    render(<App />);
    expect(await screen.findByText('Labels are questionnaire-derived')).toBeInTheDocument();
    expect(screen.getByText('0.0123')).toBeInTheDocument();
    expect(screen.getByText(/does not record an H1 verdict/)).toBeInTheDocument();
    expect(screen.getByText(/Artifact not available/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('tab', { name: /Tables/ }));
    expect(screen.getByText('results.csv')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'a' })).toBeInTheDocument();
  });
});
