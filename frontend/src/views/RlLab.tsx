import { useEffect, useState } from 'react';
import { ArrowRight, Pause, Play, RotateCcw, StepForward } from 'lucide-react';
import { api, apiOr, errorMessage } from '../api/client';
import type { RlCompare, RlReplay, RlScenarioMeta } from '../api/types';
import { Badge, Button, Card, DataTable, EmptyState, Field, FormStack, Grid, Item, ItemList, Mono, Notice, Row, Select, Split, useToast } from '../components/ui';
import { useStatus } from '../hooks/status';
import { useLoad } from '../hooks/useLoad';
import { fmt } from '../lib/format';
import { ViewState } from './ViewState';
import v from './Views.module.css';

const POLICY_CHOICES = ['greedy', 'random', 'beta_greedy', 'random_fixed'];
const FLOW = ['Current State', 'Policy Evaluation', 'Selected Question/Action', 'New Response', 'Updated State', 'Next Decision / Stop'];

export function RlLab() {
  const load = useLoad(() => apiOr<RlScenarioMeta>('/api/rl/scenarios', { scenarios: [], policies: [], budgets: [6] }));
  return <ViewState load={load}>{(meta) => <RlLabBody meta={meta} />}</ViewState>;
}

const STATUS_TEXT: Record<string, string> = { available: 'Available', available_if_data: 'Available with data', available_if_fitted: 'Needs a fitted model', available_limited: 'Limited' };
const statusText = (s: string) => STATUS_TEXT[s] ?? s.replace(/_/g, ' ');

function policyTone(status: string) {
  return status === 'available' ? 'good' : status.startsWith('available') ? 'warn' : 'danger';
}

function RlLabBody({ meta }: { meta: RlScenarioMeta }) {
  const toast = useToast();
  const { withStatus } = useStatus();
  const budgets = meta.budgets?.length ? meta.budgets : [1, 2, 3, 4, 5, 6];
  const [scenario, setScenario] = useState(meta.scenarios[0]?.id ?? '');
  const [policy, setPolicy] = useState('greedy');
  const [budget, setBudget] = useState(budgets.includes(6) ? 6 : (budgets[0] ?? 6));
  const [replay, setReplay] = useState<RlReplay | null>(null);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [compare, setCompare] = useState<RlCompare | null>(null);

  const steps = replay?.ok ? (replay.steps ?? []) : [];
  const last = Math.max(0, steps.length - 1);

  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setIdx((i) => {
        if (i >= last) { setPlaying(false); return i; }
        return i + 1;
      });
    }, 900);
    return () => window.clearInterval(t);
  }, [playing, last]);

  const run = async () => {
    try {
      const r = await withStatus('Replaying…', () => api<RlReplay>('/api/rl/replay', { method: 'POST', body: { scenario, policy, budget } }));
      setReplay(r); setIdx(0); setPlaying(false); toast.show('Replay ready', 'good');
    } catch (e) { toast.error(errorMessage(e)); }
  };

  const runCompare = async () => {
    try {
      setCompare(await withStatus('Comparing…', () => api<RlCompare>('/api/rl/compare', { method: 'POST', body: { scenario, policies: POLICY_CHOICES, budget } })));
    } catch (e) { toast.error(errorMessage(e)); }
  };

  const step = () => setIdx((i) => Math.min(i + 1, last));
  const current = steps[Math.min(idx, last)];

  return (
    <div className={v.stackLg}>
      <Split side={400}>
        <Card tone="raised" title="Replay configuration" subtitle="Synthetic, reproducible demonstration cases only. Replays use real policy/inference logic for available policies.">
          <FormStack>
            <Field label="Scenario" help="Synthetic cases (not real people).">
              <Select value={scenario} onChange={(e) => setScenario(e.target.value)}>{meta.scenarios.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}</Select>
            </Field>
            <Field label="Policy">
              <Select value={policy} onChange={(e) => setPolicy(e.target.value)}>{POLICY_CHOICES.map((p) => <option key={p} value={p}>{p} (available)</option>)}</Select>
            </Field>
            <Field label="Question budget">
              <Select value={budget} onChange={(e) => setBudget(Number(e.target.value))}>{budgets.map((b) => <option key={b} value={b}>{b}</option>)}</Select>
            </Field>
            <Row gap={8}>
              <Button onClick={() => void run()}>Replay</Button>
              <Button variant="secondary" onClick={() => void runCompare()}>Compare policies</Button>
            </Row>
            <Notice><strong>Policy availability:</strong> greedy/random/beta_greedy/random_fixed runnable live. DQN/PPO shown as unavailable without loaded checkpoints; exact DP is a limited reference (not run as a general live policy here).</Notice>
          </FormStack>
        </Card>
        <Card title="Policy status" subtitle="What can run live in this build, and why the rest cannot.">
          <ItemList dense>
            {meta.policies.map((p) => (
              <Item key={p.name} title={<>{p.name} <Badge tone={policyTone(p.status)}>{statusText(p.status)}</Badge></>}>
                <span>{p.kind} — {p.reason ?? ''}</span>
              </Item>
            ))}
          </ItemList>
        </Card>
      </Split>

      <Card title="Decision flow replay" subtitle="Each step shows the state, how the policy evaluated it, the question it chose, and the updated belief.">
        <div className={v.stepper}>
          <ol className={v.flow} aria-label="Decision flow">
            {FLOW.map((n, i) => (
              <li key={n} className={v.flowItem}><span className={`${v.node} ${i === FLOW.length - 1 ? v.nodeOn : ''}`}>{n}</span>{i < FLOW.length - 1 && <ArrowRight size={14} aria-hidden="true" className={v.arrow} />}</li>
            ))}
          </ol>
          {steps.length > 0 && (
            <div className={v.stepBar} role="group" aria-label="Replay position">
              {steps.map((_, i) => (
                <button key={i} type="button" aria-label={`Go to step ${i + 1}`} aria-current={i === Math.min(idx, last) ? 'step' : undefined}
                  className={`${v.stepSeg} ${i < idx ? v.stepSegDone : ''} ${i === Math.min(idx, last) ? v.stepSegNow : ''}`}
                  onClick={() => { setIdx(i); setPlaying(false); }} />
              ))}
            </div>
          )}
          <div className={v.transport}>
            <Button size="sm" icon={<Play size={14} aria-hidden="true" />} onClick={() => setPlaying(true)} disabled={!steps.length || playing}>Play</Button>
            <Button size="sm" variant="secondary" icon={<Pause size={14} aria-hidden="true" />} onClick={() => setPlaying(false)} disabled={!playing}>Pause</Button>
            <Button size="sm" variant="secondary" icon={<StepForward size={14} aria-hidden="true" />} onClick={step} disabled={!steps.length}>Step</Button>
            <Button size="sm" variant="ghost" icon={<RotateCcw size={14} aria-hidden="true" />} onClick={() => { setIdx(0); setPlaying(false); }} disabled={!steps.length}>Restart</Button>
            <Badge tone="accent">{steps.length ? `Step ${Math.min(idx, last) + 1}/${steps.length}` : 'Step 0'}</Badge>
          </div>
          <Grid cols={2}>
            <section>
              <h3 className={v.mb}>Current step</h3>
              {replay ? <Mono>{replay.ok === false ? (replay.error ?? 'Unavailable') : JSON.stringify(current ?? {}, null, 2)}</Mono> : <p className={v.muted}>Run a replay to inspect state, evaluation, chosen action, and belief updates.</p>}
            </section>
            <section>
              <h3 className={v.mb}>Decision history &amp; summary</h3>
              {replay?.ok ? <Mono>{JSON.stringify({ scenario: replay.scenario, policy: replay.policy, budget: replay.budget, synthetic: replay.synthetic, summary: replay.summary, trace: replay.trace }, null, 2)}</Mono> : <p className={v.muted}>Nothing to show yet.</p>}
            </section>
          </Grid>
        </div>
      </Card>

      <Card title="Policy comparison (same synthetic case)">
        {compare ? (
          <FormStack>
            <Notice>Synthetic comparison on <strong>{compare.scenario}</strong> (budget {compare.budget}). Uses real inference/policies where available.</Notice>
            <DataTable>
              <thead><tr><th>Policy</th><th>Available</th><th>Questions asked</th><th>Final p̂</th><th>Decision</th><th>Stop reason</th><th>Question order</th></tr></thead>
              <tbody>
                {compare.results.map((x, i) => x.ok
                  ? <tr key={i}><td>{x.policy}</td><td><Badge tone="good">yes</Badge></td><td>{x.summary?.questions_asked ?? '—'}</td><td>{fmt(x.summary?.final_p_hat)}</td><td>{x.summary?.decision || '—'}</td><td>{x.summary?.stop_reason || '—'}</td><td>{(x.trace ?? []).map((t) => t.item).join(' → ') || '—'}</td></tr>
                  : <tr key={i}><td>{x.policy ?? '—'}</td><td><Badge tone="danger">no</Badge></td><td colSpan={5}>{x.error ?? 'Unavailable'}</td></tr>)}
              </tbody>
            </DataTable>
          </FormStack>
        ) : <EmptyState>Run comparison to view question order, counts, belief/decision outcomes side-by-side.</EmptyState>}
      </Card>
    </div>
  );
}
