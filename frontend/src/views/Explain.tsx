import { useState } from 'react';
import { api, apiOr, errorMessage } from '../api/client';
import type { Json, ScreeningResult, SessionDetail, SessionSummary } from '../api/types';
import { BarList, Button, ButtonLink, Card, DataTable, EmptyState, Field, FormStack, Grid, Notice, Select, Split, useToast } from '../components/ui';
import { useLoad } from '../hooks/useLoad';
import { fmt, fmtDate } from '../lib/format';
import { TraceTable } from './screening/parts';
import { buildSteps } from './screening/ActiveSession';
import { BeliefTrace, Pathway, RiskScale } from './screening/Pathway';
import { ViewState } from './ViewState';
import v from './Views.module.css';

export function Explain() {
  const load = useLoad(() => apiOr<SessionSummary[]>('/api/screening/sessions', []));
  return <ViewState load={load}>{(sessions) => <ExplainBody completed={sessions.filter((x) => x.status === 'completed')} />}</ViewState>;
}

function ExplainBody({ completed }: { completed: SessionSummary[] }) {
  const toast = useToast();
  const [sel, setSel] = useState('');
  const [result, setResult] = useState<ScreeningResult | null | undefined>(undefined);
  const [loading, setLoading] = useState(false);

  const loadExplanation = async () => {
    if (!sel) return;
    setLoading(true);
    try { setResult((await api<SessionDetail>(`/api/screening/sessions/${sel}`)).result ?? null); }
    catch (e) { toast.error(errorMessage(e)); } finally { setLoading(false); }
  };

  const ex = result?.explanation ?? {};
  const contribs = ex.contributions ?? [];
  const cf = result?.counterfactual;

  return (
    <div className={v.stackLg}>
      <Split side={380}>
        <Card tone="raised" title="Explain a completed screening session" subtitle="Uses existing explanation outputs (group-Shapley attribution, counterfactuals, belief trace, uncertainty/limitations) computed by the backend.">
          <FormStack>
            <Field label="Completed session">
              <Select value={sel} onChange={(e) => setSel(e.target.value)}>
                <option value="">Select</option>
                {completed.map((x) => <option key={x.id} value={x.id}>{fmtDate(x.updated_at) || 'Session'} · {x.policy} · {x.id}</option>)}
              </Select>
            </Field>
            <div><Button onClick={() => void loadExplanation()} loading={loading} disabled={!sel}>Load explanation</Button></div>
            <Notice>Attribution ≠ causality. Explanations describe model behavior on observed inputs. &quot;Supporting/opposing&quot; reflect contribution sign relative to model baseline, not clinical evidence.</Notice>
          </FormStack>
        </Card>
        {result !== undefined ? <Card title="Belief trace / decision path (from session)">
          {result ? (
            <FormStack>
              <RiskScale value={result.p_hat ?? result.explanation?.attribution?.full_p} tau={result.explanation?.outcome?.tau ?? 0.5} />
              {(result.trace ?? []).length > 0 && <BeliefTrace trace={result.trace ?? []} tau={result.explanation?.outcome?.tau ?? 0.5} caption="Estimate after each answer. Dashed points are skipped questions." />}
              {(result.trace ?? []).length > 0 && <Pathway label="Decision path" steps={buildSteps(result.trace ?? [], {})} endLabel={`Stopped: ${result.stop_reason ?? 'finished'}`} />}
              <details>
                <summary>Belief numbers per step</summary>
                <div className={v.mt}><TraceTable trace={result.trace ?? []} /></div>
              </details>
            </FormStack>
          ) : <EmptyState>No result.</EmptyState>}
        </Card> : (
          <EmptyState>Pick a completed session and choose <strong>Load explanation</strong> to see its decision path, attributions and counterfactuals.</EmptyState>
        )}
      </Split>

      {result !== undefined && (
        <Card title="Attribution & counterfactuals">
          {result === null ? <EmptyState>No result.</EmptyState> : (
            <Grid cols={2}>
              <section className={v.artifact}>
                <h3>Group-Shapley contributions (observed responses)</h3>
                {contribs.length ? (
                  <>
                    <BarList rows={contribs.slice(0, 10).map((c) => ({ label: `${c.item} (${c.direction ?? ''})`, value: Number(c.phi ?? 0), display: fmt(c.phi, 4) }))} />
                    <DataTable>
                      <thead><tr><th>Item</th><th>Rank</th><th>φ (contribution)</th><th>Direction</th></tr></thead>
                      <tbody>{contribs.map((c, i) => <tr key={i}><td>{c.item}</td><td>{c.rank ?? ''}</td><td>{fmt(c.phi, 6)}</td><td>{c.direction ?? ''}</td></tr>)}</tbody>
                    </DataTable>
                  </>
                ) : <EmptyState>No contributions.</EmptyState>}
                <p className={v.matched}>Baseline p: {fmt(ex.attribution?.baseline_p)} → full p: {fmt(ex.attribution?.full_p)} · additivity error: {fmt(ex.attribution?.additivity_error, 8)} · method: {ex.attribution?.method ?? ''}</p>
              </section>
              <section className={v.artifact}>
                <h3>Counterfactuals &amp; uncertainty</h3>
                {cf ? <Notice>Single-answer flip: {cf.item ?? ''} {String(cf.original_value)}→{String(cf.flipped_value)} gives an estimate of {fmt(cf.new_p)}.</Notice> : <Notice>No single-flip decision change found (per backend counterfactual search) or unavailable.</Notice>}
                {ex.uncertainty && <Notice><strong>Uncertainty (model):</strong> {JSON.stringify(ex.uncertainty as Json)}</Notice>}
                {ex.limitations && ex.limitations.length > 0 && <Notice tone="warn"><strong>Limitations:</strong><ul>{ex.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul></Notice>}
                <Notice>{ex.disclaimer ?? 'Research prototype; screening/referral-research only, not a diagnosis.'}</Notice>
              </section>
            </Grid>
          )}
        </Card>
      )}

      <Card title="Synthetic policy explanation (RL Lab)" subtitle="For step-by-step policy evaluation reasons on synthetic cases, use the RL Intelligence Lab replay (shows state, evaluation, chosen action, updates).">
        <ButtonLink variant="secondary" href="#/rl-lab">Open RL Intelligence Lab</ButtonLink>
      </Card>
    </div>
  );
}
