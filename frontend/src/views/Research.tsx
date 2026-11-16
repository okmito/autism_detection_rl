import { useState, type ReactNode } from 'react';
import { apiOr } from '../api/client';
import type { ResearchDashboard } from '../api/types';
import { Badge, Card, DataTable, EmptyState, KeyValue, Mono, Notice, Tabs } from '../components/ui';
import { useLoad } from '../hooks/useLoad';
import { fmt, joinList } from '../lib/format';
import { ViewState } from './ViewState';
import v from './Views.module.css';

type Obj = Record<string, unknown>;
const asRows = (v: unknown): Obj[] => (Array.isArray(v) ? (v as Obj[]) : []);
const txt = (v: unknown): string => (v === null || v === undefined ? '' : String(v));

function summarize(name: string, obj: Obj): ReactNode {
  try {
    if (name === 'predictor_saudi_metrics.json') {
      return <KeyValue rows={[['Test Brier', fmt(obj.test_brier, 4)], ['Test AUROC', fmt(obj.test_auroc, 4)], ['Test ECE', fmt(obj.test_ece, 4)], ['Source', txt(obj.source) || '—'], ['Note', txt(obj.note) || 'Questionnaire-derived labels; interpret as research/dev metric']]} />;
    }
    if (name === 'step5_policy_benchmark_saudi.json') {
      const rows = asRows(obj.per_budget ?? obj.rows);
      const h1 = obj.h1 as { supported?: unknown; greedy_brier?: unknown; exact_fixed_subset_brier?: unknown } | undefined;
      // This artifact does not always carry an H1 verdict; say so instead of printing "undefined".
      return (
        <div className={v.artifact}>
          {h1 && typeof h1.supported === 'boolean'
            ? <Notice tone={h1.supported ? 'good' : 'warn'}>H1 supported: <strong>{String(h1.supported)}</strong> (greedy {fmt(h1.greedy_brier, 4)} vs fixed {fmt(h1.exact_fixed_subset_brier, 4)}). Policies: {joinList(obj.policies) || '-'}.</Notice>
            : <Notice tone="warn">This artifact does not record an H1 verdict.{joinList(obj.policies) ? ` Policies: ${joinList(obj.policies)}.` : ''}</Notice>}
          {rows.length > 0 && (
            <DataTable maxHeight={420}>
              <thead><tr><th>B</th><th>Policy</th><th>Brier</th><th>AUROC</th><th>Items mean</th></tr></thead>
              <tbody>{rows.slice(0, 30).map((r, i) => <tr key={i}><td>{txt(r.B)}</td><td>{txt(r.policy)}</td><td>{fmt(r.brier, 4)}</td><td>{fmt(r.auroc, 4)}</td><td>{fmt(r.items_asked_mean, 2)}</td></tr>)}</tbody>
            </DataTable>
          )}
        </div>
      );
    }
    if (name === 'benchmark_multiseed_saudi.json') {
      const rows = asRows(obj.aggregate ?? obj.rows);
      return rows.length ? (
        <DataTable maxHeight={360}>
          <thead><tr><th>Policy</th><th>B</th><th>Mean Brier</th><th>Std</th></tr></thead>
          <tbody>{rows.slice(0, 30).map((r, i) => <tr key={i}><td>{txt(r.policy ?? r.name)}</td><td>{txt(r.B ?? r.budget)}</td><td>{fmt(r.mean_brier ?? r.brier_mean ?? r.brier, 4)}</td><td>{fmt(r.std_brier ?? r.brier_std, 4)}</td></tr>)}</tbody>
        </DataTable>
      ) : <Notice>Artifact loaded (structure varies); see raw.</Notice>;
    }
    if (name === 'outcome_explainability_saudi.json') return <Notice>Outcome explainability evaluation artifact loaded. Contains faithfulness/additivity/rank-stability metrics as produced by pipeline (see raw for details).</Notice>;
    if (name === 'lambda_sweep_saudi.json') return <Notice>λ sweep artifact loaded (V-6 evidence input; not a signed-off threshold). See raw for grid and saturation/degeneracy findings.</Notice>;
    if (name === 'evoi_scale_saudi.json') return <Notice>EVOI/scale analysis artifact loaded (decision input; no threshold selected). See raw.</Notice>;
  } catch { /* fall through to the generic notice */ }
  return <Notice>Artifact loaded. Summary view not specialized; inspect raw JSON.</Notice>;
}

/** One loaded artifact, expanded beneath the reference table. */
const SUMMARISED = new Set(['predictor_saudi_metrics.json', 'step5_policy_benchmark_saudi.json', 'benchmark_multiseed_saudi.json', 'outcome_explainability_saudi.json', 'lambda_sweep_saudi.json', 'evoi_scale_saudi.json']);
const hasSummary = (name: string) => SUMMARISED.has(name);

function ArtifactDetail({ name, obj, bare }: { name: string; obj: Obj; bare?: boolean }) {
  const raw = JSON.stringify(obj, null, 2);
  return (
    <section className={v.artifact}>
      <h3><code>{name}</code></h3>
      {bare ? null : summarize(name, obj)}
      <details><summary>Raw (truncated)</summary><div className={v.mt}><Mono>{raw.slice(0, 6000)}{JSON.stringify(obj).length > 6000 ? '…' : ''}</Mono></div></details>
    </section>
  );
}

export function Research() {
  const load = useLoad(() => apiOr<ResearchDashboard>('/api/research/dashboard', { artifacts: {}, tables: {}, limitations: [] }));
  return <ViewState load={load}>{(data) => <ResearchBody data={data} />}</ViewState>;
}

function ResearchBody({ data }: { data: ResearchDashboard }) {
  const [tab, setTab] = useState<'artifacts' | 'tables'>('artifacts');
  const keys = Object.keys(data.artifacts ?? {});
  const tables = Object.entries(data.tables ?? {});
  return (
    <div className={v.stackLg}>
      <Card title="Research integrity & limitations" subtitle="Offline research benchmarks are shown separately from live app inference. Metrics refer to their recorded splits/labels as per artifacts.">
        {(data.limitations ?? []).length > 0
          ? <Notice tone="warn"><ul>{(data.limitations ?? []).map((l, i) => <li key={i}>{l}</li>)}</ul></Notice>
          : <p className={v.muted}>No limitations recorded.</p>}
      </Card>
      <Tabs label="Research data" value={tab} onChange={setTab} items={[{ key: 'artifacts', label: `Artifacts (${keys.length})` }, { key: 'tables', label: `Tables (${tables.length})` }]}>
        {tab === 'artifacts' ? (
          keys.length ? (
            <div className={v.stackLg}>
              <DataTable>
                <thead><tr><th>Artifact</th><th>Status</th></tr></thead>
                <tbody>
                  {keys.map((k) => {
                    const loaded = Boolean(data.artifacts[k]);
                    return (
                      <tr key={k}>
                        <td><code>{k}</code></td>
                        <td>{loaded ? <Badge tone="good">Loaded</Badge> : <span className={v.muted}>Artifact not available in <code>results/</code></span>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </DataTable>
              {keys.filter((k) => data.artifacts[k] && hasSummary(k)).map((k) => <ArtifactDetail key={k} name={k} obj={data.artifacts[k] as Obj} />)}
              {keys.some((k) => data.artifacts[k] && !hasSummary(k)) && (
                <details>
                  <summary>Other loaded artifacts ({keys.filter((k) => data.artifacts[k] && !hasSummary(k)).length}): no summary view yet, raw JSON only</summary>
                  <div className={v.stackLg}>
                    {keys.filter((k) => data.artifacts[k] && !hasSummary(k)).map((k) => <ArtifactDetail key={k} name={k} obj={data.artifacts[k] as Obj} bare />)}
                  </div>
                </details>
              )}
            </div>
          ) : <EmptyState>No research artifacts found in results/.</EmptyState>
        ) : (
          <Card title="Tabular artifacts (CSV)">
            {tables.length ? (
              <div className={v.stackLg}>
                {tables.map(([name, rows]) => {
                  if (!rows || !rows.length) return <p key={name} className={v.muted}>{name} not available.</p>;
                  const cols = Object.keys(rows[0] ?? {});
                  return (
                    <div key={name} className={v.artifact}>
                      <h3>{name}</h3>
                      <DataTable maxHeight={360}>
                        <thead><tr>{cols.map((h) => <th key={h}>{h}</th>)}</tr></thead>
                        <tbody>{rows.slice(0, 80).map((r, i) => <tr key={i}>{cols.map((h) => <td key={h}>{txt(r[h])}</td>)}</tr>)}</tbody>
                      </DataTable>
                    </div>
                  );
                })}
              </div>
            ) : <EmptyState>No CSV tables loaded.</EmptyState>}
          </Card>
        )}
      </Tabs>
    </div>
  );
}
