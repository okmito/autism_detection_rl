import type { PolicyInfo, RlDecision, RlPolicyMeta, SelectionDiag, TraceStep } from '../../api/types';
import { DataTable, EmptyState, Notice } from '../../components/ui';
import { fmt } from '../../lib/format';

export function TraceTable({ trace }: { trace: TraceStep[] }) {
  if (!trace.length) return <EmptyState>No answers recorded yet.</EmptyState>;
  return (
    <DataTable>
      <thead><tr><th>#</th><th>Item</th><th>Value</th><th>Belief before</th><th>Belief after</th></tr></thead>
      <tbody>
        {trace.map((t, i) => (
          <tr key={i}>
            <td>{t.step}</td><td>{t.item}</td><td>{t.skipped ? 'skipped' : String(t.value ?? '')}</td>
            <td>{fmt(t.belief_before)}</td><td>{fmt(t.belief_after)}</td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

export function FallbackBanner({ info }: { info?: PolicyInfo | null }) {
  if (!info?.fallback) return null;
  const f = info.fallback;
  return (
    <Notice tone="warn">
      <strong>Fallback in use.</strong> {f.requested} was requested but is unavailable ({f.reason ?? ''}); this session uses {f.used}.
    </Notice>
  );
}

export function rlAvailabilityText(info?: RlPolicyMeta): string {
  if (!info) return '';
  if (info.available) return `RL policy loaded (${info.agent}, trained on ${info.trained_on}).`;
  return `RL policy unavailable: ${info.reason ?? 'unknown reason'}. Choosing it falls back to greedy, and the session is labelled as a fallback.`;
}

export function DecisionNotes({ d }: { d: RlDecision }) {
  const vals = (d.action_values ?? []).slice(0, 5);
  return (
    <div style={{ display: 'grid', gap: 8 }}>
      <Notice><ul>{(d.narrative ?? []).map((l, i) => <li key={i}>{l}</li>)}</ul></Notice>
      {vals.length > 0 && (
        <DataTable>
          <thead><tr><th>Option</th><th>Agent value</th></tr></thead>
          <tbody>{vals.map((v, i) => <tr key={i}><td>{v.action}</td><td>{fmt(v.q, 4)}</td></tr>)}</tbody>
        </DataTable>
      )}
      <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
        Values are the agent&apos;s own estimates ({d.value_units ?? ''}). {d.faithful_to_agent_internals ? "Exact decomposition of the agent's value." : 'Answer probabilities come from the belief model, not the network internals.'}
      </p>
    </div>
  );
}

export function selectionText(sel?: SelectionDiag | null): string {
  if (!sel) return 'Selection diagnostic unavailable for this step. The policy chose the next question according to its implemented rule.';
  if (sel.applicable === false) return 'Random/uniform policy: no information-gain ranking to display; action chosen from legal items.';
  if (sel.criterion_vacuous) return `Criterion vacuous/tied on current support (support size ${sel.support_size ?? '—'}, pure: ${String(sel.support_is_pure)}). Policy falls back deterministically among tied items; this reflects data/support state, not a UI error.`;
  const top = (sel.top_items ?? []).map((t) => `${t.code}(${fmt(t.ig, 4)})`).join(', ') || '—';
  return `Support-consistent records: ${sel.support_size ?? '—'} (positive: ${sel.n_pos ?? '—'}). Support pure: ${String(sel.support_is_pure)}. IG spread: ${fmt(sel.ig_spread, 6)}; tied at max: ${sel.tie_at_max ?? '—'}. Top: ${top}.`;
}
