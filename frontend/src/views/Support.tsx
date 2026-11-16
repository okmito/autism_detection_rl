import { useState, type FormEvent } from 'react';
import { api, apiOr, errorMessage } from '../api/client';
import type { Observation, Strategy, Suggestion } from '../api/types';
import { Badge, Button, Card, EmptyState, Field, FormStack, Input, Item, ItemList, Row, Select, Split, Textarea, useDialogs, useToast } from '../components/ui';
import { useLoad } from '../hooks/useLoad';
import { ViewState } from './ViewState';
import v from './Views.module.css';

interface StratForm { id: string; title: string; domain: string; obs: string; why: string; how: string; rec: string; status: string; outcome: string; notes: string }
const EMPTY: StratForm = { id: '', title: '', domain: '', obs: '', why: '', how: '', rec: '', status: 'saved', outcome: '', notes: '' };

function readPreselected(): string {
  try { return sessionStorage.getItem('support_obs') ?? ''; } catch { return ''; }
}

export function Support() {
  const load = useLoad(async () => {
    const [strategies, observations] = await Promise.all([
      apiOr<Strategy[]>('/api/support/strategies', []),
      apiOr<Observation[]>('/api/observations', []),
    ]);
    return { strategies, observations };
  });
  return <ViewState load={load}>{(d) => <SupportBody {...d} reload={load.reload} />}</ViewState>;
}

function SupportBody({ strategies, observations, reload }: { strategies: Strategy[]; observations: Observation[]; reload: () => void }) {
  const toast = useToast();
  const { confirm, prompt } = useDialogs();
  const [obsId, setObsId] = useState(() => (observations.some((o) => o.id === readPreselected()) ? readPreselected() : ''));
  const [sugs, setSugs] = useState<Suggestion[] | null>(null);
  const [form, setForm] = useState<StratForm>(EMPTY);
  const [generating, setGenerating] = useState(false);
  const set = <K extends keyof StratForm>(k: K, v: StratForm[K]) => setForm((f) => ({ ...f, [k]: v }));

  const generate = async () => {
    if (!obsId) { toast.show('Select an observation', 'warn'); return; }
    setGenerating(true);
    try { setSugs(await api<Suggestion[]>('/api/support/suggest', { method: 'POST', body: { observation_id: obsId } })); }
    catch (e) { toast.error(errorMessage(e)); } finally { setGenerating(false); }
  };

  const useSuggestion = (s: Suggestion) => {
    setForm((f) => ({ ...f, title: s.title, domain: s.domain ?? '', obs: s.triggered_observation_id ?? obsId, why: s.why_useful ?? '', how: s.how_to_try ?? '', rec: s.how_to_record ?? '' }));
    toast.show('Suggestion loaded into form');
  };

  const save = async (e: FormEvent) => {
    e.preventDefault();
    const body = { title: form.title, domain: form.domain, triggered_observation_id: form.obs, why_useful: form.why, how_to_try: form.how, how_to_record: form.rec, status: form.status, outcome: form.outcome, notes: form.notes, source: 'observation-grounded rule suggestion (user-saved)' };
    try {
      if (form.id) await api(`/api/support/strategies/${form.id}`, { method: 'PUT', body });
      else await api('/api/support/strategies', { method: 'POST', body });
      toast.show('Strategy saved', 'good');
      setForm(EMPTY);
      reload();
    } catch (err) { toast.error(errorMessage(err)); }
  };

  const edit = (s: Strategy) => {
    setForm({ id: s.id, title: s.title ?? '', domain: s.domain ?? '', obs: s.triggered_observation_id ?? '', why: s.why_useful ?? '', how: s.how_to_try ?? '', rec: s.how_to_record ?? '', status: s.status ?? 'saved', outcome: s.outcome ?? '', notes: s.notes ?? '' });
    window.scrollTo({ top: 0 });
  };

  const markTried = async (s: Strategy) => {
    const outcome = await prompt({ title: 'Mark as tried', label: 'Outcome/usefulness (e.g. helped, mixed, did not help)', defaultValue: 'helpful', confirmLabel: 'Save' });
    if (outcome === null) return;
    try {
      await api(`/api/support/strategies/${s.id}`, { method: 'PUT', body: { status: 'tried', outcome } });
      await api(`/api/support/strategies/${s.id}/attempts`, { method: 'POST', body: { status: 'tried', outcome } });
      toast.show('Updated', 'good'); reload();
    } catch (e) { toast.error(errorMessage(e)); }
  };

  const remove = async (s: Strategy) => {
    if (!(await confirm({ title: 'Remove strategy?', message: 'Remove this strategy?', confirmLabel: 'Remove', danger: true }))) return;
    try { await api(`/api/support/strategies/${s.id}`, { method: 'DELETE' }); toast.show('Removed'); reload(); }
    catch (e) { toast.error(errorMessage(e)); }
  };

  return (
    <Split side={420}>
      <Card tone="raised" title={form.id ? 'Edit strategy' : 'Save / track strategy'} subtitle="Load a suggestion, adjust it, and keep notes on what you try.">
        <form onSubmit={save}>
          <FormStack>
            <div className={v.pair}>
              <Field label="Title"><Input required value={form.title} onChange={(e) => set('title', e.target.value)} /></Field>
              <Field label="Domain/context"><Input value={form.domain} onChange={(e) => set('domain', e.target.value)} /></Field>
            </div>
            <Field label="Triggered observation ID"><Input value={form.obs} onChange={(e) => set('obs', e.target.value)} /></Field>
            <Field label="Why this may be useful"><Textarea value={form.why} onChange={(e) => set('why', e.target.value)} /></Field>
            <Field label="How to try"><Textarea value={form.how} onChange={(e) => set('how', e.target.value)} /></Field>
            <Field label="How usefulness can be recorded"><Textarea value={form.rec} onChange={(e) => set('rec', e.target.value)} /></Field>
            <Field label="Status">
              <Select value={form.status} onChange={(e) => set('status', e.target.value)}>
                <option value="saved">Saved</option><option value="tried">Tried</option><option value="helpful">Helped</option><option value="not_helpful">Did not help</option><option value="mixed">Mixed/unclear</option>
              </Select>
            </Field>
            <Field label="Outcome / usefulness notes"><Textarea value={form.outcome} onChange={(e) => set('outcome', e.target.value)} /></Field>
            <Field label="Personal notes"><Textarea value={form.notes} onChange={(e) => set('notes', e.target.value)} /></Field>
            <Row gap={8}><Button type="submit">Save strategy</Button><Button variant="ghost" onClick={() => setForm(EMPTY)}>Clear</Button></Row>
          </FormStack>
        </form>
      </Card>

      <div className={v.stackLg}>
        <Card title="Suggest from observation" subtitle="Suggestions are non-diagnostic and grounded in recorded observations (rule-based, transparent).">
          <FormStack>
            <Field label="Observation">
              <Select value={obsId} onChange={(e) => setObsId(e.target.value)}>
                <option value="">Select an observation</option>
                {observations.map((o) => <option key={o.id} value={o.id}>{o.context} — {(o.what_happened ?? '').slice(0, 60)}</option>)}
              </Select>
            </Field>
            <div><Button variant="secondary" onClick={() => void generate()} loading={generating}>Generate suggestions</Button></div>
            {sugs && (sugs.length ? (
              <ItemList>
                {sugs.map((s, i) => (
                  <Item key={i} title={s.title} actions={<Button size="sm" variant="secondary" onClick={() => useSuggestion(s)}>Use this</Button>}>
                    <p><strong>Triggered by observation:</strong> {s.triggered_observation_id}</p>
                    <p className={v.matched}><strong>Matched:</strong> {(s.matched_terms ?? []).length ? (s.matched_terms ?? []).map((m) => `“${m.term}” (${m.field})`).join(', ') : 'no keyword (general check-in)'} · simple keyword rules, so dismiss it if the match is wrong</p>
                    <p><strong>Why:</strong> {s.why_useful}</p>
                    <p><strong>How to try:</strong> {s.how_to_try}</p>
                    <p><strong>Record usefulness:</strong> {s.how_to_record}</p>
                  </Item>
                ))}
              </ItemList>
            ) : <EmptyState>No suggestions.</EmptyState>)}
          </FormStack>
        </Card>

        <Card title="Saved strategies & history" subtitle={strategies.length ? `${strategies.length} saved` : undefined}>
          {strategies.length ? (
            <ItemList>
              {strategies.map((s) => (
                <Item key={s.id} title={<>{s.title} <Badge tone={s.status === 'helpful' ? 'good' : s.status === 'not_helpful' ? 'danger' : s.status === 'mixed' ? 'warn' : s.status === 'tried' ? 'accent' : 'neutral'}>{s.status || 'saved'}</Badge></>}
                  actions={<>
                    <Button size="sm" variant="secondary" onClick={() => edit(s)}>Edit</Button>
                    <Button size="sm" variant="ghost" onClick={() => void markTried(s)}>Mark tried/helped</Button>
                    <Button size="sm" variant="dangerQuiet" onClick={() => void remove(s)}>Remove</Button>
                  </>}>
                  {(s.domain || s.triggered_observation_id) && <p className={v.fieldsMeta}>{[s.domain, s.triggered_observation_id && `from observation ${s.triggered_observation_id}`].filter(Boolean).join(' · ')}</p>}
                  {s.why_useful && <p><strong>Why useful:</strong> {s.why_useful}</p>}
                  {s.how_to_try && <p><strong>How to try:</strong> {s.how_to_try}</p>}
                  {s.how_to_record && <p><strong>Record usefulness:</strong> {s.how_to_record}</p>}
                  {s.outcome && <p><strong>Outcome:</strong> {s.outcome}</p>}
                  {s.notes && <p><strong>Notes:</strong> {s.notes}</p>}
                </Item>
              ))}
            </ItemList>
          ) : <EmptyState>No strategies saved yet. Generate suggestions from an observation.</EmptyState>}
        </Card>
      </div>
    </Split>
  );
}
