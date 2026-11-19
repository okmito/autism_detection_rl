import { useMemo, useState, type FormEvent } from 'react';
import { api, apiOr, errorMessage } from '../api/client';
import type { Observation } from '../api/types';
import { Badge, Button, Card, Checkbox, EmptyState, Field, FormStack, Input, Item, ItemList, Row, SearchField, Select, Split, Textarea, useDialogs, useToast } from '../components/ui';
import { useLoad } from '../hooks/useLoad';
import { fmtDate, toLocalInput } from '../lib/format';
import { ViewState } from './ViewState';
import s from './Views.module.css';

export const CONTEXTS = ['Classroom environments', 'Social interaction and play', 'Daily routines', 'Sensory environments', 'Communication contexts', 'Transitions and unexpected changes'];

interface FormState {
  id: string; context: string; occurred_at: string; what_happened: string; frequency: string;
  difficulty: string; what_helped: string; consistency: string; insufficient_info: boolean; notes: string;
}
const EMPTY: FormState = { id: '', context: CONTEXTS[0] ?? '', occurred_at: '', what_happened: '', frequency: '', difficulty: '', what_helped: '', consistency: '', insufficient_info: false, notes: '' };

export function Context() {
  const load = useLoad(() => apiOr<Observation[]>('/api/observations', []));
  return <ViewState load={load}>{(obs) => <ContextBody observations={obs} reload={load.reload} />}</ViewState>;
}

function ContextBody({ observations, reload }: { observations: Observation[]; reload: () => void }) {
  const toast = useToast();
  const { confirm } = useDialogs();
  const [form, setForm] = useState<FormState>(EMPTY);
  const [query, setQuery] = useState('');
  const [ctxFilter, setCtxFilter] = useState('');
  const [saving, setSaving] = useState(false);
  const set = <K extends keyof FormState>(k: K, v: FormState[K]) => setForm((f) => ({ ...f, [k]: v }));

  const items = useMemo(() => {
    const q = query.toLowerCase();
    return observations.filter((o) => (!ctxFilter || o.context === ctxFilter)
      && (!q || `${o.what_happened} ${o.difficulty ?? ''} ${o.what_helped ?? ''} ${o.notes ?? ''}`.toLowerCase().includes(q)));
  }, [observations, query, ctxFilter]);

  const save = async (e: FormEvent) => {
    e.preventDefault();
    const { id, ...body } = form;
    setSaving(true);
    try {
      if (id) await api(`/api/observations/${id}`, { method: 'PUT', body });
      else await api('/api/observations', { method: 'POST', body });
      toast.show('Observation saved', 'good');
      setForm(EMPTY);
      reload();
    } catch (err) { toast.error(errorMessage(err)); } finally { setSaving(false); }
  };

  const edit = (o: Observation) => {
    setForm({
      id: o.id, context: o.context, occurred_at: o.occurred_at ? toLocalInput(o.occurred_at) : '', what_happened: o.what_happened ?? '',
      frequency: o.frequency ?? '', difficulty: o.difficulty ?? '', what_helped: o.what_helped ?? '', consistency: o.consistency ?? '',
      insufficient_info: !!o.insufficient_info, notes: o.notes ?? '',
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const remove = async (o: Observation) => {
    if (!(await confirm({ title: 'Remove observation?', message: 'Remove this observation record?', confirmLabel: 'Remove', danger: true }))) return;
    try { await api(`/api/observations/${o.id}`, { method: 'DELETE' }); toast.show('Observation removed'); reload(); }
    catch (err) { toast.error(errorMessage(err)); }
  };

  return (
    <Split side={420}>
      <Card tone="raised" title={form.id ? 'Edit observation' : 'Record observation'} subtitle="Contextual observations are stored separately from validated questionnaire responses.">
        <form onSubmit={save}>
          <FormStack>
            <Field label="Context"><Select required value={form.context} onChange={(e) => set('context', e.target.value)}>{CONTEXTS.map((c) => <option key={c} value={c}>{c}</option>)}</Select></Field>
            <Field label="What happened?"><Textarea required value={form.what_happened} onChange={(e) => set('what_happened', e.target.value)} /></Field>
            <div className={s.pair}>
              <Field label="When observed"><Input type="datetime-local" value={form.occurred_at} onChange={(e) => set('occurred_at', e.target.value)} /></Field>
              <Field label="How frequently observed?"><Input value={form.frequency} placeholder="e.g. daily" onChange={(e) => set('frequency', e.target.value)} /></Field>
            </div>
            <Field label="What seemed difficult?"><Textarea value={form.difficulty} onChange={(e) => set('difficulty', e.target.value)} /></Field>
            <Field label="What appeared to help?"><Textarea value={form.what_helped} onChange={(e) => set('what_helped', e.target.value)} /></Field>
            <Field label="Consistency">
              <Select value={form.consistency} onChange={(e) => set('consistency', e.target.value)}>
                <option value="">Select</option><option>Consistent</option><option>Occasional</option><option>Insufficient information</option>
              </Select>
            </Field>
            <Checkbox label="Mark as insufficient information" checked={form.insufficient_info} onChange={(v) => set('insufficient_info', v)} />
            <Field label="Notes"><Textarea value={form.notes} onChange={(e) => set('notes', e.target.value)} /></Field>
            <Row gap={8}>
              <Button type="submit" loading={saving}>Save observation</Button>
              <Button variant="ghost" onClick={() => setForm(EMPTY)}>Clear</Button>
            </Row>
          </FormStack>
        </form>
      </Card>
      <Card title="Observation history" subtitle={`${observations.length} recorded`}>
        <FormStack>
          <div className={s.toolbar}>
            <SearchField label="Search observations" value={query} onChange={setQuery} />
            <Select aria-label="Filter by context" value={ctxFilter} onChange={(e) => setCtxFilter(e.target.value)}>
              <option value="">All contexts</option>{CONTEXTS.map((c) => <option key={c} value={c}>{c}</option>)}
            </Select>
          </div>
          {items.length ? (
            <ItemList>
              {items.map((o) => (
                <Item key={o.id} title={<>{o.context} <Badge>{o.consistency || '—'}</Badge></>}
                  actions={<>
                    <Button size="sm" variant="secondary" onClick={() => edit(o)}>Edit</Button>
                    <Button size="sm" variant="ghost" onClick={() => void remove(o)}>Remove</Button>
                    <a href="#/support" onClick={() => { try { sessionStorage.setItem('support_obs', o.id); } catch { /* optional convenience */ } }} className={s.useLink}>Use in Support Studio</a>
                  </>}>
                  <p className={s.what}>{o.what_happened}</p>
                  <p className={s.when}>{fmtDate(o.occurred_at ?? o.created_at) || 'Date not recorded'} · {o.frequency || 'Frequency not recorded'} {o.insufficient_info && <Badge tone="warn">Insufficient info</Badge>}</p>
                  {o.difficulty && <p><strong>Difficulty:</strong> {o.difficulty}</p>}
                  {o.what_helped && <p><strong>What helped:</strong> {o.what_helped}</p>}
                  {o.notes && <p><strong>Notes:</strong> {o.notes}</p>}
                </Item>
              ))}
            </ItemList>
          ) : <EmptyState>No observations match. Add one using the form.</EmptyState>}
        </FormStack>
      </Card>
    </Split>
  );
}
