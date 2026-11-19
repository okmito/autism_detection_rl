import { useState } from 'react';
import { Download, Trash2 } from 'lucide-react';
import { api, apiOr, errorMessage } from '../api/client';
import type { ConsentRecord } from '../api/types';
import { Badge, Button, Card, Checkbox, EmptyState, Field, FormStack, Item, ItemList, Select, Split, Textarea, useDialogs, useToast } from '../components/ui';
import { useLoad } from '../hooks/useLoad';
import { downloadJson, fmtDate } from '../lib/format';
import { ViewState } from './ViewState';
import v from './Views.module.css';

const SCOPES: [string, string][] = [
  ['general_research_use', 'General research/prototype use'],
  ['store_observations', 'Store observations & support plans locally'],
  ['store_screening_sessions', 'Store screening sessions locally'],
];

const DESTRUCTIVE = [
  { path: '/api/privacy/delete_observations', label: 'Delete all observations', title: 'Delete all observations?', message: 'Delete ALL app-stored observations? This cannot be undone.', done: 'Observations deleted' },
  { path: '/api/privacy/delete_sessions', label: 'Delete all screening sessions', title: 'Delete all screening sessions?', message: 'Delete ALL stored screening sessions (answers, results and reports)? This cannot be undone.', done: 'Screening sessions deleted' },
  { path: '/api/privacy/delete_all', label: 'Delete all app data', title: 'Delete all app data?', message: 'Delete ALL app data: observations, support strategies, screening sessions and the timeline? The consent log is kept. This cannot be undone.', done: 'All app data deleted' },
] as const;

export function Privacy() {
  const load = useLoad(() => apiOr<ConsentRecord[]>('/api/consent', []));
  return <ViewState load={load}>{(consent) => <PrivacyBody consent={consent} reload={load.reload} />}</ViewState>;
}

function PrivacyBody({ consent, reload }: { consent: ConsentRecord[]; reload: () => void }) {
  const toast = useToast();
  const { confirm } = useDialogs();
  const [scope, setScope] = useState(SCOPES[0]?.[0] ?? '');
  const [granted, setGranted] = useState(false);
  const [details, setDetails] = useState('');
  const [busy, setBusy] = useState('');

  const saveConsent = async () => {
    setBusy('consent');
    try { await api('/api/consent', { method: 'POST', body: { scope, granted, details } }); toast.show('Consent saved', 'good'); setDetails(''); reload(); }
    catch (e) { toast.error(errorMessage(e)); } finally { setBusy(''); }
  };

  const exportAll = async () => {
    setBusy('export');
    try { downloadJson(await api('/api/timeline/export', { method: 'POST' }), 'neurobridge_export.json'); toast.show('Export downloaded', 'good'); }
    catch (e) { toast.error(errorMessage(e)); } finally { setBusy(''); }
  };

  const destroy = async (d: (typeof DESTRUCTIVE)[number]) => {
    if (!(await confirm({ title: d.title, message: d.message, confirmLabel: 'Delete', danger: true }))) return;
    setBusy(d.path);
    try { await api(d.path, { method: 'POST' }); toast.show(d.done); reload(); }
    catch (e) { toast.error(errorMessage(e)); } finally { setBusy(''); }
  };

  return (
    <div className={v.stackLg}>
      <Split side={420}>
        <Card tone="raised" title="Consent" subtitle="Keep a log of your consent choices for this research prototype. This is a record only: it does not switch any feature on or off, and app data is stored on this computer either way.">
          <FormStack>
            <Field label="Scope"><Select value={scope} onChange={(e) => setScope(e.target.value)}>{SCOPES.map(([val, l]) => <option key={val} value={val}>{l}</option>)}</Select></Field>
            <Checkbox checked={granted} onChange={setGranted} label="I understand this is a non-diagnostic research prototype and I consent to the selected local data handling." />
            <Field label="Details (optional)"><Textarea value={details} onChange={(e) => setDetails(e.target.value)} /></Field>
            <div><Button onClick={() => void saveConsent()} loading={busy === 'consent'}>Save consent record</Button></div>
          </FormStack>
        </Card>

        <div className={v.stackLg}>
          <Card title="Privacy information">
            <div className={v.rows}>
              <div className={v.settingRow}><h3>What is stored locally</h3><p>Screening sessions/traces, observations, support strategies/attempts, timeline events, and consent records are stored in a local app database (<code>data/neurobridge.db</code> by default). Development research datasets/artifacts remain in the project as-is.</p></div>
              <div className={v.settingRow}><h3>Data minimization &amp; safety</h3><p>Provide only information you are comfortable recording. Avoid entering unnecessary identifying details in free-text notes. No data is sent to external AI services by this app. Logs should not be used to expose sensitive content.</p></div>
              <div className={v.settingRow}><h3>Not a diagnostic or clinical system</h3><p>Outputs are for screening/research and support-planning exploration only, not medical advice or diagnosis. No claims of clinical/regulatory compliance are made.</p></div>
            </div>
          </Card>

          <Card title="Consent records" subtitle={consent.length ? `${consent.length} recorded` : undefined}>
            {consent.length ? (
              <ItemList>
                {consent.map((c, i) => (
                  <Item key={i} title={<>{c.scope} <Badge tone={c.granted ? 'good' : 'warn'}>{c.granted ? 'granted' : 'not granted'}</Badge></>}>
                    <span>{fmtDate(c.updated_at ?? c.created_at) || 'Date not recorded'} {c.details ? `· ${c.details}` : ''}</span>
                  </Item>
                ))}
              </ItemList>
            ) : <EmptyState>No consent records yet.</EmptyState>}
          </Card>
        </div>
      </Split>

      <Card title="Export & deletion (app-stored data)" subtitle={'Controls apply to everything this app stores about people: observations, support strategies, screening sessions, the timeline and the consent log. "Delete all app data" removes everything except the consent log. None of these controls touch research datasets or results artifacts.'}>
        <div className={v.dangerZone}>
          <Button variant="secondary" icon={<Download size={16} aria-hidden="true" />} loading={busy === 'export'} onClick={() => void exportAll()}>Export all app data (JSON)</Button>
          {DESTRUCTIVE.map((d) => <Button key={d.path} variant={d.path.endsWith('delete_all') ? 'danger' : 'dangerQuiet'} icon={<Trash2 size={16} aria-hidden="true" />} loading={busy === d.path} onClick={() => void destroy(d)}>{d.label}</Button>)}
        </div>
      </Card>
    </div>
  );
}
