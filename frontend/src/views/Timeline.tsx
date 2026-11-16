import { useMemo, useState } from 'react';
import { Download } from 'lucide-react';
import { api, apiOr, errorMessage } from '../api/client';
import type { TimelineEvent } from '../api/types';
import { Badge, Button, EmptyState, FormStack, SearchField, Select, useToast } from '../components/ui';
import { useLoad } from '../hooks/useLoad';
import { downloadJson, fmtDay, fmtTime } from '../lib/format';
import { ViewState } from './ViewState';
import s from './Views.module.css';

const KINDS = ['observation', 'observation_update', 'observation_delete', 'support', 'support_update', 'support_attempt', 'support_delete', 'screening', 'privacy'];

function kindClass(kind: string): string {
  if (kind.startsWith('screening')) return s.kindScreening ?? '';
  if (kind.startsWith('support')) return s.kindSupport ?? '';
  if (kind.startsWith('privacy')) return s.kindPrivacy ?? '';
  return '';
}

export function Timeline() {
  const load = useLoad(() => apiOr<TimelineEvent[]>('/api/timeline', []));
  return <ViewState load={load}>{(events) => <TimelineBody events={events} />}</ViewState>;
}

function TimelineBody({ events }: { events: TimelineEvent[] }) {
  const toast = useToast();
  const [query, setQuery] = useState('');
  const [kind, setKind] = useState('');
  const [exporting, setExporting] = useState(false);

  const items = useMemo(() => {
    const q = query.toLowerCase();
    return events.filter((e) => (!kind || e.kind === kind) && (!q || `${e.title} ${e.summary ?? ''} ${e.notes ?? ''} ${e.context ?? ''}`.toLowerCase().includes(q)));
  }, [events, query, kind]);

  const days = useMemo(() => {
    const groups: { day: string; events: TimelineEvent[] }[] = [];
    for (const e of items) {
      const day = fmtDay(e.event_time ?? e.created_at);
      const last = groups[groups.length - 1];
      if (last && last.day === day) last.events.push(e);
      else groups.push({ day, events: [e] });
    }
    return groups;
  }, [items]);

  const exportJson = async () => {
    setExporting(true);
    try { downloadJson(await api('/api/timeline/export', { method: 'POST' }), 'neurobridge_export.json'); toast.show('Export downloaded', 'good'); }
    catch (e) { toast.error(errorMessage(e)); } finally { setExporting(false); }
  };

  return (
    <FormStack>
      <div className={s.timelineBar}>
        <div className={s.toolbar}>
          <SearchField label="Search timeline" value={query} onChange={setQuery} />
          <Select aria-label="Filter by kind" value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="">All kinds</option>{KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
          </Select>
        </div>
        <Button variant="secondary" icon={<Download size={16} aria-hidden="true" />} loading={exporting} onClick={() => void exportJson()}>Export (JSON)</Button>
      </div>
      <p className={s.muted}>Chronological records of observations, support strategies, and app events. Handle exports as sensitive.</p>
      {days.length ? (
        <div className={s.days}>
          {days.map((g) => (
            <section key={g.day} className={s.day}>
              <h2>{g.day}</h2>
              <ol className={s.events}>
                {g.events.map((e, i) => {
                  const when = fmtTime(e.event_time ?? e.created_at);
                  return (
                    <li key={e.id ?? i} className={`${s.event} ${kindClass(e.kind)}`}>
                      <span className={s.time}>{when}</span>
                      <span className={s.eventDot} aria-hidden="true" />
                      <div>
                        <p className={s.eventTitle}>{e.title} <Badge>{e.kind}</Badge> <span className={s.timeInline}>{when}</span></p>
                        {(e.context || e.summary) && <p className={s.eventText}>{[e.context, e.summary].filter(Boolean).join(' · ')}</p>}
                        {e.notes && <p className={s.eventText}>{e.notes}</p>}
                      </div>
                    </li>
                  );
                })}
              </ol>
            </section>
          ))}
        </div>
      ) : <EmptyState>No timeline events yet. Record observations or screening activity to populate.</EmptyState>}
    </FormStack>
  );
}
