import { useCallback, useEffect, useRef, useState } from 'react';
import { api, apiOr, ApiError, errorMessage } from '../../api/client';
import type { AnswerResponse, Meta, Question, ScreeningResult, SessionDetail, SessionSummary, StartResponse } from '../../api/types';
import { Badge, Button, EmptyState, Field, Item, ItemList, Notice, Select } from '../../components/ui';
import { useToast } from '../../components/ui';
import { useStatus } from '../../hooks/status';
import { ActiveSession, type ItemNotes, type LiveSession } from './ActiveSession';
import { ResultView } from './ResultView';
import { rlAvailabilityText } from './parts';
import { fmtDate } from '../../lib/format';
import st from './Screening.module.css';

const DEFAULT_POLICIES = ['greedy', 'random', 'beta_greedy'];

type Pane =
  | { kind: 'idle' }
  | { kind: 'live'; sess: LiveSession }
  | { kind: 'result'; result: ScreeningResult | null | undefined };

function fromDetail(s: SessionDetail, meta: Meta | null): LiveSession {
  return {
    session_id: s.id, policy: s.policy, budget: s.budget ?? meta?.budget ?? 6, tau: s.tau ?? meta?.tau ?? 0.5,
    question: s.pending_question ?? null, belief: s.belief, asked: s.asked_count ?? 0, selection: null,
    decision: s.decision ?? null, policy_info: s.policy_info ?? null, trace: s.trace ?? [],
  };
}

export function Screening() {
  const toast = useToast();
  const { withStatus } = useStatus();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [pane, setPane] = useState<Pane>({ kind: 'idle' });
  const [policy, setPolicy] = useState('greedy');
  const [mode, setMode] = useState('interactive');
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const answering = useRef(false); // a double-click must not submit the same answer twice
  const [notes, setNotes] = useState<ItemNotes>({});
  const remember = (q: Question, answer?: string) => setNotes((n) => ({ ...n, [q.code]: { text: q.text, answer } }));

  const resume = useCallback(async (id: string, silent = false) => {
    try {
      const s = await api<SessionDetail>(`/api/screening/sessions/${id}`);
      if (s.status === 'completed') { setPane({ kind: 'result', result: s.result }); return; }
      if (s.status === 'expired') { toast.show('That session expired after a long pause. Start a new one.'); return; }
      setPane({ kind: 'live', sess: fromDetail(s, meta) });
    } catch (e) { if (!silent) toast.error(errorMessage(e)); }
  }, [meta, toast]);

  const refreshSessions = useCallback(async () => {
    const list = await apiOr<SessionSummary[]>('/api/screening/sessions', []);
    setSessions(list);
    return list;
  }, []);

  useEffect(() => {
    let live = true;
    (async () => {
      let m: Meta | null = null;
      try { m = await api<Meta>('/api/meta'); } catch (e) { if (live) toast.error(errorMessage(e)); }
      const list = await apiOr<SessionSummary[]>('/api/screening/sessions', []);
      if (!live) return;
      setMeta(m); setSessions(list); setReady(true);
      const active = list.find((s) => s.status === 'active');
      if (active) {
        try {
          const s = await api<SessionDetail>(`/api/screening/sessions/${active.id}`);
          if (live && s.status === 'active') setPane({ kind: 'live', sess: fromDetail(s, m) });
        } catch { /* a silent resume that fails just leaves the start panel */ }
      }
    })();
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const start = async () => {
    try {
      const r = await withStatus('Starting session (model may load)…', () =>
        api<StartResponse>('/api/screening/start', { method: 'POST', body: { mode, policy } }));
      if (r.result || r.mode === 'auto') setPane({ kind: 'result', result: r.result });
      else setPane({ kind: 'live', sess: {
        session_id: r.session_id, policy: r.policy, budget: r.budget || meta?.budget || 6, tau: r.tau, question: r.question ?? null,
        belief: r.belief, asked: r.asked ?? 0, selection: r.selection, decision: r.decision ?? null, policy_info: r.policy_info ?? null, trace: [],
      } });
      void refreshSessions();
    } catch (e) { toast.error(errorMessage(e)); }
  };

  const submit = async (sess: LiveSession, payload: { value: number | null; stop?: boolean; item_index?: number | null; skip?: boolean }) => {
    if (answering.current) return;
    answering.current = true; setBusy(true);
    try {
      const r = await withStatus('Updating…', () => api<AnswerResponse>('/api/screening/answer', {
        method: 'POST',
        body: { session_id: sess.session_id, value: payload.value, stop: payload.stop ?? false, item_index: payload.item_index ?? null, ...(payload.skip ? { skip: true } : {}) },
      }));
      if (r.finished) { setPane({ kind: 'result', result: r.result }); toast.show('Session completed', 'good'); void refreshSessions(); }
      else setPane({ kind: 'live', sess: { ...sess, question: r.next ?? null, belief: r.belief_after, asked: r.asked ?? sess.asked, selection: r.selection, decision: r.decision ?? null, trace: [...sess.trace, r.trace_step] } });
    } catch (e) {
      toast.error(errorMessage(e));
      if (e instanceof ApiError && e.status === 409) await resume(sess.session_id, true); // stale/duplicate answer: show the current question
    } finally { answering.current = false; setBusy(false); }
  };

  const viewSession = async (id: string) => {
    try {
      const s = await api<SessionDetail>(`/api/screening/sessions/${id}`);
      if (s.result) setPane({ kind: 'result', result: s.result }); else toast.show('No completed result');
    } catch (e) { toast.error(errorMessage(e)); }
  };

  const active = sessions.find((s) => s.status === 'active');
  const policies = Array.isArray(meta?.policies_available) ? meta.policies_available : DEFAULT_POLICIES;

  if (!ready) return <EmptyState>Loading…</EmptyState>;

  if (pane.kind === 'live') {
    return (
      <ActiveSession
        sess={pane.sess} notes={notes} busy={busy}
        onAnswer={(value, q: Question, label: string) => { remember(q, label); void submit(pane.sess, { value, item_index: q.index }); }}
        onSkip={(q: Question) => { remember(q); void submit(pane.sess, { value: null, item_index: q.index, skip: true }); }}
        onStop={() => void submit(pane.sess, { value: null, stop: true })}
      />
    );
  }

  if (pane.kind === 'result') {
    return <ResultView result={pane.result} notes={notes} tauFallback={meta?.tau} budget={meta?.budget} onNewSession={() => { setPane({ kind: 'idle' }); setNotes({}); void refreshSessions(); }} />;
  }

  return (
    <div className={st.layout}>
      <section className={st.start} aria-labelledby="start-title">
        <h2 id="start-title" className={st.startTitle}>Start a screening</h2>
        <p className={st.startLead}>Up to {meta?.budget ?? 6} short questions about your child’s everyday behaviour. It takes a few minutes. You can stop at any time.</p>
        <ol className={st.howList}>
          <li><div><strong>Answer each question</strong><span>Choose the answer that fits your child most of the time.</span></div></li>
          <li><div><strong>The tool picks what to ask next</strong><span>It chooses each question from your earlier answers and stops when it has enough.</span></div></li>
          <li><div><strong>Read the result</strong><span>You get a screening estimate, whether talking to a specialist is suggested, and support ideas. It is not a diagnosis.</span></div></li>
        </ol>
        <div className={st.actions}>
          <Button onClick={() => void start()} disabled={busy}>Start new session</Button>
          {active && <Button variant="secondary" onClick={() => void resume(active.id)}>Resume active ({active.id})</Button>}
        </div>
        <div className={st.settings}>
          <h2>Research settings</h2>
          <Field label="Policy (real implementations)" help={<>greedy, random, beta_greedy ask a fixed number of questions. rl_adaptive is the trained RL agent: it picks each question from the answers so far and decides when to stop. {rlAvailabilityText(meta?.rl_policy)}</>}>
            <Select value={policy} onChange={(e) => setPolicy(e.target.value)}>{policies.map((p) => <option key={p} value={p}>{p}</option>)}</Select>
          </Field>
          <Field label="Mode">
            <Select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="interactive">Interactive (answer each question)</option>
              <option value="auto">Auto episode (replay on held-out test record)</option>
            </Select>
          </Field>
        </div>
      </section>

      <aside className={st.sideCol} aria-label="Sessions">
        <section className={st.box}>
          <div className={st.boxHead}>
            <h2>Recent sessions</h2>
            <Button size="sm" variant="ghost" onClick={() => void refreshSessions()}>Refresh</Button>
          </div>
          {sessions.length ? (
            <ItemList>
              {sessions.slice(0, 6).map((s) => (
                <Item key={s.id} title={<>Screening, {fmtDate(s.updated_at ?? s.created_at) || s.id} <Badge tone={s.status === 'completed' ? 'good' : 'warn'}>{s.status}</Badge></>}
                  actions={s.status === 'active'
                    ? <Button size="sm" variant="ghost" onClick={() => void resume(s.id)}>Resume</Button>
                    : <Button size="sm" variant="ghost" onClick={() => void viewSession(s.id)}>View</Button>}>
                  <span title={`Session ${s.id}`}>{s.asked_count ?? 0} questions asked · method {s.policy}</span>
                </Item>
              ))}
            </ItemList>
          ) : <EmptyState>No sessions yet.</EmptyState>}
        </section>
        <Notice tone="warn">This is a screening/referral-research estimate, not a diagnosis. Results are based on a research model and dataset with known limitations (including label circularity).</Notice>
      </aside>
    </div>
  );
}
