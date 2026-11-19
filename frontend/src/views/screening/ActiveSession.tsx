import type { AnswerOption, PolicyInfo, Question, RlDecision, SelectionDiag, TraceStep } from '../../api/types';
import { Badge, Button, EmptyState, Notice } from '../../components/ui';
import { DecisionNotes, FallbackBanner, selectionText, TraceTable } from './parts';
import { Pathway, RiskScale, type PathStep } from './Pathway';
import s from './Screening.module.css';

export interface LiveSession {
  session_id: string;
  policy: string;
  budget: number;
  tau: number;
  question: Question | null;
  belief?: number;
  asked: number;
  selection?: SelectionDiag | null;
  decision?: RlDecision | null;
  policy_info?: PolicyInfo | null;
  trace: TraceStep[];
}

/** What the person saw and chose for each item code, kept client-side so the pathway can speak in their words. */
export type ItemNotes = Record<string, { text?: string; answer?: string }>;

// The person answers in the instrument's own wording; the backend sends each option's scored value.
// Never "Yes/No": for A1-A9 a literal "Yes" is the typical answer, for A10 the atypical one.
function usableOptions(q: Question): AnswerOption[] {
  return (q.options ?? []).filter((o) => o.value === 0 || o.value === 1);
}

/** A plain-language reason for the person answering; the technical diagnostic sits behind a disclosure. */
function plainWhy(policy: string, sel?: SelectionDiag | null): string {
  if (policy === 'rl_adaptive') return 'A trained agent chose this as the most useful next question, based on your answers so far. It also decides when it has heard enough.';
  if (sel?.applicable === false || policy === 'random') return 'In this research mode questions are picked at random, so smarter ordering can be compared against it.';
  if (sel?.criterion_vacuous) return 'Your answers so far do not separate the remaining questions, so the tool takes the next one in a fixed order.';
  return 'Of the questions left, this one is expected to tell the most about the outcome, given your answers so far.';
}

export function buildSteps(trace: TraceStep[], notes: ItemNotes, current?: Question | null, remaining = 0, notNeeded = 0): PathStep[] {
  const steps: PathStep[] = trace.map((t, i) => ({
    key: `t${i}-${t.item}`,
    state: t.skipped ? 'skipped' : 'answered',
    code: t.item,
    text: notes[t.item]?.text,
    answer: notes[t.item]?.answer,
    estimate: t.belief_after,
  }));
  if (current) steps.push({ key: `q-${current.code}`, state: 'current', code: current.code, text: current.text });
  if (remaining > 0) steps.push({ key: 'remaining', state: 'remaining', count: remaining });
  if (notNeeded > 0) steps.push({ key: 'notneeded', state: 'notneeded', count: notNeeded });
  return steps;
}

interface Props {
  sess: LiveSession;
  notes: ItemNotes;
  busy: boolean;
  onAnswer: (value: number, q: Question, label: string) => void;
  onSkip: (q: Question) => void;
  onStop: () => void;
}

export function ActiveSession({ sess, notes, busy, onAnswer, onSkip, onStop }: Props) {
  const q = sess.question;
  const isRL = sess.policy === 'rl_adaptive';
  const options = q ? usableOptions(q) : [];
  const position = sess.trace.length + (q ? 1 : 0);
  const left = Math.max(0, sess.budget - position);

  return (
    <div className={s.layout} aria-busy={busy}>
      <div className={s.mainCol}>
        <FallbackBanner info={sess.policy_info} />
        <div className={s.qHead}>
          <ol className={s.steps} aria-hidden="true">
            {Array.from({ length: Math.max(sess.budget, position) }, (_, i) => (
              <li key={i} className={`${s.seg} ${i < sess.trace.length ? s.segDone : i === sess.trace.length && q ? s.segNow : ''}`} />
            ))}
          </ol>
          <div className={s.qLine}>
            <span className={s.qCount}>Question {position} · at most {sess.budget}</span>
            <span className={s.qLeft}>{isRL ? 'The agent may stop sooner. ' : ''}At most<strong>{left} left</strong></span>
          </div>
        </div>

        {q ? (
          <>
            <h2 key={q.code} className={s.question}>
              <span className={s.code}>{q.code}</span>
              <span>{q.text}</span>
            </h2>
            {options.length === 0 ? (
              <Notice tone="warn">Answer options for this question are unavailable. Reload the page to continue.</Notice>
            ) : (
              <div role="group" aria-label="Answer options" className={s.options}>
                {options.map((o) => (
                  <button key={`${o.letter}-${o.value}`} type="button" className={s.option} disabled={busy} data-letter={o.letter} aria-label={o.label} onClick={() => onAnswer(o.value, q, o.label)}>
                    <span className={s.letter} aria-hidden="true">{o.letter}</span>
                    <span aria-hidden="true">{o.label}</span>
                  </button>
                ))}
              </div>
            )}
            <div className={s.actions}>
              <Button variant="secondary" disabled={busy} onClick={onStop}>Stop and see result</Button>
              {isRL && <Button variant="ghost" disabled={busy} onClick={() => onSkip(q)}>Prefer not to answer</Button>}
            </div>
          </>
        ) : <EmptyState>No pending question.</EmptyState>}

        <details className={s.details}>
          <summary>Details for clinicians and researchers</summary>
          <div>
            <div className={s.meta}>
              <span>Method</span><Badge tone="accent">{sess.policy}</Badge>
              <span>Session</span><Badge>{sess.session_id}</Badge>
              <span>Referral threshold τ = {(sess.tau || 0.5).toFixed(2)}</span>
            </div>
            <TraceTable trace={sess.trace} />
          </div>
        </details>
      </div>

      <aside className={s.sideCol} aria-label="About this question">
        <div className={s.rail}>
          <section className={s.why}>
            <h2>Why this question?</h2>
            <p>{plainWhy(sess.policy, sess.selection)}</p>
            <details className={s.details}>
              <summary>How it was chosen</summary>
              <div>{sess.decision ? <DecisionNotes d={sess.decision} /> : <p>{selectionText(sess.selection)}</p>}</div>
            </details>
          </section>

          <section className={s.railPart}>
            <h2>Current estimate</h2>
            <RiskScale value={sess.belief} tau={sess.tau || 0.5} caption="Updates after each answer. Not a diagnosis." />
          </section>

          <section className={s.railPart}>
            <h2>Your path so far</h2>
            <Pathway label="Questions so far" steps={buildSteps(sess.trace, notes, q, left)} />
          </section>
        </div>
      </aside>
    </div>
  );
}
