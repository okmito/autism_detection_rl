import { Info } from 'lucide-react';
import type { ScreeningResult } from '../../api/types';
import { BarList, Badge, Button, ButtonLink, EmptyState, Item, ItemList, KeyValue, Notice } from '../../components/ui';
import { fmt, joinList } from '../../lib/format';
import { buildSteps, type ItemNotes } from './ActiveSession';
import { FallbackBanner, TraceTable } from './parts';
import { BeliefTrace, Pathway, RiskScale } from './Pathway';
import s from './Screening.module.css';

const STOP_TEXT: Record<string, string> = {
  user_stop: 'you chose to stop',
  budget_exhausted: 'the question limit was reached',
  policy_stop: 'the tool had heard enough',
};
function stopText(code?: string | null): string {
  if (!code) return '—';
  return STOP_TEXT[code] ?? code.replace(/_/g, ' ');
}

/** The backend's reason is a lowercase technical clause; say it as a sentence. */
function plainReason(reason?: string | null): string {
  if (!reason) return 'No confidence range is shown for this estimate, so read it with caution.';
  const t = reason.trim();
  return `No confidence range is shown for this estimate: ${t.charAt(0).toLowerCase()}${t.slice(1)}${/[.!?]$/.test(t) ? '' : '.'}`;
}

function outcomeCopy(dec: string): { title: string; meaning: string; refer: boolean | null } {
  const refer = dec.includes('REFER') && !dec.includes('NO_');
  if (refer) {
    return {
      refer: true,
      title: 'Talking to a specialist is suggested',
      meaning: 'The estimate is at or above the referral line. This does not mean your child is autistic. It suggests that an assessment by a specialist could help. Share this result with your doctor.',
    };
  }
  if (dec.includes('NO_REFERRAL')) {
    return {
      refer: false,
      title: 'No referral suggested from these answers',
      meaning: 'The estimate is below the referral line. A screening can miss things: if you still have concerns about your child’s development, talk to your doctor.',
    };
  }
  return { refer: null, title: 'Screening finished', meaning: 'No referral recommendation was returned for this session.' };
}

export function ResultView({ result, notes = {}, tauFallback, budget, onNewSession }: { result?: ScreeningResult | null; notes?: ItemNotes; tauFallback?: number; budget?: number; onNewSession: () => void }) {
  const r = result ?? {};
  const dec = r.decision ?? '—';
  const copy = outcomeCopy(dec);
  const tau = r.explanation?.outcome?.tau ?? tauFallback ?? 0.5;
  const contribs = r.explanation?.contributions ?? [];
  const cf = r.counterfactual;
  const support = r.support_report ?? {};
  const recs = support.recommendations ?? [];
  const unc = r.explanation?.uncertainty;
  const asked = r.asked ?? (r.items_asked ?? []).length;
  const traceLen = (r.trace ?? []).length;
  const notNeeded = budget ? Math.max(0, budget - traceLen) : 0;
  const hasReport = Array.isArray(support.recommendations);
  const uncovered = (support.unassessed_areas ?? []).map((a) => String(a).split(' — ')[0]);

  return (
    <div className={s.mainCol}>
      <section className={s.outcome} aria-labelledby="result-title">
        <div className={s.outcomeHead}>
          <h2 id="result-title" className={s.outcomeTitle}>{copy.title}</h2>
          <p className={s.outcomeLabel}>Screening result (research prototype)</p>
          <div className={s.facts}>
            <span>Questions asked <strong>{asked}</strong></span>
            <span>Stopped because <strong>{stopText(r.stop_reason)}</strong></span>
          </div>
          <p className={s.notDiag}><Info size={18} aria-hidden="true" /><span><strong>Not a diagnosis.</strong> {copy.meaning}</span></p>
        </div>
        <div className={s.figure}><RiskScale value={r.p_hat} tau={tau} /></div>
      </section>

      <FallbackBanner info={r.policy_info} />

      <div className={s.twoCol}>
        <section className={s.section}>
          <h2>The path this screening took</h2>
          {(r.trace ?? []).length
            ? <Pathway label="Questions asked in this screening" steps={buildSteps(r.trace ?? [], notes, null, 0, notNeeded)} endLabel={`Stopped: ${stopText(r.stop_reason ?? 'finished')}`} />
            : <p className={s.sectionLead}>Asked: {joinList(r.items_asked) || 'no questions recorded'}.</p>}
        </section>

        <section className={s.section}>
          <h2>Why the questionnaire stopped</h2>
          {r.stop_explanation
            ? <>
                <ul>{(r.stop_explanation.narrative ?? []).map((l, i) => <li key={i}>{l}</li>)}</ul>
                {(r.stop_explanation.limitations ?? []).length > 0 && (
                  <ul className={s.small}>{r.stop_explanation.limitations?.map((l, i) => <li key={i}>{l}</li>)}</ul>
                )}
              </>
            : <p className={s.sectionLead}>The screening stopped because {stopText(r.stop_reason)}.</p>}
          <details className={s.details}>
            <summary>Numbers for clinicians and researchers</summary>
            <div>
          <KeyValue rows={[
            ['Recommendation', <Badge key="d" tone={copy.refer ? 'warn' : 'neutral'}>{dec.replace(/_/g, ' ')}</Badge>],
            ['Stop code', r.stop_reason ?? '—'],
            ['Estimate p̂', fmt(r.p_hat, 3)],
            ['Threshold τ', fmt(tau, 2)],
            ['Items asked', joinList(r.items_asked) || '—'],
            ...(r.policy_info?.agent ? [['Policy', `${r.policy_info.agent} · artifact ${String(r.policy_info.artifact_sha256 ?? '').slice(0, 12)}`] as [string, string]] : []),
          ]} />
            </div>
          </details>
        </section>
      </div>

      <section className={s.section}>
        <h2>Support ideas from these answers</h2>
        <p className={s.sectionLead}>Generated from the answers already given. No extra questions. These are ideas to try, not treatment.</p>
        {recs.length ? (
          <ItemList>
            {recs.map((rec, i) => (
              <Item key={i} title={rec.title ?? 'Suggestion'}>
                <p><strong>Idea:</strong> {rec.description ?? ''}</p>
                <p><strong>Why:</strong> {rec.why_selected ?? ''}</p>
                <p><strong>Triggered by:</strong> {(rec.triggering_question_ids ?? []).map((code, k) => `${code} (${(rec.triggering_responses ?? [])[k] === 1 ? 'atypical' : 'typical'})`).join(', ') || '-'}</p>
                <p>Source: {rec.source_attribution ?? 'rule-based, evidence-linked'} · {rec.review_status ?? 'review status as provided'}</p>
                {(rec.limitations ?? []).length > 0 && <ul className="rec-limitations">{rec.limitations?.map((l, k) => <li key={k}>{l}</li>)}</ul>}
              </Item>
            ))}
          </ItemList>
        ) : hasReport
          ? <Notice>None of the support rules were triggered by these answers, so there are no suggestions this time.</Notice>
          : <Notice>The support report is not available for this session.</Notice>}
        {uncovered.length > 0 && (
          <ul className={s.uncovered}><li><strong>Not covered by this questionnaire:</strong> {uncovered.join(', ')}.</li></ul>
        )}
        {((support.limitations ?? []).length > 0 || support.disclaimer) && (
          <details className={s.details}>
            <summary>About these suggestions</summary>
            <div>
              {(support.limitations ?? []).length > 0 && (
                <div className="report-limitations"><strong>Limitations of this support report:</strong><ul>{support.limitations?.map((l, i) => <li key={i}>{l}</li>)}</ul></div>
              )}
              {support.disclaimer && <p className="report-disclaimer">{support.disclaimer}</p>}
            </div>
          </details>
        )}
      </section>

      <section className={s.section}>
        <h2>Evidence for clinicians and researchers</h2>
        <div className={s.twoCol}>
          <div className={s.evidence}>
            {(r.trace ?? []).length > 0 && (
              <>
                <h3>How the estimate moved</h3>
                <BeliefTrace trace={r.trace ?? []} tau={tau} caption="Estimate after each answer, on the same 0 to 100 scale. Dashed points are skipped questions." />
              </>
            )}
            <h3>Which answers moved the estimate</h3>
            {contribs.length ? (
              <>
                <BarList rows={contribs.slice(0, 10).map((c) => ({ label: `${c.item} (${c.direction ?? ''})`, value: Number(c.phi ?? 0), display: fmt(c.phi, 4) }))} />
                <p className={s.small}>Exact group-Shapley contributions over observed responses (baseline = prior/unasked). Attribution ≠ causality.</p>
              </>
            ) : <EmptyState>No contribution breakdown available.</EmptyState>}
          </div>
          <div className={s.evidence}>
            <h3>What would change it</h3>
            {cf
              ? <Notice tone="accent">Flipping <strong>{cf.item ?? cf.feature ?? 'an item'}</strong> from {String(cf.original_value)} to {String(cf.flipped_value)} changes the estimate to {fmt(cf.new_p, 3)} (decision flip example where supported).</Notice>
              : <Notice>No single-answer flip counterfactual found at this state/threshold (or unavailable).</Notice>}
            <Notice><strong>Confidence range.</strong> {unc?.available ? JSON.stringify(unc) : plainReason(unc?.reason)}</Notice>
          </div>
        </div>
        <details className={s.details}>
          <summary>Answer-by-answer numbers</summary>
          <div><TraceTable trace={r.trace ?? []} /></div>
        </details>
      </section>

      <div className={s.actions}>
        <Button onClick={onNewSession}>Start a new screening</Button>
        <ButtonLink variant="secondary" href="#/explain">Open Explainable AI</ButtonLink>
      </div>
    </div>
  );
}
