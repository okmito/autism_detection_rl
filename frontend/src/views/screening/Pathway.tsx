import type { TraceStep } from '../../api/types';
import { fmt } from '../../lib/format';
import s from './Screening.module.css';

export type StepState = 'answered' | 'skipped' | 'current' | 'remaining' | 'notneeded';

export interface PathStep {
  key: string;
  state: StepState;
  code?: string;
  text?: string;
  /** What the person chose, in the instrument's own wording, when known. */
  answer?: string;
  /** Screening estimate after this step, 0..1. */
  estimate?: number;
  /** For a collapsed remaining / not-needed row: how many questions it stands for. */
  count?: number;
}

/** The numbered pathway: every question asked, skipped, current, or still possible stays visible. */
export function Pathway({ steps, endLabel, label }: { steps: PathStep[]; endLabel?: string; label: string }) {
  let n = 0;
  return (
    <ol className={s.path} aria-label={label}>
      {steps.map((st) => {
        const numbered = st.state === 'answered' || st.state === 'current';
        if (numbered) n += 1;
        return (
          <li key={st.key} className={`${s.step} ${s[st.state]}`} aria-current={st.state === 'current' ? 'step' : undefined}>
            <span className={s.disc} aria-hidden="true">{numbered ? n : st.state === 'skipped' ? '–' : ''}</span>
            <span className={s.stepText}>
              {st.state === 'remaining'
                ? <span className={s.stepTitle}>Up to {st.count ?? 1} more {st.count === 1 ? 'question' : 'questions'}, only if needed</span>
                : st.state === 'notneeded'
                ? <span className={s.stepTitle}>{st.count ?? 1} further {st.count === 1 ? 'question' : 'questions'} not asked</span>
                : <>
                    {st.code && <span className={s.stepCode}>{st.code}</span>}
                    <span className={s.stepTitle}>{st.text ?? `Question ${st.code ?? ''}`}</span>
                  </>}
            </span>
            <span className={s.stepAnswer}>
              {st.state === 'current' ? 'Now'
                : st.state === 'skipped' ? 'Not answered'
                : st.state === 'remaining' ? 'If needed'
                : st.state === 'notneeded' ? 'Not needed'
                : st.answer ?? 'Answered'}
              {st.estimate !== undefined && st.state === 'answered' && (
                <span className={s.stepEst}>estimate {fmt(st.estimate * 100, 0)}</span>
              )}
            </span>
          </li>
        );
      })}
      {endLabel && (
        <li className={`${s.step} ${s.end}`}>
          <span className={s.disc} aria-hidden="true" />
          <span className={s.stepText}><span className={s.stepTitle}>{endLabel}</span></span>
          <span className={s.stepAnswer} />
        </li>
      )}
    </ol>
  );
}

/**
 * The estimate after each answer, on the same fixed 0-100 scale as the risk scale, with the referral line drawn.
 * The numbers behind it sit in the answer-by-answer table next to every use, so the picture is never the only record.
 */
export function BeliefTrace({ trace, tau, caption }: { trace: TraceStep[]; tau: number; caption?: string }) {
  if (!trace.length) return null;
  const W = 380, H = 190, L = 36, R = 16, T = 14, B = 34;
  const clamp = (v: number) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
  const first = trace[0];
  const pts = [
    { label: 'Start', v: clamp(first ? first.belief_before : 0), skipped: false },
    ...trace.map((t) => ({ label: t.item, v: clamp(t.belief_after), skipped: Boolean(t.skipped) })),
  ];
  const x = (i: number) => L + (pts.length === 1 ? 0 : (i * (W - L - R)) / (pts.length - 1));
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const t = clamp(tau);
  const line = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' ');
  const aria = `Screening estimate after each answer, out of 100, with the referral line at ${fmt(t * 100, 0)}: ${pts.map((p) => `${p.label} ${fmt(p.v * 100, 0)}`).join(', ')}.`;
  const labelEvery = Math.ceil(pts.length / 8);
  return (
    <figure className={s.trace}>
      <svg className={s.traceSvg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={aria}>
        <rect className={s.traceBand} x={L} y={T} width={W - L - R} height={y(t) - T} rx="6" />
        <rect className={s.traceBandLow} x={L} y={y(t)} width={W - L - R} height={H - B - y(t)} rx="6" />
        <line className={s.traceThr} x1={L} x2={W - R} y1={y(t)} y2={y(t)} />
        <text className={`${s.traceTxt} ${s.traceTxtStrong}`} x={W - R} y={y(t) - 5} textAnchor="end">Referral line {fmt(t * 100, 0)}</text>
        <text className={s.traceTxt} x={L - 8} y={T + 4} textAnchor="end">100</text>
        <text className={s.traceTxt} x={L - 8} y={H - B + 4} textAnchor="end">0</text>
        <path className={s.traceLine} d={line} />
        {pts.map((p, i) => (
          <g key={`${p.label}-${i}`}>
            <circle className={p.skipped ? s.tracePtSkip : s.tracePt} cx={x(i)} cy={y(p.v)} r="4.5" />
            {(i % labelEvery === 0 || i === pts.length - 1) && <text className={s.traceTxt} x={x(i)} y={H - B + 18} textAnchor="middle">{p.label}</text>}
          </g>
        ))}
      </svg>
      {caption && <figcaption className={s.traceCap}>{caption}</figcaption>}
    </figure>
  );
}

/** One fixed 0-100 scale with the referral line, the same on every screen. */
export function RiskScale({ value, tau, caption }: { value?: number | null; tau: number; caption?: string }) {
  const has = typeof value === 'number' && Number.isFinite(value);
  const v = has ? Math.max(0, Math.min(1, value as number)) : 0;
  const t = Math.max(0, Math.min(1, tau));
  const above = has && v >= t;
  const text = has
    ? `Screening estimate ${fmt(v * 100, 0)} out of 100. Referral line at ${fmt(t * 100, 0)}. ${above ? 'At or above the line.' : 'Below the line.'}`
    : 'No estimate yet.';
  return (
    <figure className={s.scale}>
      <div className={s.scaleTop}>
        <span className={s.scaleValue}>{has ? fmt(v * 100, 0) : '—'}<span className={s.scaleOf}> / 100</span></span>
        <span className={s.scaleSide}>{has ? (above ? 'At or above the referral line' : 'Below the referral line') : 'No estimate yet'}</span>
      </div>
      <div className={s.scaleBar} role="img" aria-label={text} style={{ ['--t' as string]: `${t * 100}%` }}>
        <span className={s.scaleThr} style={{ left: `${t * 100}%` }} />
        {has && <span className={s.scalePt} style={{ left: `${v * 100}%` }} />}
      </div>
      <div className={s.scaleLabels} aria-hidden="true">
        <span>0</span>
        <span className={s.scaleThrLabel} style={{ left: `${t * 100}%` }}>Referral line {fmt(t * 100, 0)}</span>
        <span>100</span>
      </div>
      {caption && <figcaption className={s.scaleCaption}>{caption}</figcaption>}
    </figure>
  );
}
