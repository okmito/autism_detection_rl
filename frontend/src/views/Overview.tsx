import { ArrowRight, Brain, FlaskConical, LineChart, ShieldCheck } from 'lucide-react';
import { apiOr } from '../api/client';
import type { ConsentRecord, Observation, ResearchDashboard, SessionSummary, Strategy } from '../api/types';
import { Badge, ButtonLink, EmptyState, Notice } from '../components/ui';
import { useLoad } from '../hooks/useLoad';
import { fmt, fmtDate } from '../lib/format';
import { ViewState } from './ViewState';
import s from './Overview.module.css';

interface H1 { supported?: boolean; greedy_brier?: number; exact_fixed_subset_brier?: number }

async function loadOverview() {
  const [sessions, obs, strategies, research, consent] = await Promise.all([
    apiOr<SessionSummary[]>('/api/screening/sessions', []),
    apiOr<Observation[]>('/api/observations', []),
    apiOr<Strategy[]>('/api/support/strategies', []),
    apiOr<ResearchDashboard>('/api/research/dashboard', { artifacts: {} }),
    apiOr<ConsentRecord[]>('/api/consent', []),
  ]);
  return { sessions, obs, strategies, research, consent };
}

const RESEARCH_LINKS = [
  { href: '#/rl-lab', icon: FlaskConical, title: 'RL Intelligence Lab', text: 'Replay synthetic cases and compare real policy decisions step by step.' },
  { href: '#/explain', icon: Brain, title: 'Explainable AI', text: 'Attributions, counterfactuals and belief traces for completed screenings.' },
  { href: '#/research', icon: LineChart, title: 'Research & Analytics', text: 'Benchmark artifacts from results/, with their limitations and caveats.' },
  { href: '#/privacy', icon: ShieldCheck, title: 'Privacy & Consent', text: 'Consent records, export and deletion of app-stored data.' },
];

function greeting(returning: boolean): string {
  if (!returning) return 'Welcome to NeuroBridge';
  const h = new Date().getHours();
  return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
}

const STATUS_LABEL: Record<string, string> = { completed: 'Completed', active: 'In progress', expired: 'Expired' };

export function Overview() {
  const load = useLoad(loadOverview);
  return (
    <ViewState load={load}>
      {({ sessions, obs, strategies, research, consent }) => {
        const active = sessions.find((x) => x.status === 'active');
        const completed = sessions.filter((x) => x.status === 'completed').length;
        const tried = strategies.filter((x) => ['tried', 'helpful', 'not_helpful', 'mixed'].includes(x.status ?? '')).length;
        const bench = research.artifacts['step5_policy_benchmark_saudi.json'] as { h1?: H1 } | null | undefined;
        const h1 = bench?.h1 ?? null;
        const contexts = new Set(obs.map((o) => o.context)).size;
        const stats = [
          { label: 'Screenings', value: sessions.length, note: `${completed} completed${active ? ', 1 in progress' : ''}` },
          { label: 'Observations', value: obs.length, note: `Across ${contexts} ${contexts === 1 ? 'context' : 'contexts'}` },
          { label: 'Support ideas', value: strategies.length, note: `${tried} tried or reviewed` },
          { label: 'Consent records', value: consent.length, note: 'Kept on this computer' },
        ];
        return (
          <>
            <section className={s.welcome} aria-labelledby="ov-welcome">
              <div className={s.welcomeText}>
                <p id="ov-welcome" className={s.hello}>{greeting(sessions.length > 0)}</p>
                <p className={s.lede}>
                  {active
                    ? 'You have a screening in progress. Pick up where you stopped; nothing has been lost.'
                    : 'Answer a few short questions about your child’s everyday behaviour. The tool chooses each next question, explains why, and stops when it has enough.'}
                </p>
                <div className={s.actions}>
                  <ButtonLink variant="primary" href="#/screening">{active ? 'Resume screening' : 'Start screening'}</ButtonLink>
                  <ButtonLink variant="secondary" href="#/context">Record an observation</ButtonLink>
                </div>
              </div>
              <dl className={s.stats} aria-label="At a glance">
                {stats.map((x) => (
                  <div key={x.label} className={s.stat}>
                    <dt>{x.label}</dt>
                    <dd>{x.value}</dd>
                    <span>{x.note}</span>
                  </div>
                ))}
              </dl>
            </section>

            <div className={s.two}>
              <section aria-labelledby="ov-sessions">
                <div className={s.head}><h2 id="ov-sessions">Recent screenings</h2><a href="#/screening">View all</a></div>
                {sessions.length ? (
                  <ol className={s.feed}>
                    {sessions.slice(0, 5).map((x) => (
                      <li key={x.id} className={`${s.feedItem} ${x.status === 'completed' ? s.done : s.live}`} title={`Session ${x.id}`}>
                        <span className={s.dot} aria-hidden="true" />
                        <div className={s.feedBody}>
                          <p className={s.feedTitle}>Screening <Badge tone={x.status === 'completed' ? 'good' : 'warn'}>{STATUS_LABEL[x.status] ?? x.status}</Badge></p>
                          <p className={s.feedMeta}>{fmtDate(x.updated_at ?? x.created_at) || 'Date not recorded'} · {x.asked_count ?? 0} {x.asked_count === 1 ? 'question' : 'questions'} · {x.policy}</p>
                        </div>
                      </li>
                    ))}
                  </ol>
                ) : <EmptyState>No screenings yet. Your first one will appear here.</EmptyState>}
              </section>

              <section aria-labelledby="ov-obs">
                <div className={s.head}><h2 id="ov-obs">Recent observations</h2></div>
                {obs.length ? (
                  <ul className={s.notes}>
                    {obs.slice(0, 4).map((o) => (
                      <li key={o.id}>
                        <p className={s.noteTitle}>{o.context}</p>
                        <p className={s.noteText}>{(o.what_happened ?? '').slice(0, 140)}</p>
                      </li>
                    ))}
                  </ul>
                ) : <EmptyState>No observations yet. Context Explorer is where you add one.</EmptyState>}
                <p className={s.more}>{obs.length < 3 ? 'Short notes on what happens, where and what helped build a useful picture. ' : ''}<a href="#/context">Open Context Explorer</a></p>
              </section>
            </div>

            <section aria-labelledby="ov-research">
              <div className={s.head}><h2 id="ov-research">Research evidence</h2></div>
              {h1 && typeof h1.supported === 'boolean' ? (
                <Notice tone={h1.supported ? 'good' : 'warn'}>
                  H1 (adaptive beats exact best fixed subset) at matched budget: <strong>{h1.supported ? 'supported in this artifact' : 'not supported in this artifact'}</strong>. Greedy Brier: {fmt(h1.greedy_brier, 4)}, Fixed Brier: {fmt(h1.exact_fixed_subset_brier, 4)} (lower is better). Interpret with dataset/label limitations.
                </Notice>
              ) : <Notice>Step-5 benchmark summary not available in current results artifacts; see Research Dashboard for loaded files.</Notice>}
              <ul className={s.index}>
                {RESEARCH_LINKS.map(({ href, icon: Icon, title, text }) => (
                  <li key={href}>
                    <a href={href}>
                      <span className={s.indexIcon}><Icon size={20} strokeWidth={1.75} aria-hidden="true" /></span>
                      <span className={s.indexText}><strong>{title}</strong><span>{text}</span></span>
                      <ArrowRight size={16} className={s.go} aria-hidden="true" />
                    </a>
                  </li>
                ))}
              </ul>
              <p className={s.note}>Only metrics backed by existing files under <code>results/</code> are shown. No values are fabricated.</p>
            </section>
          </>
        );
      }}
    </ViewState>
  );
}
