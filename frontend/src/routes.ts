import { Activity, BookOpenCheck, Brain, ClipboardList, FlaskConical, LayoutDashboard, LineChart, ShieldCheck, Sparkles, type LucideIcon } from 'lucide-react';

export type RouteKey = 'overview' | 'screening' | 'context' | 'support' | 'timeline' | 'rl-lab' | 'explain' | 'research' | 'privacy';

export interface RouteDef {
  key: RouteKey;
  nav: string;
  title: string;
  sub: string;
  icon: LucideIcon;
  /** Rail section: day-to-day use by families and clinicians, or the research evidence. */
  group: 'care' | 'research' | 'data';
}

/** Same hash URLs and page copy as the legacy UI, so bookmarks and docs keep working. */
export const ROUTES: RouteDef[] = [
  { key: 'overview', nav: 'Overview', title: 'Overview Dashboard', sub: 'Start or resume screening, review recent activity, and access research & support modules.', icon: LayoutDashboard, group: 'care' },
  { key: 'screening', nav: 'Guided Screening', title: 'Screening questions', sub: 'Short questions about your child’s everyday behaviour. Each one is explained.', icon: ClipboardList, group: 'care' },
  { key: 'context', nav: 'Context Explorer', title: 'Context Explorer', sub: 'Record contextual observations across environments. Kept separate from validated questionnaire responses.', icon: BookOpenCheck, group: 'care' },
  { key: 'support', nav: 'Support Studio', title: 'Support Studio', sub: 'Observation-grounded, non-diagnostic support ideas with tracking and notes.', icon: Sparkles, group: 'care' },
  { key: 'timeline', nav: 'Timeline', title: 'Observation Timeline', sub: 'Chronological view of observations, strategies, and session events.', icon: Activity, group: 'care' },
  { key: 'rl-lab', nav: 'RL Intelligence Lab', title: 'RL Intelligence Lab', sub: 'Interactive, reproducible synthetic replay using real policy/inference logic where available.', icon: FlaskConical, group: 'research' },
  { key: 'explain', nav: 'Explainable AI', title: 'Explainable AI', sub: 'Model explanations (attribution, counterfactuals, belief/decision traces). Attribution is not causality.', icon: Brain, group: 'research' },
  { key: 'research', nav: 'Research & Analytics', title: 'Research & Analytics', sub: 'Actual available offline research artifacts and metrics. No fabricated results.', icon: LineChart, group: 'research' },
  { key: 'privacy', nav: 'Privacy & Consent', title: 'Privacy & Consent', sub: 'Data handling, consent records, export, and deletion controls for app-stored information.', icon: ShieldCheck, group: 'data' },
];

const KEYS = new Set<string>(ROUTES.map((r) => r.key));

export function parseHash(hash: string): RouteKey {
  const name = (hash.replace(/^#\/?/, '') || 'overview').split('?')[0] ?? 'overview';
  return KEYS.has(name) ? (name as RouteKey) : 'overview';
}
