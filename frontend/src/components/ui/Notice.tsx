import type { ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, Info, OctagonAlert } from 'lucide-react';
import { cx } from './cx';
import type { Tone } from './Badge';
import s from './Notice.module.css';

const ICONS = { neutral: Info, accent: Info, good: CheckCircle2, warn: AlertTriangle, danger: OctagonAlert } as const;

export function Notice({ tone = 'neutral', children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  const Icon = ICONS[tone];
  return (
    <div className={cx(s.notice, s[tone], className)} role={tone === 'danger' ? 'alert' : undefined}>
      <Icon size={18} className={s.icon} aria-hidden="true" />
      <div className={s.body}>{children}</div>
    </div>
  );
}
