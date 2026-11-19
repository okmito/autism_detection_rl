import type { ReactNode } from 'react';
import { cx } from './cx';
import s from './Badge.module.css';

export type Tone = 'neutral' | 'accent' | 'good' | 'warn' | 'danger';

export function Badge({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return <span className={cx(s.badge, s[tone])}>{children}</span>;
}
