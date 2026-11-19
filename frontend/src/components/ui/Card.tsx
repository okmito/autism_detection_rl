import type { HTMLAttributes, ReactNode } from 'react';
import { cx } from './cx';
import s from './Card.module.css';

interface CardProps extends Omit<HTMLAttributes<HTMLElement>, 'title'> {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  /** `default` is an open section. `raised` is a white surface for something held (a form); `inset` is a quiet tinted block. */
  tone?: 'default' | 'raised' | 'inset';
  as?: 'section' | 'div' | 'article';
}

export function Card({ title, subtitle, actions, tone = 'default', as: Tag = 'section', className, children, ...rest }: CardProps) {
  return (
    <Tag className={cx(s.card, tone === 'raised' && s.raised, tone === 'inset' && s.inset, className)} {...rest}>
      {(title || actions) && (
        <header className={s.head}>
          <div>
            {title && <h2 className={s.title}>{title}</h2>}
            {subtitle && <p className={s.subtitle}>{subtitle}</p>}
          </div>
          {actions && <div className={s.actions}>{actions}</div>}
        </header>
      )}
      {children}
    </Tag>
  );
}

export function MetricCard({ label, value, note, small }: { label: string; value: ReactNode; note?: ReactNode; small?: boolean }) {
  return (
    <div className={cx(s.card, s.metric)}>
      <div className={s.metricLabel}>{label}</div>
      <div className={cx(s.metricValue, small && s.metricSmall)}>{value}</div>
      {note && <div className={s.metricNote}>{note}</div>}
    </div>
  );
}
