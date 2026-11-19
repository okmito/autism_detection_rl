import type { ReactNode } from 'react';
import { Inbox, Loader2 } from 'lucide-react';
import s from './States.module.css';

export function EmptyState({ children, icon, action }: { children: ReactNode; icon?: ReactNode; action?: ReactNode }) {
  return (
    <div className={s.empty}>
      <span className={s.emptyIcon} aria-hidden="true">{icon ?? <Inbox size={22} />}</span>
      <p>{children}</p>
      {action}
    </div>
  );
}

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <span className={s.spinner} role="status">
      <Loader2 size={18} className={s.spin} aria-hidden="true" />
      <span className="visually-hidden">{label}</span>
    </span>
  );
}

/** Placeholder blocks shown while a view fetches its data. */
export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return (
    <div className={s.loading} role="status" aria-busy="true">
      <span className="visually-hidden">{label}…</span>
      <div className={s.skeleton} style={{ height: 88 }} />
      <div className={s.row}>
        <div className={s.skeleton} style={{ height: 180 }} />
        <div className={s.skeleton} style={{ height: 180 }} />
      </div>
    </div>
  );
}
