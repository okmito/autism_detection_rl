import s from './ProgressBar.module.css';

export function ProgressBar({ value, max = 100, label }: { value: number; max?: number; label: string }) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className={s.track} role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={max} aria-valuenow={Math.min(value, max)}>
      <div className={s.fill} style={{ width: `${pct}%` }} />
    </div>
  );
}
