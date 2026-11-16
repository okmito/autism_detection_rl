import type { ReactNode } from 'react';
import { cx } from './cx';
import s from './Data.module.css';

/** Table that scrolls inside its own box so a wide table never widens the page. */
export function DataTable({ children, maxHeight }: { children: ReactNode; maxHeight?: number }) {
  return (
    <div className={s.tableWrap} style={maxHeight ? { maxHeight } : undefined} tabIndex={0} role="region" aria-label="Data table (scrollable)">
      <table className={s.table}>{children}</table>
    </div>
  );
}

export function ItemList({ children, dense }: { children: ReactNode; dense?: boolean }) {
  return <div className={cx(s.list, dense && s.listDense)}>{children}</div>;
}

export function Item({ title, children, actions }: { title?: ReactNode; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className={s.item}>
      {title && <h4 className={s.itemTitle}>{title}</h4>}
      {children && <div className={s.itemBody}>{children}</div>}
      {actions && <div className={s.itemActions}>{actions}</div>}
    </div>
  );
}

export function KeyValue({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className={s.kv}>
      {rows.map(([k, v]) => (<div key={k} className={s.kvRow}><dt>{k}</dt><dd>{v}</dd></div>))}
    </dl>
  );
}

/**
 * Diverging bars on one shared scale. Positive values grow right of a zero axis, negative left of it.
 * When every value has the same sign the axis moves to the edge so the bars use the whole track.
 */
export function BarList({ rows, positiveLabel = 'Raises the estimate', negativeLabel = 'Lowers the estimate' }: { rows: { label: ReactNode; value: number; display: string }[]; positiveLabel?: string; negativeLabel?: string }) {
  const maxPos = Math.max(0, ...rows.map((r) => r.value));
  const maxNeg = Math.max(0, ...rows.map((r) => -r.value));
  const total = Math.max(0.0001, maxPos + maxNeg);
  const zero = (maxNeg / total) * 100;
  const hasNeg = maxNeg > 0;
  return (
    <div className={s.bars}>
      {rows.map((r, i) => {
        const w = (Math.abs(r.value) / total) * 100;
        const style = r.value < 0 ? { width: `${w}%`, right: `${100 - zero}%` } : { width: `${w}%`, left: `${zero}%` };
        return (
          <div key={i} className={s.barRow}>
            <div className={s.barLabel}>{r.label}</div>
            <div className={s.barTrack} aria-hidden="true">
              {hasNeg && <span className={s.barAxis} style={{ left: `${zero}%` }} />}
              <div className={cx(s.barFill, r.value < 0 && s.neg)} style={style} />
            </div>
            <div className={s.barValue}>{r.display}</div>
          </div>
        );
      })}
      {(hasNeg || maxPos > 0) && (
        <div className={s.barKey} aria-hidden="true">
          {maxPos > 0 && <span><i />{positiveLabel}</span>}
          {hasNeg && <span><i className={s.negKey} />{negativeLabel}</span>}
        </div>
      )}
    </div>
  );
}

export function Row({ children, between, gap = 8 }: { children: ReactNode; between?: boolean; gap?: number }) {
  return <div className={s.row} style={{ gap, justifyContent: between ? 'space-between' : undefined }}>{children}</div>;
}

export function Grid({ cols = 2, children }: { cols?: 2 | 3 | 4; children: ReactNode }) {
  return <div className={cx(s.grid, s[`cols${cols}`])}>{children}</div>;
}

/** A form on the left, the thing it feeds on the right; stacks on narrow screens. */
export function Split({ children, side = 400 }: { children: ReactNode; side?: number }) {
  return <div className={s.split} style={{ ['--side-col' as string]: `${side}px` }}>{children}</div>;
}

/** Fields stacked with even rhythm. */
export function FormStack({ children }: { children: ReactNode }) {
  return <div className={s.formStack}>{children}</div>;
}

export function Stack({ children, gap = 16 }: { children: ReactNode; gap?: number }) {
  return <div style={{ display: 'grid', gap, minWidth: 0 }}>{children}</div>;
}

export function Mono({ children }: { children: ReactNode }) {
  return <pre className={s.mono} tabIndex={0}>{children}</pre>;
}
