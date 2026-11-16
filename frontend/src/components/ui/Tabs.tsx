import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react';
import s from './Tabs.module.css';

export interface TabItem<K extends string> {
  key: K;
  label: string;
  badge?: ReactNode;
}

interface TabsProps<K extends string> {
  items: TabItem<K>[];
  value: K;
  onChange: (key: K) => void;
  label: string;
  /** Render the active panel; the tab/panel ids are wired for aria-controls / aria-labelledby. */
  children: ReactNode;
}

export function Tabs<K extends string>({ items, value, onChange, label, children }: TabsProps<K>) {
  const uid = useId();
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});

  const onKey = (e: KeyboardEvent, idx: number) => {
    const n = items.length;
    let next = -1;
    if (e.key === 'ArrowRight') next = (idx + 1) % n;
    else if (e.key === 'ArrowLeft') next = (idx - 1 + n) % n;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = n - 1;
    if (next < 0) return;
    e.preventDefault();
    const item = items[next];
    if (!item) return;
    onChange(item.key);
    refs.current[item.key]?.focus();
  };

  return (
    <div>
      <div role="tablist" aria-label={label} className={s.list}>
        {items.map((it, i) => {
          const active = it.key === value;
          return (
            <button
              key={it.key}
              ref={(el) => { refs.current[it.key] = el; }}
              role="tab"
              type="button"
              id={`${uid}-tab-${it.key}`}
              aria-selected={active}
              aria-controls={`${uid}-panel-${it.key}`}
              tabIndex={active ? 0 : -1}
              className={s.tab}
              onClick={() => onChange(it.key)}
              onKeyDown={(e) => onKey(e, i)}
            >
              {it.label}
              {it.badge}
            </button>
          );
        })}
      </div>
      <div role="tabpanel" id={`${uid}-panel-${value}`} aria-labelledby={`${uid}-tab-${value}`} className={s.panel}>
        {children}
      </div>
    </div>
  );
}
