import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react';
import type { Tone } from './Badge';
import s from './Toast.module.css';

interface ToastItem { id: number; message: string; tone: Tone }
interface ToastApi {
  show: (message: string, tone?: Tone) => void;
  error: (message: string) => void;
}

const ToastContext = createContext<ToastApi | null>(null);
const ICONS = { neutral: Info, accent: Info, good: CheckCircle2, warn: AlertTriangle, danger: AlertTriangle } as const;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => setItems((cur) => cur.filter((t) => t.id !== id)), []);
  const show = useCallback((message: string, tone: Tone = 'neutral') => {
    const id = nextId.current++;
    setItems((cur) => [...cur.slice(-3), { id, message, tone }]);
    window.setTimeout(() => dismiss(id), tone === 'danger' ? 7000 : 4000);
  }, [dismiss]);
  const api = useMemo<ToastApi>(() => ({ show, error: (m) => show(m, 'danger') }), [show]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className={s.region} role="region" aria-label="Notifications" aria-live="polite">
        {items.map((t) => {
          const Icon = ICONS[t.tone];
          return (
            <div key={t.id} className={`${s.toast} ${s[t.tone]}`} role={t.tone === 'danger' ? 'alert' : 'status'}>
              <Icon size={18} aria-hidden="true" className={s.icon} />
              <span className={s.msg}>{t.message}</span>
              <button type="button" className={s.close} onClick={() => dismiss(t.id)} aria-label="Dismiss notification"><X size={14} aria-hidden="true" /></button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>');
  return ctx;
}
