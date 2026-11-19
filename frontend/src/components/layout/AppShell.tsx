import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Info, Lock, Menu, X } from 'lucide-react';
import { ROUTES, type RouteDef, type RouteKey } from '../../routes';
import { ThemeToggle } from './ThemeToggle';
import s from './AppShell.module.css';

const GROUPS: { key: RouteDef['group']; label: string }[] = [
  { key: 'care', label: 'Screening and support' },
  { key: 'research', label: 'Research evidence' },
  { key: 'data', label: 'Your data' },
];

/** Two banks joined by a span: a bridge, drawn once, in the cobalt of the product. */
function Mark() {
  return (
    <svg className={s.mark} viewBox="0 0 32 32" width="32" height="32" aria-hidden="true">
      <rect width="32" height="32" rx="10" fill="currentColor" />
      <path d="M6 21.5h20" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
      <path d="M9 21.5c1.2-6.5 4-9.5 7-9.5s5.8 3 7 9.5" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
      <path d="M12.5 21.5v-4M16 21.5v-9.5M19.5 21.5v-4" stroke="#fff" strokeOpacity=".7" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function Brand() {
  return (
    <div className={s.brand}>
      <Mark />
      <div className={s.brandText}>
        <span className={s.brandName}>NeuroBridge</span>
        <span className={s.brandSub}>Research prototype</span>
      </div>
    </div>
  );
}

function Nav({ active, onNavigate }: { active: RouteKey; onNavigate: () => void }) {
  return (
    <nav aria-label="Primary" className={s.nav}>
      {GROUPS.map((g) => (
        <div key={g.key} className={s.group}>
          <h2 className={s.groupLabel}>{g.label}</h2>
          {ROUTES.filter((r) => r.group === g.key).map((r) => {
            const Icon = r.icon;
            return (
              <a key={r.key} href={`#/${r.key}`} className={s.link} aria-current={r.key === active ? 'page' : undefined} onClick={onNavigate}>
                <Icon size={18} strokeWidth={1.75} aria-hidden="true" />
                <span>{r.nav}</span>
              </a>
            );
          })}
        </div>
      ))}
    </nav>
  );
}

interface Props {
  route: RouteDef;
  status: string;
  children: ReactNode;
}

export function AppShell({ route, status, children }: Props) {
  const [drawer, setDrawer] = useState(false);
  const mainRef = useRef<HTMLElement>(null);
  const menuRef = useRef<HTMLButtonElement>(null);

  // Move focus to the page on navigation so keyboard and screen-reader users land on the new content.
  useEffect(() => {
    setDrawer(false);
    mainRef.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  }, [route.key]);

  useEffect(() => {
    if (!drawer) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setDrawer(false); menuRef.current?.focus(); } };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [drawer]);

  return (
    <div className={s.shell}>
      <a className={s.skip} href="#main" onClick={(e) => { e.preventDefault(); mainRef.current?.focus(); }}>Skip to content</a>

      <header className={s.mobileBar}>
        <button ref={menuRef} type="button" className={s.iconBtn} aria-label={drawer ? 'Close menu' : 'Open menu'} aria-expanded={drawer} aria-controls="sidebar" onClick={() => setDrawer((d) => !d)}>
          {drawer ? <X size={20} aria-hidden="true" /> : <Menu size={20} aria-hidden="true" />}
        </button>
        <Brand />
        <ThemeToggle />
      </header>

      {drawer && <div className={s.scrim} onClick={() => setDrawer(false)} aria-hidden="true" />}

      <aside id="sidebar" className={`${s.side} ${drawer ? s.open : ''}`}>
        <div className={s.sideHead}><Brand /></div>
        <Nav active={route.key} onNavigate={() => setDrawer(false)} />
        <div className={s.sideFoot}>
          <div className={s.themeRow}><span>Dark theme</span><ThemeToggle /></div>
          <p><Lock size={13} aria-hidden="true" /> Everything you enter stays on this computer.</p>
        </div>
      </aside>

      <main id="main" ref={mainRef} tabIndex={-1} className={s.main}>
        <div className={s.content}>
          {/* The overview opens with its own welcome; its h1 stays in the document for assistive technology. */}
          <header className={`${s.pageHead} ${route.key === 'overview' ? s.pageHeadQuiet : ''}`}>
            <div className={s.headText}>
              <h1 className={route.key === 'overview' ? 'visually-hidden' : s.title}>{route.title}</h1>
              {route.key !== 'overview' && <p className={s.sub}>{route.sub}</p>}
            </div>
            <div className={s.headAside}>
              <div className={s.status} aria-live="polite">{status}</div>
              <p className={s.notDx}><Info size={14} aria-hidden="true" /> Screening, not a diagnosis</p>
            </div>
          </header>
          {children}
          <footer className={s.footer}>
            <p><strong>Important:</strong> NeuroBridge is an explainable, adaptive screening and support <em>research prototype</em>. It provides screening/referral-research estimates and non-diagnostic support ideas only — it does not confirm or rule out autism and is not a substitute for professional assessment. Some research results shown are based on questionnaire-derived/circular labels and have known limitations.</p>
          </footer>
        </div>
      </main>
    </div>
  );
}
