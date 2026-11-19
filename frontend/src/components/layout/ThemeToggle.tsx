import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../../hooks/theme';
import s from './ThemeToggle.module.css';

/** Sun/moon switch: both icons are always mounted so the swap can animate (rotate + fade). */
export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const dark = theme === 'dark';
  return (
    <button
      type="button"
      className={s.toggle}
      onClick={toggle}
      role="switch"
      aria-checked={dark}
      aria-label="Dark theme"
      title={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      data-testid="theme-toggle"
    >
      <span className={s.thumb} aria-hidden="true">
        <Sun size={14} className={s.sun} />
        <Moon size={14} className={s.moon} />
      </span>
    </button>
  );
}
