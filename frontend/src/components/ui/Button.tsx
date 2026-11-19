import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react';
import { Loader2 } from 'lucide-react';
import { cx } from './cx';
import s from './Button.module.css';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'dangerQuiet';
type Size = 'sm' | 'md';

interface Common {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
  children?: ReactNode;
}

type ButtonProps = Common & ButtonHTMLAttributes<HTMLButtonElement> & { loading?: boolean };
type LinkProps = Common & AnchorHTMLAttributes<HTMLAnchorElement>;

export function Button({ variant = 'primary', size = 'md', icon, loading, children, className, disabled, type = 'button', ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      className={cx(s.btn, s[variant], s[size], className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <Loader2 size={16} className={s.spin} aria-hidden="true" /> : icon}
      {children}
    </button>
  );
}

/** An anchor that looks like a Button (used for in-app hash navigation). */
export function ButtonLink({ variant = 'secondary', size = 'md', icon, children, className, ...rest }: LinkProps) {
  return (
    <a className={cx(s.btn, s[variant], s[size], className)} {...rest}>
      {icon}
      {children}
    </a>
  );
}
