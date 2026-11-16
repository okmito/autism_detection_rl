import { forwardRef, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from 'react';
import { ChevronDown, Search, X } from 'lucide-react';
import { cx } from './cx';
import s from './Form.module.css';

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...rest }, ref) {
  return <input ref={ref} className={cx(s.control, className)} {...rest} />;
});

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea({ className, ...rest }, ref) {
  return <textarea ref={ref} className={cx(s.control, s.textarea, className)} {...rest} />;
});

/** Native <select> (best keyboard/mobile behaviour), styled to match. */
export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select({ className, children, ...rest }, ref) {
  return (
    <span className={s.selectWrap}>
      <select ref={ref} className={cx(s.control, s.select, className)} {...rest}>{children}</select>
      <ChevronDown size={16} className={s.chevron} aria-hidden="true" />
    </span>
  );
});

interface SearchFieldProps {
  value: string;
  onChange: (v: string) => void;
  label: string;
  placeholder?: string;
}

export function SearchField({ value, onChange, label, placeholder }: SearchFieldProps) {
  return (
    <span className={s.searchWrap}>
      <Search size={16} className={s.searchIcon} aria-hidden="true" />
      <input
        type="search"
        className={cx(s.control, s.search)}
        value={value}
        placeholder={placeholder ?? label}
        aria-label={label}
        onChange={(e) => onChange(e.target.value)}
      />
      {value && (
        <button type="button" className={s.clear} aria-label={`Clear ${label.toLowerCase()}`} onClick={() => onChange('')}>
          <X size={14} aria-hidden="true" />
        </button>
      )}
    </span>
  );
}

interface CheckProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  label: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function Checkbox({ label, checked, onChange, className, ...rest }: CheckProps) {
  return (
    <label className={cx(s.check, className)}>
      <input type="checkbox" className={s.checkInput} checked={checked} onChange={(e) => onChange(e.target.checked)} {...rest} />
      <span className={s.checkBox} aria-hidden="true">
        <svg viewBox="0 0 16 16" width="12" height="12"><path d="M3 8.5l3.2 3L13 4.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </span>
      <span>{label}</span>
    </label>
  );
}

interface ToggleProps {
  label: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function Toggle({ label, checked, onChange }: ToggleProps) {
  return (
    <label className={s.toggle}>
      <input type="checkbox" role="switch" className={s.toggleInput} checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className={s.toggleTrack} aria-hidden="true"><span className={s.toggleThumb} /></span>
      <span>{label}</span>
    </label>
  );
}
