import { cloneElement, isValidElement, useId, type ReactElement, type ReactNode } from 'react';
import s from './Form.module.css';

interface FieldProps {
  label: string;
  help?: ReactNode;
  /** A single form control; it receives the generated id and aria-describedby. */
  children: ReactElement<{ id?: string; 'aria-describedby'?: string }>;
}

export function Field({ label, help, children }: FieldProps) {
  const id = useId();
  const helpId = `${id}-help`;
  const control = isValidElement(children)
    ? cloneElement(children, { id, ...(help ? { 'aria-describedby': helpId } : {}) })
    : children;
  return (
    <div className={s.field}>
      <label htmlFor={id} className={s.label}>{label}</label>
      {control}
      {help && <div id={helpId} className={s.help}>{help}</div>}
    </div>
  );
}
