import { createContext, useCallback, useContext, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { Modal } from './Modal';
import { Button } from './Button';
import { Input } from './Inputs';
import { Field } from './Field';

/** Promise-based replacements for window.confirm / window.prompt, rendered with the app's own Modal. */

interface ConfirmOptions { title: string; message: ReactNode; confirmLabel?: string; danger?: boolean }
interface PromptOptions { title: string; label: string; defaultValue?: string; confirmLabel?: string }

interface DialogApi {
  confirm: (o: ConfirmOptions) => Promise<boolean>;
  prompt: (o: PromptOptions) => Promise<string | null>;
}

type Pending =
  | { kind: 'confirm'; opts: ConfirmOptions; resolve: (v: boolean) => void }
  | { kind: 'prompt'; opts: PromptOptions; resolve: (v: string | null) => void };

const DialogContext = createContext<DialogApi | null>(null);

export function DialogProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<Pending | null>(null);
  const [text, setText] = useState('');
  const settled = useRef(false);

  const confirm = useCallback((opts: ConfirmOptions) => new Promise<boolean>((resolve) => {
    settled.current = false;
    setPending({ kind: 'confirm', opts, resolve });
  }), []);
  const prompt = useCallback((opts: PromptOptions) => new Promise<string | null>((resolve) => {
    settled.current = false;
    setText(opts.defaultValue ?? '');
    setPending({ kind: 'prompt', opts, resolve });
  }), []);
  const api = useMemo(() => ({ confirm, prompt }), [confirm, prompt]);

  const finish = (value: boolean | string | null) => {
    if (!pending || settled.current) return;
    settled.current = true;
    if (pending.kind === 'confirm') pending.resolve(value === true);
    else pending.resolve(typeof value === 'string' ? value : null);
    setPending(null);
  };

  const cancel = () => finish(pending?.kind === 'confirm' ? false : null);
  const submit = (e: FormEvent) => { e.preventDefault(); finish(text); };

  return (
    <DialogContext.Provider value={api}>
      {children}
      {pending?.kind === 'confirm' && (
        <Modal
          open
          title={pending.opts.title}
          onClose={cancel}
          footer={<>
            <Button variant="ghost" onClick={cancel}>Cancel</Button>
            <Button variant={pending.opts.danger ? 'danger' : 'primary'} data-autofocus onClick={() => finish(true)}>{pending.opts.confirmLabel ?? 'Confirm'}</Button>
          </>}
        >
          {pending.opts.message}
        </Modal>
      )}
      {pending?.kind === 'prompt' && (
        <Modal open title={pending.opts.title} onClose={cancel}>
          <form onSubmit={submit} style={{ display: 'grid', gap: 16 }}>
            <Field label={pending.opts.label}>
              <Input value={text} onChange={(e) => setText(e.target.value)} />
            </Field>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <Button variant="ghost" onClick={cancel}>Cancel</Button>
              <Button type="submit">{pending.opts.confirmLabel ?? 'Save'}</Button>
            </div>
          </form>
        </Modal>
      )}
    </DialogContext.Provider>
  );
}

export function useDialogs(): DialogApi {
  const ctx = useContext(DialogContext);
  if (!ctx) throw new Error('useDialogs must be used inside <DialogProvider>');
  return ctx;
}
