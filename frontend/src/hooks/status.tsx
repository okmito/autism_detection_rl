import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

interface StatusApi {
  status: string;
  /** Shows `message` in the header pill while `fn` runs. */
  withStatus: <T>(message: string, fn: () => Promise<T>) => Promise<T>;
}

const StatusContext = createContext<StatusApi | null>(null);

export function StatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState('');
  const withStatus = useCallback(async <T,>(message: string, fn: () => Promise<T>): Promise<T> => {
    setStatus(message);
    try { return await fn(); } finally { setStatus(''); }
  }, []);
  const value = useMemo(() => ({ status, withStatus }), [status, withStatus]);
  return <StatusContext.Provider value={value}>{children}</StatusContext.Provider>;
}

export function useStatus(): StatusApi {
  const ctx = useContext(StatusContext);
  if (!ctx) throw new Error('useStatus must be used inside <StatusProvider>');
  return ctx;
}
