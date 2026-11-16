import type { ReactNode } from 'react';
import { Card, LoadingState, Notice } from '../components/ui';
import type { LoadState } from '../hooks/useLoad';

/** Shared loading / error handling for views that fetch their data up front. */
export function ViewState<T>({ load, children }: { load: LoadState<T>; children: (data: T) => ReactNode }) {
  if (load.data === null && load.loading) return <LoadingState />;
  if (load.data === null) return <Card><Notice tone="danger">Failed to load module: {load.error ?? 'unknown error'}</Notice></Card>;
  return <>{children(load.data)}</>;
}
