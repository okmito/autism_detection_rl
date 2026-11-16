import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

/** jsdom has no matchMedia; tests control the "system" colour scheme through this. */
export function mockSystemScheme(dark: boolean) {
  const listeners = new Set<() => void>();
  const mq = {
    matches: dark,
    media: '(prefers-color-scheme: dark)',
    addEventListener: (_: string, fn: () => void) => listeners.add(fn),
    removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
  };
  window.matchMedia = vi.fn().mockReturnValue(mq) as unknown as typeof window.matchMedia;
  return { set(next: boolean) { mq.matches = next; listeners.forEach((fn) => fn()); } };
}

beforeEach(() => {
  window.scrollTo = vi.fn() as unknown as typeof window.scrollTo;
  localStorage.clear();
  sessionStorage.clear();
  document.documentElement.removeAttribute('data-theme');
  window.location.hash = '';
  mockSystemScheme(false);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
