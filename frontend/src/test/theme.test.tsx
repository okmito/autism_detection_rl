import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import { ThemeProvider, THEME_STORAGE_KEY } from '../hooks/theme';
import { ThemeToggle } from '../components/layout/ThemeToggle';
import { mockSystemScheme } from './setup';

const html = () => document.documentElement.getAttribute('data-theme');
const mount = () => render(<ThemeProvider><ThemeToggle /></ThemeProvider>);

describe('theme', () => {
  it('uses the light theme when the system is light and nothing is stored', () => {
    mockSystemScheme(false);
    mount();
    expect(html()).toBe('light');
  });

  it('respects a dark system preference on the first visit, without writing to storage', () => {
    mockSystemScheme(true);
    mount();
    expect(html()).toBe('dark');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
  });

  it('a stored choice beats the system preference', () => {
    mockSystemScheme(true);
    localStorage.setItem(THEME_STORAGE_KEY, 'light');
    mount();
    expect(html()).toBe('light');
  });

  it('ignores a corrupt stored value', () => {
    mockSystemScheme(false);
    localStorage.setItem(THEME_STORAGE_KEY, 'purple');
    mount();
    expect(html()).toBe('light');
  });

  it('toggle switches the theme, flips the switch state and persists the choice', async () => {
    mount();
    const toggle = screen.getByRole('switch', { name: /dark theme/i });
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    await userEvent.click(toggle);
    expect(html()).toBe('dark');
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
    await userEvent.click(toggle);
    expect(html()).toBe('light');
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');
  });

  it('a remounted app (page reload) restores the persisted theme', async () => {
    const first = mount();
    await userEvent.click(screen.getByRole('switch'));
    first.unmount();
    document.documentElement.removeAttribute('data-theme');
    mount();
    expect(html()).toBe('dark');
  });

  it('follows the system until the person chooses, then stops following', async () => {
    const sys = mockSystemScheme(false);
    mount();
    act(() => sys.set(true));
    expect(html()).toBe('dark');
    await userEvent.click(screen.getByRole('switch')); // explicit: dark -> light
    expect(html()).toBe('light');
    act(() => sys.set(true));
    expect(html()).toBe('light');
  });

  it('still works when localStorage throws (blocked site data)', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
    mockSystemScheme(true);
    mount();
    expect(html()).toBe('dark');
    await userEvent.click(screen.getByRole('switch'));
    expect(html()).toBe('light');
  });
});
