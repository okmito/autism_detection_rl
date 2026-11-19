import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { Checkbox, DialogProvider, ProgressBar, SearchField, Tabs, Toggle, ToastProvider, useDialogs, useToast } from '../components/ui';

function TabsHarness() {
  const [v, setV] = useState<'a' | 'b' | 'c'>('a');
  return (
    <Tabs label="Demo" value={v} onChange={setV} items={[{ key: 'a', label: 'Alpha' }, { key: 'b', label: 'Beta' }, { key: 'c', label: 'Gamma' }]}>
      <p>panel-{v}</p>
    </Tabs>
  );
}

describe('Tabs', () => {
  it('exposes tab semantics and moves with arrow keys, wrapping around', async () => {
    render(<TabsHarness />);
    const alpha = screen.getByRole('tab', { name: 'Alpha' });
    expect(alpha).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tabpanel')).toHaveTextContent('panel-a');
    alpha.focus();
    await userEvent.keyboard('{ArrowRight}');
    expect(screen.getByRole('tab', { name: 'Beta' })).toHaveFocus();
    expect(screen.getByRole('tabpanel')).toHaveTextContent('panel-b');
    await userEvent.keyboard('{ArrowLeft}{ArrowLeft}');
    expect(screen.getByRole('tab', { name: 'Gamma' })).toHaveFocus();
    expect(screen.getByRole('tab', { name: 'Alpha' })).toHaveAttribute('tabindex', '-1');
  });
});

describe('form controls', () => {
  it('Checkbox and Toggle report changes', async () => {
    const onCheck = vi.fn(); const onToggle = vi.fn();
    render(<><Checkbox label="Agree" checked={false} onChange={onCheck} /><Toggle label="Enable" checked={false} onChange={onToggle} /></>);
    await userEvent.click(screen.getByLabelText('Agree'));
    await userEvent.click(screen.getByRole('switch', { name: 'Enable' }));
    expect(onCheck).toHaveBeenCalledWith(true);
    expect(onToggle).toHaveBeenCalledWith(true);
  });

  it('SearchField shows a clear button only when it has text', async () => {
    const Harness = () => { const [v, setV] = useState(''); return <SearchField label="Search things" value={v} onChange={setV} />; };
    render(<Harness />);
    expect(screen.queryByRole('button', { name: /clear/i })).toBeNull();
    await userEvent.type(screen.getByRole('searchbox', { name: 'Search things' }), 'abc');
    await userEvent.click(screen.getByRole('button', { name: /clear search things/i }));
    expect(screen.getByRole('searchbox')).toHaveValue('');
  });

  it('ProgressBar clamps its value and is labelled', () => {
    render(<ProgressBar value={9} max={6} label="Progress" />);
    const bar = screen.getByRole('progressbar', { name: 'Progress' });
    expect(bar).toHaveAttribute('aria-valuenow', '6');
  });
});

function ConfirmHarness({ onResult }: { onResult: (v: boolean) => void }) {
  const { confirm } = useDialogs();
  return <button onClick={async () => onResult(await confirm({ title: 'Delete it?', message: 'Sure?', confirmLabel: 'Delete', danger: true }))}>open</button>;
}

describe('Dialogs', () => {
  it('confirm resolves true on confirm, false on cancel and false on Escape; focus returns to the opener', async () => {
    const onResult = vi.fn();
    render(<DialogProvider><ConfirmHarness onResult={onResult} /></DialogProvider>);
    const opener = screen.getByRole('button', { name: 'open' });

    await userEvent.click(opener);
    expect(screen.getByRole('dialog', { name: 'Delete it?' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(onResult).toHaveBeenLastCalledWith(true));
    expect(screen.queryByRole('dialog')).toBeNull();

    await userEvent.click(opener);
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(onResult).toHaveBeenLastCalledWith(false));

    await userEvent.click(opener);
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(onResult).toHaveBeenLastCalledWith(false));
    expect(opener).toHaveFocus();
  });

  it('traps Tab inside the dialog', async () => {
    render(<DialogProvider><ConfirmHarness onResult={() => {}} /></DialogProvider>);
    await userEvent.click(screen.getByRole('button', { name: 'open' }));
    const dialog = screen.getByRole('dialog');
    for (let i = 0; i < 6; i++) {
      await userEvent.tab();
      expect(dialog).toContainElement(document.activeElement as HTMLElement);
    }
  });
});

describe('Toast', () => {
  it('announces errors with role=alert and can be dismissed', async () => {
    const Harness = () => { const t = useToast(); return <button onClick={() => t.error('Boom')}>go</button>; };
    render(<ToastProvider><Harness /></ToastProvider>);
    await userEvent.click(screen.getByText('go'));
    expect(screen.getByRole('alert')).toHaveTextContent('Boom');
    await userEvent.click(screen.getByRole('button', { name: /dismiss/i }));
    expect(screen.queryByRole('alert')).toBeNull();
  });
});
