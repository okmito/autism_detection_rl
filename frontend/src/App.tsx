import { ROUTES, type RouteKey } from './routes';
import { AppShell } from './components/layout/AppShell';
import { ErrorBoundary } from './components/layout/ErrorBoundary';
import { DialogProvider, ToastProvider } from './components/ui';
import { StatusProvider, useStatus } from './hooks/status';
import { ThemeProvider } from './hooks/theme';
import { useHashRoute } from './hooks/useHashRoute';
import { Context } from './views/Context';
import { Explain } from './views/Explain';
import { Overview } from './views/Overview';
import { Privacy } from './views/Privacy';
import { Research } from './views/Research';
import { RlLab } from './views/RlLab';
import { Screening } from './views/screening/Screening';
import { Support } from './views/Support';
import { Timeline } from './views/Timeline';

const VIEWS: Record<RouteKey, () => React.JSX.Element> = {
  overview: Overview, screening: Screening, context: Context, support: Support, timeline: Timeline,
  'rl-lab': RlLab, explain: Explain, research: Research, privacy: Privacy,
};

function Routed() {
  const key = useHashRoute();
  const { status } = useStatus();
  const route = ROUTES.find((r) => r.key === key) ?? ROUTES[0]!;
  const View = VIEWS[route.key];
  return (
    <AppShell route={route} status={status}>
      {/* keyed so every navigation remounts the view and re-fetches, like the legacy router */}
      <ErrorBoundary resetKey={route.key}><View key={route.key} /></ErrorBoundary>
    </AppShell>
  );
}

export function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <DialogProvider>
          <StatusProvider>
            <Routed />
          </StatusProvider>
        </DialogProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}
