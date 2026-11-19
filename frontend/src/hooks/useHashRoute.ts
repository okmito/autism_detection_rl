import { useEffect, useState } from 'react';
import { parseHash, type RouteKey } from '../routes';

export function useHashRoute(): RouteKey {
  const [route, setRoute] = useState<RouteKey>(() => parseHash(window.location.hash));
  useEffect(() => {
    if (!window.location.hash) window.location.hash = '#/overview';
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}
