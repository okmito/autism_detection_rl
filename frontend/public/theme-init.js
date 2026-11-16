// Runs synchronously in <head>, before the page paints, so the page never flashes the wrong theme.
// Keep in sync with src/hooks/theme.tsx (same storage key and attribute).
(function () {
  var theme = '';
  try { theme = localStorage.getItem('nb_ui_theme') || ''; } catch (e) { /* storage blocked: fall through to system */ }
  if (theme !== 'light' && theme !== 'dark') {
    theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.setAttribute('data-theme', theme);
  document.documentElement.style.colorScheme = theme;
})();
