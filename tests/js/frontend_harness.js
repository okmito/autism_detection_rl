// Runs the real app/static/js/app.js against the real app/static/index.html inside jsdom, with `fetch`
// stubbed from a scenario file, and prints a JSON report that the pytest wrappers assert on.
//
//   node tests/js/frontend_harness.js <scenario.json>
//
// scenario = {
//   "hash":  "#/support",                         // route to open
//   "api":   {"/api/meta": {...}, ...},           // exact-path -> JSON body (default: {ok:true,data:[]});
//                                                  // or {__status, __delay, body} for an error / slow reply
//   "prompt": "helpful",                          // value returned by window.prompt (optional)
//   "steps": [ {"wait": 100},
//              {"set": ["#supObs", "obs_1"]},
//              {"click": "#genSug"},
//              {"collect": ["name", "#selector", "value|text|html|exists|count"]} ]
// }
//
// Output = { errors: [...], calls: [{path, method, body}], collected: {name: value}, viewText, viewHtml }.
// This is test tooling only (needs `npm install` for jsdom); nothing here ships with the app.
'use strict';
const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..', '..');
const scenario = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const appJs = fs.readFileSync(path.join(repo, 'app', 'static', 'js', 'app.js'), 'utf8');
// The page's <script src> is evaluated by hand below so that errors surface in this process.
const html = fs
  .readFileSync(path.join(repo, 'app', 'static', 'index.html'), 'utf8')
  .replace(/<script src[^>]*><\/script>/, '');

const errors = [];
const calls = [];
const collected = {};
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (e) => errors.push(String((e && e.message) || e)));
virtualConsole.on('error', (...a) => errors.push(a.join(' ')));
process.on('unhandledRejection', (e) => errors.push('unhandledRejection: ' + String((e && e.message) || e)));

const dom = new JSDOM(html, {
  url: 'http://127.0.0.1:8010/' + (scenario.hash || ''),
  runScripts: 'outside-only',
  pretendToBeVisual: true,
  virtualConsole,
});
const w = dom.window;
const api = scenario.api || {};

w.fetch = async (p, opts) => {
  opts = opts || {};
  calls.push({ path: p, method: opts.method || 'GET', body: opts.body || null });
  const hit = Object.prototype.hasOwnProperty.call(api, p) ? api[p] : { ok: true, data: [] };
  // An entry may be wrapped as {__status, __delay (ms), body} to simulate an error status or a slow server.
  const wrapped = !!(hit && (hit.__status || hit.__delay));
  if (wrapped && hit.__delay) await new Promise((r) => setTimeout(r, hit.__delay));
  const status = wrapped && hit.__status ? hit.__status : 200;
  return {
    ok: status < 400,
    status,
    statusText: status < 400 ? 'OK' : 'ERR',
    json: async () => (wrapped ? hit.body : hit),
  };
};
w.scrollTo = () => {};
if (scenario.breakStorage) {
  // Simulates a browser that blocks storage: every access throws, as in some private windows.
  Object.defineProperty(w, 'localStorage', { get() { throw new w.DOMException('blocked', 'SecurityError'); } });
}
w.confirm = () => true;
w.prompt = () => (scenario.prompt === undefined ? '' : scenario.prompt);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  try {
    w.eval(appJs);
  } catch (e) {
    errors.push('app.js failed to evaluate: ' + e.message);
  }
  await sleep(200);
  const d = w.document;
  for (const step of scenario.steps || []) {
    try {
      if (step.wait !== undefined) await sleep(step.wait);
      else if (step.set) {
        const el = d.querySelector(step.set[0]);
        if (!el) throw new Error('no element ' + step.set[0]);
        el.value = step.set[1];
      } else if (step.click) {
        const el = d.querySelector(step.click);
        if (!el) throw new Error('no element ' + step.click);
        el.click();
        await sleep(step.after === undefined ? 120 : step.after);
      } else if (step.collect) {
        const [name, sel, what] = step.collect;
        const els = Array.from(d.querySelectorAll(sel));
        const el = els[0];
        collected[name] =
          what === 'exists' ? els.length > 0
          : what === 'count' ? els.length
          : !el ? null
          : what === 'value' ? el.value
          : what === 'html' ? el.innerHTML
          : el.textContent;
      }
    } catch (e) {
      errors.push('step ' + JSON.stringify(step) + ' failed: ' + e.message);
    }
  }
  const view = d.querySelector('#view');
  process.stdout.write(
    JSON.stringify({ errors, calls, collected, viewText: view ? view.textContent : '', viewHtml: view ? view.innerHTML : '' })
  );
  process.exit(0);
})();
