# TODO — SOFT / SYSTEM React + TypeScript frontend

Central progress tracker for the NeuroBridge UI redesign. Legend:
`[ ]` not started · `[x]` done and verified · `[~]` built, partly verified (the note says what is missing) · `[!]` blocked

Last updated: 2026-11-16 (Patient Leaflet redesign, see section 12). State: the full UI is built, type-checks under strict mode, builds, and passes 44 automated
tests. **Nothing has been looked at in a real browser** (see Blocked), so every visual claim below is unverified.

## Findings from inspection

- Backend: Flask (`app/server.py`), JSON API under `/api/*`, SQLite. Preserved; one additive route (`/ui/`) was added.
- Legacy UI: hand-written vanilla JS SPA (`app/static/`), nine hash-routed views. It is pinned by Python/jsdom tests and
  has uncommitted edits by the user, so it was left untouched. `scripts/demo_static/index.html` is a separate demo, out of scope.
- The server's CSP (`script-src 'self'`, `style-src 'self'`, no inline script/style elements, `font-src 'self'`) shaped the design:
  the theme guard is an external script, fonts are the system stack, styling is CSS Modules (no CSS-in-JS).
- Mutating requests must be `application/json`; the API client always sends that header.

## Decisions

- **D1** The app lives in `frontend/` with its own `package.json` (the root one is jsdom tooling for the legacy tests).
- **D2** Vite builds to `app/static/ui/` (gitignored, generated); Flask serves it at `/ui/`. Legacy UI stays at `/`.
- **D3** Hash routing with the same URLs as the legacy UI. No router dependency.
- **D4** CSS Modules + `styles/tokens.css`; `lucide-react` icons; no UI framework, no data-fetching library.
- **D5** System font stack (CSP).
- **D6** `window.confirm` / `window.prompt` were replaced by in-app, focus-trapped dialogs (promise-based `useDialogs`).

## 1. Repository inspection and planning
- [x] Read server, CSP, API surface, legacy SPA, and the tests that pin it
- [x] Check toolchain (Node 24, npm registry)
- [x] Record findings and decisions
- [!] Reference image `/mnt/data/image(4).png` does not exist on this machine. Design follows the written palette alone; supply the image to check the match.

## 2. React + TypeScript + Vite setup
- [x] `frontend/` scaffold: React 19, strict TS (`noUncheckedIndexedAccess`, unused checks), Vite, Lucide, Vitest + Testing Library
- [x] `vite.config.ts` (base `/static/ui/`, outDir `app/static/ui`)
- [~] Dev proxy `/api` → `:8010` is configured; `npm run dev` was never started
- [x] Scripts `build` (tsc then vite), `typecheck`, `test` all run clean
- [x] Flask route `/ui` and `/ui/` (`tests/test_ui_route.py`: 5 tests; 404 with build hint when not built)
- [x] External blocking `theme-init.js`, loaded before the app bundle (asserted by a Python test)

## 3. Design tokens and theme architecture
- [x] Tokens: colour (light + dark), spacing, type scale, radii, shadows, motion, z-index (`styles/tokens.css`)
- [x] `ThemeProvider`: system preference when nothing is stored, `localStorage` persistence, follows OS changes until a manual choice, safe when storage throws (8 tests)
- [x] Dark palette is hand-tuned, not an inversion; text/surface pairs measured, see section 7

## 4. Reusable UI components (`components/ui`, `components/layout`)
- [x] Button / ButtonLink, Card / MetricCard, Input, Textarea, SearchField, Select, Field
- [x] Toggle, Checkbox, Tabs (roving tabindex, arrow/Home/End), ProgressBar, Badge, Notice
- [x] Toast (aria-live, role=alert for errors), Modal + Dialogs (focus trap, Esc, focus restore), EmptyState, Spinner, LoadingState skeleton
- [x] DataTable (scrolls inside its own box), ItemList / Item, KeyValue, BarList, Mono
- [x] AppShell (sidebar, mobile drawer, skip link, focus moves to main on navigation), ThemeToggle (sun/moon), ErrorBoundary
- [~] Keyboard and ARIA behaviour is tested in jsdom; how it looks and feels has not been seen

## 5. Application views (all nine legacy views ported, same endpoints)
- [x] Typed API client + payload types; types compared with real payloads from the running server (meta, start, answer, stop/result, session detail, sessions, RL scenarios)
- [x] Overview (render test per route; data-bearing path is not separately tested)
- [x] Guided Screening: start/resume, answer, rl_adaptive skip/stop, auto mode, fallback banner, 409 resync, result view (5 tests; real API exercised with curl, not through the UI)
- [x] Context Explorer: create, edit (PUT), search/filter, remove behind a dialog (2 tests)
- [x] Support Studio: suggest, use, save, mark tried via prompt dialog, empty-selection warning (3 tests)
- [x] Timeline: search, kind filter (1 test). [~] JSON export download is untested (needs a browser)
- [x] RL Intelligence Lab: replay, step, restart, failed replay (2 tests). [~] Play/pause timer and "Compare policies" untested
- [x] Explainable AI: lists completed sessions (active ones are excluded), loads the explanation (1 test)
- [x] Research & Analytics: artifact summaries, missing-H1 notice, CSV tables tab (1 test)
- [x] Privacy & Consent: consent save, destructive actions require confirmation, JSON content type (2 tests). [~] Export download untested

## 6. Light mode
- [~] Ivory / cobalt / mist / charcoal palette applied through tokens. Never rendered, so spacing, shadows and the "soft, tactile" feel are unchecked

## 7. Dark mode
- [~] Every component takes its colours from tokens (the exceptions are the white-on-accent text and the gradient brand mark). Never rendered
- [x] Contrast computed for token pairs (WCAG ratio). Light: text 13.8, muted 6.0, faint 5.4, accent-ink 7.3, white on accent 5.5, status text on soft fills 5.1–5.3. Dark: text 12.2, muted 6.7, faint 5.0, accent-ink 7.7, white on accent 4.7, status text 7.2–8.1, white on danger 5.2. Two pairs failed on the initial pass and were corrected (faint text, dark accent).
- [ ] Measured on rendered pages (overlays, focus rings, disabled state) — needs a browser

## 8. Responsive design
- [~] Breakpoints written (drawer below 1024px, single column below 760px, tables scroll in their own box). **Unverified**: no screenshots at 1440 / 1024 / 768 / 375, no horizontal-overflow check
- [ ] Verify at four widths

## 9. Functionality and interactions
- [x] Workflows covered by tests listed in section 5
- [ ] Walk every workflow in a browser against the live server (downloads, Play timer, dialogs, drawer)
- [~] Reduced-motion rule and focus rings are in the CSS; not observed

## 10. Testing and quality assurance (actual results)
- [x] `tsc --noEmit` (strict): clean
- [x] `vite build`: clean — JS 309.5 kB (94.5 kB gzip), CSS 23.8 kB (5.6 kB gzip)
- [x] Vitest: 4 files, **44 passed**. Mutation checks confirmed the suite catches: literal Yes/No answers, theme not persisted, wrong strategy `source`, ignored timeline filter
- [~] One guard is redundant under test: removing the `answering` ref in Screening is not caught, because the answer buttons are already `disabled` while a request is in flight. The ref matters just for same-tick programmatic calls
- [x] Python suite: **12 failed / 1015 passed / 80 skipped**. Baseline before any change: 12 failed / 1010 passed / 80 skipped. The failing set is identical (frozen-artifact hashes, markdown phrase rule, V4/V7 validation); the +5 passes are the new route tests
- [x] Legacy UI tests (`test_app_frontend`, `test_frontend_regression`, `test_app_headers`) still pass
- [x] Live server (scratch database, real model): `/ui/` returns 200 with the CSP; `theme-init.js` and bundles served; screening start → answer → stop → result worked through the API
- [!] Browser console errors, CSP violations in a real page, theme persistence across a real reload, no-flash check, screenshots: **not done**. See Blocked.
- [ ] CSP behaviour in a browser (the policy compatibility of the built HTML is checked statically)

## 11. Final polish
- [x] README section "4a" (build, run, dev server, tests, theme)
- [x] `TODO.md` reconciled with the real state
- [ ] Visual polish pass after screenshots exist

## Blocked

- **Browser QA.** The cached Playwright Chromium needs system libraries that are not installed (`libnss3`, `libnspr4`, `libatk-1.0`, X11 libs, `libgbm`, `libxkbcommon`, `libasound` and others; Arch Linux). Installing them needs root. An attempt to fetch them into a scratch directory was declined by the environment's permission check, and I stopped there. To unblock: `sudo pacman -S nss nspr at-spi2-core libxcomposite libxdamage libxrandr libxkbcommon mesa alsa-lib` (or any browser the tools can drive), then run the screenshot/console script.

## Remaining / open decisions for the user

- Run the browser pass (Blocked, above), then fix whatever it shows. Expect some visual adjustments.
- Promote the React UI from `/ui/` to `/`? That would retire the legacy page and the tests that pin it (`tests/test_app_frontend.py`, `tests/test_frontend_regression.py`, parts of `tests/test_app_headers.py`), all of which have your uncommitted edits.
- Supply the reference image if a closer visual match is wanted.
- Nothing was committed.

## 12. Redesign: "The Patient Leaflet" (2026-11-16)

Replaces the SOFT / SYSTEM ivory/cobalt look. Product context in `PRODUCT.md`; design contract in `.impeccable/surfaces/frontend-src-app-tsx.md`; system in `DESIGN.md`.

- [x] Tokens rewritten: ink-blue rail, white ground, grey panels, action blue, yellow reserved for "why" and "not a diagnosis", amber for research caveats; four-step type scale (14/16/20/32); Public Sans bundled via @fontsource (CSP-safe)
- [x] Shell: grouped ink rail, yellow screening strip on every page, ruled page head
- [x] All shared components restyled (buttons, callouts, badges, tabs, forms, tables, dialogs, toasts, disclosure chevrons)
- [x] Screening rebuilt: plain start screen, one-tap lettered answers, "Why this question?", fixed 0-100 estimate scale, numbered pathway (skipped / remaining / not-needed rows stay visible)
- [x] Result rebuilt: plain headline, "Not a diagnosis" meaning, path, support ideas, clinician numbers behind disclosures
- [x] Overview, Explain (pathway + scale), Research (artifact reference table) rebuilt; other views inherit through tokens/components
- [x] Browser QA done: 9 routes x light/dark x 1440/390, no console errors or CSP violations, no horizontal overflow
- [x] Independent finish review: 8 material fixes, all resolved over two verdict passes; one last spacing tweak verified by screenshot
- [x] Vitest 44/44 (3 queries updated: the item code now renders in its own chip beside the question); Python suite 12 failed / 1015 passed / 80 skipped (same 12 environmental failures)
- [ ] Bespoke layouts for Context, Support, Timeline, RL Lab, Privacy (currently styled through shared components)

## 13. Redesign: soft ivory / cobalt SaaS (2026-11-16, owner brief)

Replaces the Patient Leaflet look from section 12. Contract: `.impeccable/surfaces/frontend-src-app-tsx.md`.

### Audit findings (before)
- Every view wraps each section in a bordered `Card`; `Grid cols=2` of cards is the layout everywhere (Context, Support, Privacy, RL Lab, Explain).
- 2px ink rules under every page head and section, 8px coloured callout rules, a yellow full-width strip: heavy, "printed" look.
- Hairline dividers between every list row; tables boxed in a bordered wrapper.
- Overview: a dark navy hero block, session ids in the feed, an "At a glance" list that reads as a form.
- Inline `style={{...}}` layout in views (display grid / gap) duplicated 20+ times: no shared layout primitives for form stacks or page sections.
- Reusable and kept: `Card`, `Notice`, `Badge`, `Button`, `Field`/inputs, `DataTable`, `ItemList`, `BarList`, `Pathway`, `RiskScale`, `Tabs`, dialogs, toasts. All colours already come from `tokens.css`.

### Design-system spec (summary; DESIGN.md is rewritten at finish)
- Colour: ivory ground `#f8f6f1`, white surface, warm hairline, cobalt `#1f48c8` (action / active / data), ink `#16181d`. Status: green, amber, red. No yellow. Dark: blue-black.
- Type: Geist Variable, scale 13 / 15 / 17 / 22 / 30 / 44, display tracking -0.03em, tabular figures.
- Shape and depth: radii 10 / 14 / 20, two-layer soft shadow on raised surfaces only, 1px hairlines used sparingly.
- Containers: open sections by default. A raised white surface only for a form, the live question, the screening evidence rail, modals and menus.
- Components: primary / secondary / ghost / danger buttons; 44px fields with soft focus ring; callouts as tinted soft surfaces with icon; pill badges; borderless tables with a quiet header; segmented step progress; activity feed with dot rail.
- Motion: 120 / 200 / 360ms exponential ease-out; one settle on new question / result; reduced motion respected.

### Prioritised tasks
- [x] P0 Tokens + global base (Geist, ivory, cobalt, shadows, dark theme)
- [x] P0 Shell: light sidebar, logo mark, grouped nav, active pill, quiet screening-not-diagnosis line, top bar, mobile drawer
- [x] P0 Shared components: Card (open by default), Notice, Badge, Button, forms, tables, tabs, empty / loading states, dialogs, toasts, theme toggle
- [x] P0 Overview rebuilt (welcome, CTA, inline stats, activity feed, observations, research index)
- [x] P0 Screening: start, live question, pathway, estimate scale, result
- [x] P1 Explainability visuals: diverging contribution bars, belief-trace chart, risk scale
- [x] P1 Context, Support, Timeline (day-grouped feed), RL Lab (stepper), Privacy (settings rows), Research
- [~] P1 Light + dark (captured and clean before the review fixes; not recaptured after them), 1440 / 1024 / 768 / 390 screenshots of all 9 routes, console / CSP / overflow checks
- [x] P1 Vitest, typecheck, build, Python UI route tests
- [ ] P2 Independent finish review, DESIGN.md + sidecar, README note

Review: disposition fix, 8 material fixes applied (session IDs, Research grouping, plain-language copy, RL status labels, Privacy danger weight, mono prose, Overview observations, paired fields). No verdict pass or recapture yet. DESIGN.md still describes the Patient Leaflet.
