---
version: 1
slug: "frontend-src-app-tsx"
primary_target: "frontend/src/App.tsx"
related_targets: ["frontend/src/views/screening/Screening.tsx"]
---

# NeuroBridge app (frontend/, served at /ui/)

Scope: the whole React app, all nine views. Mode: Operate. Priority surfaces: Overview (first impression) and Screening (question flow, result, explainability) used by parents and doctors; research views (RL Lab, Explain, Research) serve examiners in the same world.

Constraints: CSP `script-src/style-src/font-src 'self'` (fonts bundled, no inline style elements), same API and hash routes, light and dark themes, keyboard and WCAG AA, screening-not-diagnosis labelling wherever a result appears, never imitate a real health service's branding. All feature behaviour and the Vitest suite are preserved; no new runtime dependencies beyond swapping the bundled font.

Redesign note (2026-11-16): the owner replaced the earlier "Patient Leaflet" world (ink rail, yellow rules, 2px borders) with the soft ivory / cobalt SaaS direction below, in their own words. The old look is anti-reference; product truth (pathway with skipped steps visible, fixed 0 to 100 risk scale with the referral line, plain-language result) carries over.

## Direction contract

THESIS: A calm research clinic, not a dashboard. Every page is an open editorial sheet on ivory: type size, weight and whitespace carry hierarchy, and a white raised surface appears only where something is genuinely held (a form, the live question, the evidence rail). Refuses the Bootstrap grid of bordered cards, the ruled leaflet and yellow callouts.

OWN-WORLD: Warm ivory ground (#f8f6f1) with white surfaces, 1px warm hairlines and a two-layer soft shadow. One cobalt (#1f48c8) for action, active state and data; warm-neutral ink text. Geist variable, a six-step scale (13 / 15 / 17 / 22 / 30 / 44) with tight display tracking. Radii 10 / 14 / 20. Light sidebar one shade deeper than the ground, active item a white pill with soft shadow and a cobalt icon. Callouts are tinted soft surfaces with a line icon, never a side bar. Status hues green / amber / red only; no yellow. Dark theme is blue-black with lifted surfaces and a lighter cobalt.

STORY: A parent answers one question at a time without feeling boxed in, sees why it is asked and where they are, and reads a result that says what it means and cannot mean. A clinician opens the same pathway as evidence. An examiner reads honest, caveated research. Nothing is hidden: skipped and not-needed questions stay marked on the pathway.

FIRST VIEWPORT: Overview: 248px sidebar; header with a welcome H1, one supporting line, and a cobalt "Start screening" primary beside a quiet secondary; four compact stats set inline with no boxes; below, recent screenings as a dotted activity feed beside recent observations as a readable list; research modules as an open index. Screening: segmented step progress, the question at 36px beside a small item chip, large white answer rows that answer on one tap, quiet Stop; one white evidence rail holding why, estimate scale and pathway.

FORM: pinned by the owner's brief, seed: none (pinned). Pin words: "Soft ivory backgrounds and white surfaces. Cobalt blue as the primary accent. Subtle shadows and understated borders. Reduce the number of cards and rectangular containers."

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
