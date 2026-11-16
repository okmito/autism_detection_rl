---
name: NeuroBridge
description: Adaptive, explainable autism screening and support research prototype, set as a public-health patient leaflet.
colors:
  ink-rail: "#163a5f"
  ink-rail-deep: "#0f2c4a"
  ink-rail-line: "#2b5279"
  rail-text: "#e4ecf4"
  rail-muted: "#b7c8da"
  action-blue: "#1d5fd1"
  action-blue-hover: "#174fb0"
  action-blue-soft: "#e8f0fd"
  scale-blue-secondary: "#6f9fe8"
  leaflet-yellow: "#ffd23f"
  leaflet-yellow-soft: "#fff7d1"
  leaflet-yellow-line: "#f0dc8a"
  leaflet-yellow-ink: "#4a3b00"
  paper-white: "#ffffff"
  panel-grey: "#eef2f5"
  hover-grey: "#f3f6f8"
  track-grey: "#dde4ea"
  hairline: "#d6dee5"
  rule-grey: "#7a8a99"
  ink-text: "#13202e"
  muted-text: "#4d5d6c"
  faint-text: "#5f6d7a"
  done-green: "#0b7a4b"
  done-green-soft: "#e3f3eb"
  caveat-amber: "#8a5a00"
  caveat-amber-soft: "#fff3d6"
  urgent-red: "#b42318"
  urgent-red-soft: "#fbe7e5"
  scale-low: "#e3edf8"
  scale-high: "#f8e3e1"
  selection-blue: "#c9dbf8"
typography:
  display:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "2rem"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "-0.025em"
    fontFeature: "tnum"
  question:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "2rem"
    fontWeight: 750
    lineHeight: 1.25
    letterSpacing: "-0.018em"
    fontFeature: "tnum"
  headline:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "-0.012em"
    fontFeature: "tnum"
  body:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.55
    fontFeature: "tnum"
  label:
    fontFamily: "Public Sans Variable, Public Sans, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 600
    lineHeight: 1.5
    letterSpacing: "0em"
    fontFeature: "tnum"
  mono:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "0.875rem"
    fontWeight: 400
rounded:
  xs: "3px"
  sm: "6px"
  md: "6px"
  lg: "8px"
  pill: "999px"
spacing:
  "1": "4px"
  "2": "8px"
  "3": "12px"
  "4": "16px"
  "5": "20px"
  "6": "24px"
  "8": "32px"
  "10": "40px"
  "12": "48px"
components:
  button-primary:
    backgroundColor: "{colors.action-blue}"
    textColor: "{colors.paper-white}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: "11px 20px"
    height: "44px"
  button-primary-hover:
    backgroundColor: "{colors.action-blue-hover}"
  button-secondary:
    backgroundColor: "{colors.paper-white}"
    textColor: "{colors.action-blue-hover}"
    rounded: "{rounded.sm}"
    padding: "11px 20px"
    height: "44px"
  button-secondary-hover:
    backgroundColor: "{colors.action-blue-soft}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink-text}"
    rounded: "{rounded.sm}"
    padding: "11px 20px"
  button-danger:
    backgroundColor: "{colors.urgent-red}"
    textColor: "{colors.paper-white}"
    rounded: "{rounded.sm}"
    padding: "11px 20px"
  button-sm:
    typography: "{typography.label}"
    padding: "6px 12px"
    height: "34px"
  input:
    backgroundColor: "{colors.paper-white}"
    textColor: "{colors.ink-text}"
    rounded: "{rounded.sm}"
    padding: "9px 12px"
    height: "44px"
  answer-option:
    backgroundColor: "{colors.paper-white}"
    textColor: "{colors.ink-text}"
    typography: "{typography.headline}"
    rounded: "{rounded.sm}"
    padding: "14px 16px"
  answer-option-hover:
    backgroundColor: "{colors.action-blue-soft}"
  item-code:
    backgroundColor: "{colors.ink-rail}"
    textColor: "{colors.paper-white}"
    rounded: "{rounded.sm}"
    height: "40px"
    width: "52px"
  step-disc:
    backgroundColor: "{colors.ink-rail}"
    textColor: "{colors.paper-white}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    size: "30px"
  callout-why:
    backgroundColor: "{colors.leaflet-yellow-soft}"
    textColor: "{colors.ink-text}"
    padding: "16px 20px"
  callout-info:
    backgroundColor: "{colors.action-blue-soft}"
    textColor: "{colors.ink-text}"
    padding: "12px 16px"
  panel-inset:
    backgroundColor: "{colors.panel-grey}"
    rounded: "{rounded.md}"
    padding: "24px"
  card:
    backgroundColor: "{colors.paper-white}"
    rounded: "{rounded.md}"
    padding: "24px"
  badge:
    backgroundColor: "{colors.paper-white}"
    textColor: "{colors.muted-text}"
    typography: "{typography.label}"
    rounded: "{rounded.xs}"
    padding: "1px 8px"
  nav-link:
    backgroundColor: "{colors.ink-rail}"
    textColor: "{colors.rail-text}"
    typography: "{typography.body}"
    padding: "10px 24px"
  nav-link-active:
    backgroundColor: "{colors.ink-rail-deep}"
    textColor: "{colors.paper-white}"
---

# Design System: NeuroBridge

## Overview

**Creative North Star: "The Patient Leaflet"**

Every screen reads like a well-made public-health patient leaflet: plain words up front, the evidence one step behind. An ink-blue rail holds the navigation down the left edge; the working ground is white paper with cool grey panels. Rank is carried by weight and by printed rules (a 2px ink rule under every page head and section, an 8px rule on the reading edge of every callout), never by size creep or by lifting things off the page with shadows.

The system serves a parent reading calmly, a clinician checking reasoning and an examiner auditing evidence, in one world. Parent-facing surfaces stay spacious and plain; research surfaces use the same rules, tables and ruled figures at higher density. Nothing is hidden: the numbered pathway keeps every asked, skipped, current and still-possible question on the page, and the risk estimate always sits on the same fixed 0 to 100 scale with the referral line drawn in.

The world rejects the category default of white SaaS cards, a teal accent, a centred question card and a thin progress bar.

**Key Characteristics:**
- Ink-blue rail, white working ground, cool grey panels, one action blue.
- Ruled callouts colour-coded by meaning: yellow for why / not a diagnosis, blue for information, amber for research caveats, red for urgent, green for done.
- Four type sizes (14 / 16 / 20 / 32) in Public Sans with tabular figures everywhere.
- Filled navy numbered discs joined by a spine: the signature pathway.
- Flat content; overlays (drawer, modal, toast) alone carry a shadow.

## Colors

A cool printed-leaflet palette: navy ink and white paper, with saturated colour reserved for meaning.

### Primary
- **Leaflet Ink Blue** (ink-rail): the shell rail, the mobile bar, numbered discs, item codes, the risk-scale marker, bar-chart fills and the top rule of a modal. It is the colour of the printed furniture, not of actions. In dark theme the discs lift to a lighter navy so they stay legible.
- **Action Blue** (action-blue): the one colour that means "you can press this". Primary buttons, checked controls, the selected tab rule, the current pathway disc, link ink (via the hover shade) and the focus ring. Its soft tint (action-blue-soft) is the hover ground of answer rows and secondary buttons and the ground of information callouts.

### Secondary
- **Leaflet Yellow** (leaflet-yellow): the active-page rule and icon on the rail, the rail focus ring, the theme-toggle thumb and the rule of the "Why this question?" callout. Its soft tint and dark ink set the "screening, not a diagnosis" strip across the top of every page and the not-a-diagnosis callout on the result.

### Tertiary (status)
- **Done Green** (done-green / soft): completed sessions, granted consent, success callouts.
- **Caveat Amber** (caveat-amber / soft): research caveats, limitations, label-circularity warnings.
- **Urgent Red** (urgent-red / soft): errors, destructive actions, urgent notices.

### Neutral
- **Paper White** (paper-white): the working ground and every card or field surface.
- **Panel Grey** (panel-grey): inset panels (research settings, "Your path so far", the result block), table heads, code blocks.
- **Hover Grey** (hover-grey) and **Track Grey** (track-grey): row hover and the empty part of bars and toggles.
- **Hairline** (hairline): 1px dividers between list rows, table rows and pathway steps.
- **Rule Grey** (rule-grey): 2px control borders, unfilled answer letters, dashed skipped discs, the default callout rule.
- **Ink Text** (ink-text), **Muted Text** (muted-text), **Faint Text** (faint-text): body, secondary and placeholder text. Ink text also draws the 2px section rules and the referral line on the scale.
- **Scale Low / Scale High** (scale-low, scale-high): the two halves of the risk scale either side of the referral line, a pale blue and a pale red.

Every colour is a custom property in `frontend/src/styles/tokens.css`, defined for light and dark (`:root[data-theme='dark']`). Components reference tokens, not literals.

### Named Rules
**The Yellow Means Why Rule.** Leaflet yellow marks why a question is asked, that a result is not a diagnosis, and where you are on the rail. It is never an action colour and never a generic warning; caveats take amber.

**The One Action Blue Rule.** Pressable things are action blue; printed furniture (discs, codes, rail, bars) is ink blue. If a navy element can be clicked, it is in the wrong colour.

## Typography

**Display Font:** Public Sans Variable (bundled via @fontsource, with system-ui fallback)
**Body Font:** Public Sans Variable
**Label/Mono Font:** Public Sans for labels; the system monospace stack for raw data blocks and inline code.

**Character:** A plain, civic grotesque at heavy weights. Headings are set tight (letter-spacing around -0.02em) at 750 to 800 weight so a heading is distinct from body by weight alone.

### Hierarchy
- **Display** (800, 32px, 1.25): page titles in the page head and the result outcome title. Drops to 20px under 640px.
- **Question** (750, 32px, 1.25, max 30ch): the live screening question beside its item code. Drops to 20px under 640px.
- **Headline** (800, 20px, 1.25): card titles, section heads, callout heads, brand name; also the answer-row text (600) and the lead paragraph on the start panel.
- **Body** (400, 16px, 1.55): running text, field labels (700), nav links (500). Prose measures cap between 56ch and 72ch.
- **Label** (600 to 700, 14px): meta lines, table cells, badges, pathway step text, scale labels, the top strip. Sentence case, no tracking.

### Named Rules
**The Four Sizes Rule.** Type sizes are 14, 16, 20 and 32px. `--text-xs` and `--text-2xl` are aliases that resolve to 14 and 32. A new size is a defect; reach for weight or a rule instead.

**The Tabular Figures Rule.** `font-variant-numeric: tabular-nums` is set on the body, so every count, estimate and metric lines up in columns.

## Layout

A two-part shell: a fixed 256px ink rail on the left (sticky, full height) and a working column capped at 1280px with 40px side padding, 32px top and 48px bottom. Below 1024px the rail becomes an off-canvas drawer behind a sticky navy mobile bar with a 44px menu button; below 640px side padding tightens to 16px.

Every page begins with the full-width yellow "screening, not a diagnosis" strip, then a page head (title and optional subtitle) closed by a 2px ink rule, then content stacked on a 24px gap.

The screening view is a two-column leaflet: a fluid question column and a 400px side column (340px under 1180px, stacked under 960px) holding the why callout, the risk scale and the pathway, sticky at 24px. Result and overview splits use a fluid main column beside a 380 to 400px column. Grids collapse 4 to 2 columns under 1100px and to 1 under 760px.

Spacing runs on a 4px base (4, 8, 12, 16, 20, 24, 32, 40, 48). List rows breathe at 10 to 16px vertical; panels pad 20 to 24px; columns separate at 32 to 40px.

## Elevation & Depth

Content is flat. `--shadow-sm` and `--shadow-md` resolve to `none`; depth on the page comes from tonal panels (paper white against panel grey) and from printed rules. Overlays alone lift: the mobile drawer, the modal and the toast share one shadow and sit over a navy scrim.

### Shadow Vocabulary
- **Overlay** (`box-shadow: 0 2px 6px rgba(19, 32, 46, 0.08), 0 18px 40px -12px rgba(19, 32, 46, 0.32)`; dark: `0 2px 6px rgba(0,0,0,0.4), 0 20px 44px -12px rgba(0,0,0,0.7)`): drawer, modal, toast.
- **Focus ring** (`box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px` action blue): every focusable element on the paper. On the rail, a 3px inset yellow ring instead.
- **Halo** (`0 0 0 4px` action-blue-soft, breathing to 7px over 2.4s): the current pathway disc. A state signal, not elevation.

### Named Rules
**The Printed Page Rule.** If it lives on the page, it does not cast a shadow. Separate with a panel tint, a hairline, or a 2px ink rule.

## Shapes

Nearly square, like print. Controls, buttons, answer rows and item codes take a 6px radius; badges and checkboxes 3px; panels and cards 6px; the modal 8px. Circles are reserved for numbered discs, answer letters, the scale marker and toggle thumbs; pills for toggle tracks and the scrollbar thumb. Bars and the risk scale have a 2px corner.

Borders are structural: 2px for controls and section rules (`--rule`), 8px on the reading edge of callouts, toasts and the modal top (`--rule-callout`), 1px hairlines between rows. Skipped and not-yet-asked pathway discs switch to a 2px dashed outline; the end-of-path disc is a hollow ring with a 4px ink border.

## Components

### Buttons
Solid, square-shouldered and weighty.
- **Shape:** gently squared (6px), 2px border always present.
- **Primary:** action blue fill, white text, 700 weight, 11px 20px padding, 44px minimum height. Hover deepens to action-blue-hover.
- **Secondary:** paper fill, action-blue border and ink; hover tints to action-blue-soft.
- **Ghost:** transparent, underlined ink text; hover thickens the underline to 2px and adds a hover-grey ground.
- **Danger:** urgent red fill, white text.
- **Small:** 14px text, 6px 12px padding, 34px minimum height.
- **Press / Disabled:** a 1px downward nudge on press; 50% opacity when disabled.

### Answer Rows
Full-width lettered rows that answer on one tap. A 32px circular letter on the left, 20px 600-weight text, 2px rule-grey border, 6px corner, 14px 16px padding, max 640px wide. Hover turns border and letter action blue over the soft blue ground; press fills the letter solid blue.

### Callouts (Notice)
- **Style:** 8px left rule, tinted ground, 14px text with a leading 16px line icon, no radius.
- **Tones:** neutral (rule grey on panel grey), info (action blue), done (green), caveat (amber), urgent (red).
- **Why this question:** the same form at 20px headline, 16px body, yellow rule on yellow soft.

### Cards / Containers
- **Corner Style:** 6px.
- **Background:** paper white with a 1px hairline, or panel grey with no border (inset).
- **Shadow Strategy:** none (see Elevation).
- **Internal Padding:** 24px; head row 20px above body.
- **Ruled figure (metric):** no box at all; a 2px ink rule on top, 14px label, 32px 800-weight value.

### Inputs / Fields
- **Style:** 2px rule-grey border, 6px corner, paper fill, 44px minimum height, 9px 12px padding. Labels sit above at 16px / 700, help text below at 14px muted.
- **Hover / Focus:** border darkens to muted text on hover; on focus the border turns action blue and the focus ring appears.
- **Select / Search:** custom chevron and search glyph as inline line icons; a 28px square clear button.
- **Checkbox / Toggle:** 22px square box (3px corner) filling action blue when checked; 44 by 26px pill toggle with a sliding 18px thumb.

### Badges
Small bordered labels: 3px corner, 1px border, 14px / 700, 1px 8px padding. Tones mirror the callouts.

### Navigation
- **Rail:** ink-blue ground, 20px / 800 white brand name with a 14px muted subtitle, group labels at 14px / 600 rail-muted in sentence case.
- **Links:** 16px / 500, 10px vertical padding, a 16px line icon in rail-muted. Hover deepens to ink-rail-deep.
- **Active:** ink-rail-deep ground, 5px yellow left rule, yellow icon, white 700 text.
- **Mobile:** sticky navy bar with brand and a 44px menu button; the rail slides in as a 300px drawer over the scrim.
- **Tabs:** 16px / 600 muted text on a 2px hairline baseline; selected tab goes 800 ink with a 4px action-blue underline.

### Numbered Pathway (signature)
The spine of the product, reused for the live path, the result trace, Explanations and the RL Lab replay. An ordered list of rows, each a 30px filled navy disc with a white number, the item code (14px / 800 muted) and question text, and a right-aligned answer with the running estimate. A 2px spine at 45% rule-grey joins the discs; 1px hairlines divide the rows.
- **Current:** disc turns action blue with a breathing soft-blue halo; answer reads "Now".
- **Skipped:** dashed hollow disc with an en dash, struck-through muted title, "Not answered".
- **Remaining / Not needed:** collapsed rows with a dashed hollow disc stating how many questions they stand for.
- **End:** hollow disc with a 4px navy ring and an 800-weight end label.

### Item Code
The Q-CHAT item code set as a filled navy tile (52 by 40px, 6px corner, 16px / 800 white) beside the question, so the instrument reference travels with every question.

### Risk Scale (signature)
One fixed 0 to 100 scale on every screen. A 32px / 800 value with "/ 100", a 14px state line ("Below the referral line" / "At or above the referral line"), then a 14px bar split at the referral line into scale-low and scale-high, a 2px ink referral line overhanging the bar by 7px, and a 22px navy marker with a white 3px ring. Labels 0, "Referral line n" and 100 sit beneath. The scale never rescales or recolours by result.

### Motion
One easing curve, `cubic-bezier(0.16, 1, 0.3, 1)`, at 120ms (hover, colour), 220ms (theme, drawer, modal) and 420ms (scale marker travel, question settle). New questions and results settle in with a 6px rise and fade. Reduced-motion collapses all of it.

## Do's and Don'ts

### Do:
- **Do** take every colour from `tokens.css` custom properties and define both light and dark values for any new token.
- **Do** separate sections with a 2px ink rule (`--rule`) and give every callout an 8px reading-edge rule (`--rule-callout`).
- **Do** keep type on the four sizes (14 / 16 / 20 / 32) and carry rank with 700 to 800 weight.
- **Do** show a sequence of questions or decisions as the numbered pathway, and any risk estimate on the fixed 0 to 100 risk scale with the referral line drawn.
- **Do** keep skipped and not-needed steps visible in the pathway, marked, never removed.
- **Do** carry the yellow "screening, not a diagnosis" strip across the top of every page.
- **Do** give every interactive control a 44px minimum target and the 2px + 4px focus ring.

### Don't:
- **Don't** cast shadows on content; overlays (drawer, modal, toast) are the sole exception.
- **Don't** use leaflet yellow as an action colour or a generic warning; caveats are amber, actions are action blue.
- **Don't** use teal, gradients on content, or a centred floating question card with a thin progress bar.
- **Don't** introduce uppercase tracked labels or eyebrows above headings; group labels and meta stay sentence case at 14px.
- **Don't** rescale or recolour the risk scale per result, or present an estimate without its referral line.
- **Don't** round past 8px on containers; the leaflet is printed, not pillowy.
