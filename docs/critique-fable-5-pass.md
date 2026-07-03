# Design Critique — Adaptive Capacity Explorer (Fable 5 pass)

_Re-run of the critique pass on a newer frontier model. Four independent critics
were run in parallel — three on **Fable 5** (visual/aesthetic, voice/microcopy,
data-visualization) and one general-model pass on accessibility/interaction —
each reading the real code plus rendered screenshots (light + dark, desktop +
mobile) captured against the live app with a locally-vendored Chart.js._

## Method note & one important correction

Findings were grounded in **rendered** screenshots, not just code. That surfaced
one thing worth stating up front, because three of the four critics initially
led with it:

> **The charts are _not_ broken in normal use.** In full-page screenshot capture
> the scatter and the component bars appear collapsed into a blob at the plot
> origin. Re-capturing each chart in a **fixed viewport** (and reproducing the
> app headlessly against the real 741-occupation dataset) shows both render
> **correctly** — a well-spread bubble chart and clean percentile bars.

What the blob actually reveals is a **real but situational fragility** (see
FIX-3): charts are created synchronously in the same task that unhides the
section and starts a smooth-scroll, with Chart.js's default ~1s entry animation
and a `max-width:100%` canvas. On a slow first paint, during screenshot capture,
or in link-preview renderers, the elements animate from a stale near-origin
measurement while the axes are already at full size — producing the blob. It's a
~10-line fix and worth doing (this tool's centerpiece is a chart, and the blob
appears in exactly the contexts used for sharing), but it is **not** a data or
axis-domain bug and does not affect the normal interactive experience.

---

## Priority 0 — Accessibility & correctness (ship these first)

### FIX-1 · Text contrast fails WCAG AA in several places (computed twice, independently)
Normal-weight/AA (4.5:1) failures, measured from the actual tokens:

| Element | Foreground / background | Ratio | Where |
|---|---|---|---|
| `.indicator-warning` text | `#d97706` on `#fef3c7` | **~2.9:1** | `css/style.css:718` |
| `.indicator-danger` text | `#dc2626` on `#fee2e2` | **~3.98:1** | `css/style.css:712` |
| Neighbor "AI exposure" amber | `#d97706` on card `#f7f6f4` | **~2.96:1** | `js/app.js:1496` |
| **Dark-mode** primary button/tab/roadmap badge | `#fff` on `#818cf8` | **~2.9:1** | `style.css:1264` + `.btn-primary`/`.tab.active`/`.roadmap-item::before` |
| Scatter "Lowest vulnerability" bubbles | `rgba(189,210,245,.55)` on warm white | **~1.2:1** | `SCATTER_CONFIG.tiers`, `app.js:81` |

**Fix:** darken the ink used _as text on tint_ (warning → `#92400e`/`#b45309`,
danger → `#b91c1c`); the colored left border already carries the semantic. In
dark mode, pair the `#818cf8` fill with dark ink (`#1e1b4b`) or keep a
`#4f46e5`/`#6366f1` fill with white text. Raise the lightest scatter tier's
opacity/border to ≥3:1.

### FIX-2 · Charts have no text alternative — screen readers get nothing
Both `<canvas>` (`index.html:63`, `:272`) are bare — no `role`, `aria-label`, or
adjacent equivalent (WCAG 1.1.1). The scatter (the whole concept explainer) and
the component percentiles (a primary result) are invisible to AT.
**Fix:** `role="img"` + concise `aria-label` on each canvas, plus a
visually-hidden `<table>`/`<dl>` of the underlying values next to each (the
component percentiles are already in `state`).

### FIX-3 · Chart first-paint race + no reduced-motion for Chart.js
Charts are instantiated synchronously while the section is unhiding and
scrolling (`app.js:1818–1826`, `2146–2149`); Chart.js's default ~1000ms
animation is never disabled, and `prefers-reduced-motion` is honored in CSS but
**not** by Chart.js. This is the cause of the capture blob and a genuine
reduced-motion gap.
**Fix:** create charts after layout settles (double `requestAnimationFrame` or
`transitionend`); set `animation:false` (or `{duration:300}`), and
`if (matchMedia('(prefers-reduced-motion: reduce)').matches) Chart.defaults.animation = false;`
Remove `max-width:100%` from the canvas (`style.css:284`), give
`.chart-container` an explicit height, and use `maintainAspectRatio:false`.

### FIX-4 · Programmatic smooth-scroll ignores `prefers-reduced-motion`
`scrollIntoView({behavior:'smooth'})` at `app.js:1815, 1822, 2065, 2138, 2147,
2205` overrides the CSS `scroll-behavior:auto` reduced-motion rule (WCAG 2.3.3).
**Fix:** compute `reduce = matchMedia('(prefers-reduced-motion: reduce)').matches`
and pass `behavior: reduce ? 'auto' : 'smooth'` at all six sites.

### FIX-5 · Combobox pattern is visual-only for AT
Search/metro typeaheads declare `role="combobox"`/`listbox`/`option` but keyboard
highlight only mutates inline `style.background` (`app.js:673, 826`). No
`aria-activedescendant`, no option `id`s, `aria-selected` never toggled, missing
`aria-autocomplete="list"`; the listbox also stays open (and `aria-expanded=true`)
when Tab moves focus away.
**Fix:** id each `<li>`, toggle `aria-selected`, set `aria-activedescendant` on
the input during keyboard nav, add `aria-autocomplete="list"`, and close on
`blur`/`focusout`.

### FIX-6 · Focus is dropped on step/results transitions
`showResults()` and `goToStep()` hide the currently-focused control without
moving focus and never announce the change (WCAG 2.4.3 / 4.1.3). Keyboard/SR
users are stranded on `<body>`.
**Fix:** move focus to the new step's `<h3>` / results `<h2>` (`tabindex="-1"`),
or expose a polite live-region status ("Your results are ready").

_Also in this tier (lower drama, still real):_ quiz `role="radio"` group lacks a
linked name + roving-tabindex/arrow keys (`app.js:527`); `role="progressbar"`
has no `aria-label`/`aria-valuetext` (`index.html:88`); age/runway validation
errors aren't announced or `aria-describedby`-linked (`app.js:1964`); `#explainer`
and `#results` sit outside any landmark (`<main>` wraps only the assessment);
quiz auto-advance `setTimeout(400)` is a hostile context change for AT
(`app.js:426`).

---

## Priority 1 — Brand fidelity & trust (the heart of the critique)

### FIX-7 · The "warm studio" palette is imperceptible; the page reads white-then-black
The warm tokens exist (`--color-bg:#f7f6f4`, cards `#fefefe`) but the delta is so
small the page reads as clinical white with 1px borders doing all the separation
work — the "elevation through outlines" the guidelines explicitly reject. The
hero sits on `#fefefe` and the footer is `#1a1917`, so the opening viewport is
literally a white half above a black half: the harshest possible temperature
statement, zero warmth.
**Fix:** deepen the page background a step (`~#f3f1ec`) so `#fefefe` cards lift
off it; give the hero the warm bg instead of white; warm the shadows
(`rgba(45,35,20,.07)` instead of pure-black alpha); in light mode make the footer
slate `#1e293b` with warm text `#f1f0ee` (and `var(--color-surface)` in dark) so
it participates in the system.

### FIX-8 · The hero reads as generic bold-black-on-white SaaS
A five-line centered near-black 700 headline with a `next?` widow, gray subtitle,
two indigo pills — nothing signals "knowledgeable friend." The `36rem` measure +
`--text-4xl` yields ragged breaks.
**Fix:** `text-wrap:balance` on the `h1`; tighten measure (~22ch) or
`clamp(2.25rem,5vw,3rem)` with `letter-spacing:-.02em`; set heading ink to warm
slate (`#292524`) not max-contrast `#1a1917`; consider one indigo-accented phrase
to tie the headline to the brand.

### FIX-9 · The sourcing story contradicts itself — and over-claims
`index.html:35–37` introduces "Manning & Aguirre," then attributes the
measurement to "**Researchers at Brookings**" (Brookings only published a
_summary_; the citation is NBER WP 34705). For an audience that "respects
transparency about methodology," this reads like two studies and undercuts
trust. "This **unofficial** tool" is a disclaimer word ("someone might object");
the footer's "a research translation project" is honest without being
apologetic. Separately, runway/age band copy invokes "**The research says…**" /
"**The data says…**" (`app.js:29–30`) to authorize the tool's _own_ 3/6/12/18
thresholds, which the paper doesn't bless — exactly the over-claiming the footer
disavows.
**Fix:** name the authors + NBER, note Brookings as the summary, replace
"unofficial" with "independent," and lead with the strongest idea (occupation →
you). Reserve "the research says" for claims the paper actually makes.

### FIX-10 · Traffic-light red/green as the sole signal (bars + neighbor list)
Component bars encode the strong/middling/weak judgment _only_ by hue
(`app.js:1197`); `#047857` and `#dc2626` are near-isoluminant — under
deuteranopia/protanopia they're the same bar (WCAG 1.4.1, and the brand's own
"HR software" anti-reference). The neighbor list then repeats "AI exposure: 93%…
88%…" in danger-red down all ten cards — a wall of red after a "solid
foundation" message: fear styling, not "informed and motivated."
**Fix:** render bars in the indigo family with a 50th-percentile (and optional
benchmark) reference line + a short text tag per bar ("strength"/"focus area");
set neighbor exposure in secondary text with the numeral in 600 weight, coloring
only genuinely exceptional values, and add a word ("high"/"moderate"/"lower").

---

## Priority 2 — Results storytelling & copy quality

- **FIX-11 · "57 of 100" fights the percentile model** (flagged by two critics).
  The ring says "57 **of 100**" (a score) while the prose says "57th
  **percentile**" (a rank); users read the ring as a graded test ("I got a 57").
  Standardize on "57th percentile" in the ring (`app.js:1345`).
- **FIX-12 · The benchmark sentence is the hardest sentence at the most important
  moment** (`app.js:1363`): two parentheticals + a dangling "which," percentiles
  subtracted as "points," no interpretation. Rewrite plainly and **promote the
  benchmark into the ring** as a tick mark at 79 — the single most actionable
  comparison, currently buried in small gray text.
- **FIX-13 · Tier & band copy** (`app.js`): "**Early Stages**" is a euphemism that
  misfires on the older/thin-savings users most likely to receive it (and the
  body contradicts the label) → e.g. "Headwinds"; "**Solid Foundation**"
  under-sells the 50–75th percentile with a limp double-hedge; the "**the
  research is honest/doesn't sugarcoat**" device repeats verbatim on Step 4 _and_
  in results for older users — exposing the template. Comma-splices are a house
  style ("You're not starting from zero, there are real things…") — use the em
  dash the copy already wields well. The clipboard export (`buildResultsText`,
  `app.js:1291`) reverts the UI's thoughtful "Factor to plan around" back to blunt
  "Weakest" — the _most shareable_ artifact has the least considered voice.
- **FIX-14 · Step framing** — intros restate the "Why does this matter?" hints in
  academic register, so disclosure rewards a click with a paraphrase; and step
  titles alternate question ("What do you do?") vs noun ("Financial runway").
  Make intros do only the instruction; make all four titles questions (or none).
- **Smaller copy polish:** hero headline ("dramatically… tomorrow") opens in the
  doom register the subtitle immediately contradicts; primary CTA leans on the
  jargon the secondary CTA exists to explain (and never sets a "4 steps, ~3 min"
  expectation before asking for financial details); "shots on goal,"
  "speciality" (British), Title-vs-sentence-case inconsistency, `...` vs `…`.

## Priority 2 — Scatter refinements (renders fine, but)

- **FIX-15 · The user's amber "you are here" dot is drawn _under_ all 741
  bubbles.** Chart.js draws datasets in reverse index order; the highlight is
  `push`ed last, so it renders at the bottom of the z-stack and can be occluded
  in dense regions (`app.js:1038`). Give it `order:-1`, keep an
  employment-scaled radius with a thick amber ring, and label it "You — …".
- **FIX-16 · Zero label-collision handling + clumsy truncation** for a
  _hand-picked_ set (`app.js:1124`): "Marketing managers"/"Lawyers" collide;
  "Heavy and tractor-trail…" truncates. Add per-SOC short display names + offset
  overrides (only 12 SOCs) and a one-pass greedy vertical nudge.
- **FIX-17 · The quadrant framing is the chart's best idea but never explained** —
  just "Median" twice, and the x-axis "Median" collides with the `0.30` tick
  (`app.js:1106`). Name the quadrants in muted corner text; move the median
  labels inside the plot, clear of ticks.
- **FIX-18 · Encoding/legend mismatches:** ~53% of bubbles sit at the 2.5px size
  floor (size≈fiction for half the data); the static HTML size-legend circles are
  ~25–30% smaller than the marks they describe; legend color swatches use
  different alphas than the plotted fills. Generate legends from `SCATTER_CONFIG`
  (one source of truth); lower the floor.

---

## Priority 3 — Code hygiene (cheap, do alongside the above)

- `.whatif-metro-wrap` input has `class="input"` but no `.input` rule exists →
  renders as an unstyled white UA input, broken in dark mode (`index.html:255`,
  `style.css:1144`); its listbox references a nonexistent `.search-results` class.
- `.dimension-hint-body` uses `var(--radius-sm)` (`style.css:665`) — that token is
  never defined (only `--border-radius`/`-lg` exist) → radius silently 0.
- `.btn-link` is defined **twice** with conflicting `display`/`color`/`:hover`
  (`style.css:796` and `1183`); the later wins and partially overrides the "Start
  over" link's intended styling. Consolidate.
- `.search-item:focus` style never applies (items aren't focusable) — dead rule
  or unfinished pattern (`style.css:610`); prefer `aria-activedescendant` (FIX-5).
- Prose measure runs ~85–95 ch at `--container-max:48rem` (principle #1 is
  "legibility first") — cap prose blocks at `~68ch` while charts/cards keep the
  wider container.
- Mobile: "Why does this matter?" and the step-nav links are sub-44px targets;
  let `.step-nav` wrap on narrow screens.

---

## What's working (keep)

- The **privacy notice**, placed exactly where financial input is requested.
- The **footer caveats** ("not a prediction… financial advice… career
  counseling") and the What-If **methodology note** — genuine model transparency.
- Renaming "Weakest" → **"Factor to plan around"** when age is the low dimension.
- The **best sentences** already hit the target voice — "Enough to be somewhat
  selective, not enough to be patient," "a transition is a strategic choice, not
  an emergency response," "the clock ticks louder," "roles that are available
  rather than roles that are right." Most copy fixes are about raising the rest to
  this bar.
- The **bones**: spacing scale, radius consistency, weight contrast, visible
  `:focus-visible` states, correct heading order, `hidden` sections removed from
  AT, the score circle's value exposed as real text. This is a calibration/QA
  pass, not a rebuild.
</content>
