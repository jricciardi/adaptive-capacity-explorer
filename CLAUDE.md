# Adaptive Capacity Tool — Claude Guidelines

## Design Context

### Users
Curious professionals proactively exploring their career positioning in an AI-driven labor market. They arrive with genuine interest and mild uncertainty; the tool should leave them feeling **informed and motivated** — with a clearer picture of their strengths and a sense of concrete next steps. They're smart, not specialists, and they respect transparency about methodology.

### Brand Personality
**Warm, grounded, practical.**

Not a cold enterprise dashboard. Not a slick startup product. A knowledgeable friend who happens to have done the research — thoughtful, direct, honest about what the data can and can't tell you. The tone is accessible without being dumbed down.

### Aesthetic Direction
- **Color:** Shift primary from corporate `#1a56db` blue toward **indigo-600 / slate-700** territory. Warmer neutral backgrounds (off-white → warm gray `#f8f7f5` or similar). Less "IT department", more "thoughtful studio".
- **Typography:** System fonts only (no external loading). Lean into `system-ui` / `-apple-system`. Push for stronger weight contrast between headings and body — bold `700` headers, comfortable `400` body. No decorative typefaces.
- **Surfaces:** Warm white cards, borders stay subtle. Elevation through shadow, not heavy outlines.
- **Data visualization:** Keep the blue→red vulnerability gradient (it maps to the Brookings research). The amber highlight for the user's position is a feature, not a brand color.
- **Dark mode:** Support `prefers-color-scheme: dark`. Dark slate backgrounds (`#0f172a` / `#1e293b`), warm text (`#f1f0ee`), muted borders.
- **Anti-references:** Avoid generic SaaS blue, clinical white + gray dashboards, anything that feels like an HR software product.

### Design Principles
1. **Legibility first.** Every design decision serves reading comprehension — line length, contrast, hierarchy. If it looks good but reads poorly, it's wrong.
2. **Earned warmth.** Warmth comes from considered choices (color temperature, weight contrast, breathing room) — not from decorative flourishes or rounded-everything.
3. **Data deserves respect.** Visualizations are primary content, not embellishments. Give charts space, label them clearly, never style over substance.
4. **Accessible by default.** WCAG AA minimum. Focus states visible. Color never the sole signal. Reduced motion respected.
5. **Quiet confidence.** The interface doesn't need to shout. Restraint, consistency, and precision communicate credibility better than decoration.

### Tech Constraints
- Vanilla JS + HTML/CSS (no framework, no build step)
- System fonts only — `system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif`
- Chart.js 4.x for visualizations
- WCAG AA required
- Dark mode via `prefers-color-scheme` media query
