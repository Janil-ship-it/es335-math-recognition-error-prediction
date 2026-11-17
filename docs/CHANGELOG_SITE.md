# Site redesign changelog

## What moved where

| Old location | New location |
|--------------|--------------|
| index.html monolithic sections (question, data, models, evaluation, results, budget, errors, limits) | Split into step pages; home is overview only |
| Region explorer on explore.html | errors.html |
| Review-budget lab mixed with explorer | explore.html (lab only) |
| Inline CSS / divergent tokens on index | Single `style.css` design system |
| Hardcoded metrics in HTML/JS | `app.js` fetches `data/site_results.json` |

## New files

- `problem.html` — research question, unit definition, forbidden features
- `errors.html` — region explorer + filters
- `app.js` — shared nav, progress, data loading, charts helpers
- `CHANGELOG_SITE.md` — this file

## Architecture choice

**No Jinja build step.** Shared chrome (nav, progress, prev/next) is injected by `app.js` so GitHub Pages stays zero-build: edit HTML, commit, deploy from `/docs`. A Python/Jinja pipeline (`site_src` + `build_site.py`) would reduce HTML duplication further but adds a required build before every publish; deferred unless the team prefers it.

## Checklist

- [x] No page-local `<style>` blocks (only `style.css`)
- [x] No hardcoded experimental metrics in HTML/JS (loaded from JSON)
- [x] Nav order: Home → Problem → Data → Methods → Results → Review lab → Errors → Limits → Reproduce
- [x] Step progress + Previous/Next on steps 1–7
- [x] Persistent “Synthetic development data” badge
- [x] CI-includes-zero stated on home and results
- [x] Review lab uses only measured 20/30/50% rates (chips, no interpolation)
- [x] Relative links only
