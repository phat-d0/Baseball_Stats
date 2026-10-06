# Brief: UI Design

You own the iPhone web app's look and usability. The owner uses it on an iPhone, at the
ballpark or on the couch, to decide on DraftKings bets.
Branch: `team/ui`. Read `CLAUDE.md` and `docs/WORK_PLAN.md` first, then use the app with
real data: build a `data.json` locally with
`python -m baseball_stats publish --out <scratch>/site` and serve that folder.

## You own

- `web/` (`index.html`, `app.js`, `style.css`, `sw.js`, manifest, icons via
  `scripts/make_icons.py`)

Don't change `data.json`'s shape. If you need a new field, ask the Lead in your PR; the
backend is shared.

## How your work is judged

- **Screenshots** at 390×844 in light and dark mode for every changed screen (Playwright;
  see `CLAUDE.md`). Include them in the PR description (attach or describe), with before
  and after.
- **No page errors, no horizontal overflow.** Text never truncates important numbers.
  The app also works offline.
- **Bump `CACHE` in `web/sw.js`** whenever `app.js` or `style.css` change.
- **Honest presentation.** Edges are small now (mostly 1–4%). Don't make them look bigger
  or more certain than they are. Keep the notes that explain what a number means.

## First tasks

1. **Usability audit.** Go through every tab and sheet (Games, game sheet, Hitters,
   Pitchers, player sheet, Portfolio, Record) at phone size. List problems: crowding,
   unclear labels, things that need too many taps, numbers that are hard to compare with
   the DraftKings app. Fix the clear ones; list the rest in your PR.
2. **Edge view.** The owner compares the app to the DraftKings app (which shows implied
   %). Make it obvious, per prop:
   - DraftKings' price and implied %,
   - our blended chance,
   - the edge, and whether a paper trade was placed.
3. **Record tab.** It has grown: market card, shadow lines, calibration. Make it scannable,
   with the most important number first: are we beating DraftKings (CLV, return,
   blend vs DraftKings log loss)?
4. **Portfolio.** Strategy comparison, open vs settled, and the profit chart should read at
   a glance.
