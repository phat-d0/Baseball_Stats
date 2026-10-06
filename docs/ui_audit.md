# UI audit and edge view (UI Design, Oct 6, 2026)

I went through every tab and sheet at 390×844 in light and dark mode. The data was a
local `publish` of Oct 6 (2 postseason games today, 4 tomorrow). Since there's no Odds
API key locally, I filled in today's slate from the latest DraftKings prices in the
price log (17 players priced, fetched 17:41 UTC). Screenshots are in `docs/ui/edge-view/`.
No page errors and no horizontal overflow, before or after.

## Fixed in this change

| Where | Problem | Fix |
|---|---|---|
| Value picks, game-sheet edges | Couldn't be compared with the DraftKings app: the price had no implied %, "chance 49%" didn't say whose chance, and the edge rounded to whole % (+2.1% showed as "+2%") | Each row now has three boxes: **DraftKings** price + implied % (as DK's app shows it), **Our chance** (the blend), **Edge**, to one decimal |
| Value picks | No sign of whether a paper trade was placed | A "Paper $10 · open/won/lost" pill, as the game sheet already had |
| Player sheet → DraftKings table | 7 columns, over and under prices stacked in one cell, with DK's chance given only with the margin removed, so it didn't match DK's app | One row per bet (O 4.5, U 4.5): DK price, DK %, Ours, Edge. Rows at or above the minimum edge are highlighted, and a row that got a paper trade carries the pill. The model's own chance stays, in the note under the table |
| Badges ("✓ Over 4.5 +123 · +9%") | Unclear what the % was | "Over 4.5 at +123 · edge +9.1%" |
| Portfolio | Opened on the retired "Model 12%+" strategy for anyone who never tapped the switch (old default `edge12`) | Opens on the live blended strategy; the saved choice moves to a new storage key so it resets once |
| Portfolio rules note | Said "model edge" for every strategy | Says "our blended chance", or "the model's own chance (retired Oct 6…)" for the retired ones |
| Portfolio comparison table | "−$21.61" broke across two lines | Numbers in table cells don't wrap |
| Game card / sheet | Weather showed "nan" when wind was missing (Petco Park) | Hidden |
| Pitcher "Why" grid | "Umpire K effect −0%" | "none" when it rounds to 0 |

## Still open (proposed for later PRs)

1. **Record tab (task 3).** "Are we beating DraftKings?" should come first, with CLV,
   return and its range, and log loss. The log loss line compares the *model alone* with
   DraftKings, but edges now use the blend. **Backend request:** add `logloss_blend` to
   `record.market.<kind>` (closing rows, `p_blend`) so the app can show blend vs
   DraftKings. Until then I'll label that line "model alone".
2. **Portfolio (task 4).** With nothing settled, the tiles read "Profit +$0.00" and
   "Return –". Retired strategies take up two of the three switch buttons and wrap. I plan
   a compact strategy list (live first, retired muted), open/settled counts in the
   headings, and a profit chart with a labeled zero line.
3. **Hitter and pitcher rows show the model's own chance** as the big number (e.g.
   Pivetta "model over 4.5: 66%" while our blended chance is 49%). It's labeled honestly,
   but it's the less accurate number. Proposal: when a DraftKings line exists, show "ours"
   (the blend) big and the model small.
4. **Hitter rows mix lines.** The list is sorted by and shows over 1.5, but the DK meta
   can show the 0.5 line (Yelich, Pratt). Show DK's price at the selected line, or say
   when DK only offers another line.
5. **Pitcher sheet** shows the over/under table *and* the DraftKings table, with three
   different chances. Consider folding the "Fair" table into a collapsible section.
6. **Game sheet title** wraps to two lines on long team names ("Milwaukee Brewers @ San
   Diego Padres"); abbreviations would keep the sheet's top compact.
