# Brief: Lead / Reviewer

The session that coordinates the team. It works on the default branch
(`claude/stoic-davinci-v4p6k9`), which deploys the app.

## Responsibilities

- **Review every team PR:**
  - read the diff;
  - run `python -m pytest -q tests`;
  - for model PRs, re-run or spot-check the gate numbers;
  - for app PRs, take the screenshots;
  - check that the PR stays in its role's files.
- **Merge only with the owner's go-ahead**, one PR at a time, and keep the deploy green
  (watch the next "Publish app" run).
- **Merge order:**
  - model PRs (Strikeouts, Hitters) before Edge refits the blend;
  - after a model merges, ask Edge to refit;
  - UI PRs whenever they're ready, since they don't conflict with model work.
- **Docs:** keep `docs/WORK_PLAN.md` and `CLAUDE.md` current with each merge. Fold the
  roles' write-ups into the plan.
- **Credits:** watch the shared Odds API balance in the publish logs. Stop Edge's spending
  if the balance nears 25,000.
- **Sessions:** the team sessions are tagged `baseball-team`. Their IDs are listed below.

## Team sessions

| Role | Session | Branch |
|---|---|---|
| Edge | (filled in at start) | `team/edge` |
| Strikeouts | (filled in at start) | `team/strikeouts` |
| Hitters | (filled in at start) | `team/hitters` |
| UI Design | (filled in at start) | `team/ui` |
