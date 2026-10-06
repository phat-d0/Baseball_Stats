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
| Lead | `session_01KJq9DXdy6f5BygSys7RpVc` | `claude/stoic-davinci-v4p6k9` |
| Edge | `session_01CfMzWx5KSxECoT7kRa9292` | `team/edge` |
| Strikeouts | `session_01CWbLsGVwBUJmyqW4hV4su2` | `team/strikeouts` |
| Hitters | `session_011LnD1SKBMiJZonm2Srk2sH` | `team/hitters` |
| UI Design | `session_015vKbR4Xd31AW4cDA14bXax` | `team/ui` |
