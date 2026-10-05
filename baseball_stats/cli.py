"""Command line entry point: ``python -m baseball_stats <command>``."""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

from . import collect, config, features, mlb_api, publish, slate, storage


def _date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="baseball_stats", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="download boxscores (+ optional Statcast) for a date range")
    c.add_argument("--start", type=_date, required=True)
    c.add_argument("--end", type=_date, default=date.today() - timedelta(days=1))
    c.add_argument("--game-type", default=mlb_api.ALL_GAME_TYPES,
                   help="comma-separated: R regular season, F,D,L,W postseason rounds")
    c.add_argument("--statcast", action="store_true", help="also fetch Statcast pitch data (slow)")

    sub.add_parser("features", help="build model training tables from processed data")

    p = sub.add_parser("publish", help="update data, train models, write the phone app to --out")
    p.add_argument("--out", default="_site")
    p.add_argument("--history-start", type=_date, default=None,
                   help="first date to collect on a fresh run (default: March 1, two seasons back)")
    p.add_argument("--statcast", action="store_true",
                   help="also keep Statcast pitch data up to date (first run downloads all of it)")

    s = sub.add_parser("slate", help="feature rows for upcoming games on a date")
    s.add_argument("--date", type=_date, default=date.today())

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "collect":
        collect.collect_games(args.start, args.end, game_type=args.game_type)
        if args.statcast:
            collect.collect_statcast(args.start, args.end, game_type=args.game_type)
    elif args.cmd == "features":
        out = features.build_features(features.load_inputs())
        for name, df in out.items():
            path = config.FEATURES_DIR / f"{name}.parquet"
            storage.write(df, path)
            logging.info("wrote %s: %d rows x %d cols", path, *df.shape)
    elif args.cmd == "publish":
        publish.publish(args.out, history_start=args.history_start, statcast=args.statcast)
    elif args.cmd == "slate":
        out = slate.build_slate(args.date)
        for name, df in out.items():
            path = config.FEATURES_DIR / "slates" / f"{name}_{args.date}.parquet"
            storage.write(df, path)
            logging.info("wrote %s: %d rows", path, len(df))


if __name__ == "__main__":
    main()
