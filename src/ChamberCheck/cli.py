"""CLI entrypoint for the chambercheck package."""

import argparse

from chambercheck import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chambercheck")
    parser.add_argument("--version", action="store_true", help="Print package version and exit")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        print(__version__)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
