"""Module entry point enabling `python -m trace2fold ...`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
