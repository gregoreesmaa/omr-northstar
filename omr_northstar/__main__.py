"""Allow running `python -m omr_northstar`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
