"""Backward-compatible wrapper: `python convert.py ...` == `luce convert ...`."""
from luce.convert import main

if __name__ == "__main__":
    main()
