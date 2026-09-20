"""jevlocal — compatibility shim. The package was renamed to `luce` in 0.2.0; this alias will be removed in 0.3."""

from __future__ import annotations

import importlib
import sys
import warnings

import luce as _luce

warnings.warn("`jevlocal` was renamed to `luce`; update imports (this shim will be removed in 0.3).", DeprecationWarning, stacklevel=2)

__version__ = _luce.__version__
for _mod in ("core", "data", "model", "decision", "metrics", "train", "eval", "eval_logprob", "eval_dump", "server"):
    sys.modules[f"jevlocal.{_mod}"] = importlib.import_module(f"luce.{_mod}")


def __getattr__(name):
    return getattr(_luce, name)
