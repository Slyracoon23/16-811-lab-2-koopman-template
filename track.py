"""Experiment tracking, through Weights & Biases.

Every question in this lab is a sweep — which dictionary, how many basis functions, how much data
— and a sweep whose results live in your terminal scrollback is a sweep you will run twice. W&B is
what the job uses for this, so it is what the lab uses.

**Offline by default, so nothing blocks.** `wandb` runs with no account and writes to `./wandb/`;
`wandb sync` uploads later if you want it, and `WANDB_MODE=online` with `wandb login` goes straight
there. A missing or broken tracker degrades to printing, because the hardest dependency must never
be the reason the lab does not run.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")


@contextlib.contextmanager
def run(name: str, config: dict[str, Any] | None = None):
    """Yield something with `.log(dict)`. A W&B run where it works, a printer where it does not."""
    try:
        import wandb
    except ImportError:
        yield _Printer(name)
        return

    try:
        handle = wandb.init(project="16-811-lab-2-koopman", name=name, config=config or {}, reinit=True)
    except Exception as failure:  # a broken tracker must not take the lab down with it
        print(f"[track] wandb unavailable ({failure}); printing instead")
        yield _Printer(name)
        return

    try:
        yield handle
    finally:
        handle.finish()


class _Printer:
    def __init__(self, name: str) -> None:
        self.name = name

    def log(self, values: dict[str, Any]) -> None:
        body = "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items())
        print(f"[{self.name}] {body}")
