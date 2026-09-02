"""The half that is yours: choose the dictionary by conditioning, not by size.

Switches, not branches — a reader should be able to turn each idea off from the command line and
watch the number move. And it has to be the *same* number `reproduce.py` reports.
"""

from __future__ import annotations

import argparse
import json

VARIANTS = ["identity", "rbf", "polynomial", "delay"]


def run(variant: str, seed: int) -> float:
    """Fit with this dictionary on the pendulum and return the multi-step prediction error."""
    raise NotImplementedError(f"Your idea: {variant}. See 'Past the paper' on the lab sheet.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    chosen = VARIANTS if args.all else [args.variant or "identity"]
    records = [
        {"variant": v, "seed": s, "pred_err": run(v, s)} for v in chosen for s in range(args.seeds)
    ]
    with open("extensions.json", "w") as handle:
        json.dump({"records": records}, handle, indent=2)
    print(f"{len(records)} records into extensions.json")


if __name__ == "__main__":
    main()
