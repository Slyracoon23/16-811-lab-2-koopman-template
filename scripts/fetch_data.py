"""Real robot trajectories, when you are ready for them.

Deliberately fetches nothing. Everything here should first be proved on `synthetic.py`, where the
answer is a matrix you generated — so the first run needs no download at all.

soft-robot-koopman (Bruder et al., IEEE T-RO 2021), with the pneumatic arm data used in the paper:
    https://github.com/ramvasudevan/soft-robot-koopman
MuJoCo, for an arm the paper never touched:
    https://mujoco.org

Both are a `git clone` and a `pip install` away. Neither belongs in the image, because the
mathematics is what you are writing and it needs neither.
"""

from __future__ import annotations

if __name__ == "__main__":
    print(__doc__)
