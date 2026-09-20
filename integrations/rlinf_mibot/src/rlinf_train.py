#!/usr/bin/env python3
"""Register MiBoT, then enter RLinf's unmodified embodied training launcher.

Hydra arguments are forwarded unchanged. Typical container invocation::

    python /integration/src/rlinf_train.py \
      --config-path /integration/configs --config-name rlinf_mibot_grpo
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    rlinf_root = Path(os.environ.get("RLINF_ROOT", "/opt/rlinf"))
    entrypoint = rlinf_root / "examples" / "embodiment" / "train_embodied_agent.py"
    if not entrypoint.is_file():
        raise SystemExit(f"RLinf embodied launcher was not found: {entrypoint}")

    from rlinf_env import register_mibot
    from mibot_rlinf_env import install_mibot_env_override

    register_mibot()
    install_mibot_env_override()
    # run_name=__main__ preserves RLinf's own Hydra decorator and launch lifecycle.
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
