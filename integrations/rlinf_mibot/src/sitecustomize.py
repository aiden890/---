"""Optional MiBoT registration hook for RLinf Ray worker interpreters.

Ray workers are fresh Python processes: mutations made to RLinf's model registry by the
driver are not inherited.  Python imports ``sitecustomize`` from ``PYTHONPATH`` during
interpreter startup, so native Ray containers opt into this hook with
``RLINF_ENABLE_MIBOT_REGISTRATION=1``.  Non-native tools remain unaffected.
"""
from __future__ import annotations

import os


if os.environ.get("RLINF_ENABLE_MIBOT_REGISTRATION") == "1":
    from rlinf_env import register_mibot
    from mibot_rlinf_env import install_mibot_env_override

    register_mibot()
    install_mibot_env_override()
