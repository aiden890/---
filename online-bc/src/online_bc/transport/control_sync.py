"""Keep exclusion controls current through HF; no inbound learner ports."""

import json
import threading
import time
from pathlib import Path
from online_bc.data.data_control import atomic_json


class ControlSync:
    def __init__(self, root, run, sync, interval=10):
        self.root = Path(root)
        self.run = run
        self.sync = sync
        self.interval = interval
        self.path = self.root / "controls.json"
        self.health = self.root / "control-health.json"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def once(self):
        with self.lock:
            return self._once()

    def _once(self):
        folder = self.root / "incoming-controls"
        r = self.sync("download", folder, f"{self.run}/controls")
        if r.returncode:
            raise RuntimeError("Review control download failed")
        incoming = json.loads((folder / "controls.json").read_text())
        old = json.loads(self.path.read_text()) if self.path.exists() else {"revision": -1}
        if incoming["revision"] >= old["revision"]:
            atomic_json(self.path, incoming)
        atomic_json(self.health, {"last_success": time.time(), "revision": incoming["revision"]})
        state = {
            "updated_at": time.time(),
            "control_revision": incoming["revision"],
            "status": "waiting_for_rollouts",
            "used": {},
            "rounds": [],
        }
        status = self.root / "service-status.json"
        if status.exists():
            state.update(json.loads(status.read_text()))
        for folder in sorted((self.root / "training").glob("*")):
            progress = folder / "data-usage.json"
            if progress.exists():
                d = json.loads(progress.read_text())
                state["rounds"].append({"round": folder.name, **d})
                for eid, count in d.get("used", {}).items():
                    state["used"][eid] = state["used"].get(eid, 0) + count
                state["status"] = d.get("status", state["status"])
        state_dir = self.root / "published-state"
        atomic_json(state_dir / "state.json", state)
        self.sync("upload", state_dir, f"{self.run}/learner-state/pi05")

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.once()
            except (OSError, ValueError, KeyError, RuntimeError):
                pass  # Health expires and training pauses on a stale control channel.
            self.stop_event.wait(self.interval)

    def start(self):
        self.once()  # Require an initial control snapshot before any gradient step.
        threading.Thread(target=self.loop, daemon=True).start()
