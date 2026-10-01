from pathlib import Path

import argparse
import json
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from online_bc.data.data_control import atomic_json, read_controls, select_candidates, control_ready
from online_bc.review.review_server import ReviewServer


class ReviewTests(unittest.TestCase):
    def test_excluded_episode_cannot_be_sampled_and_restore_works(self):
        pool = {
            0: (Path("a"), dict(model="pi05", seed=1, samples=[{"skill": "cup_placement"}])),
            1: (Path("b"), dict(model="xiaomi", seed=2, samples=[{"skill": "cup_placement"}])),
        }
        controls = dict(revision=1, excluded={"pi05-seed1": {"reason": "bad"}})
        self.assertEqual([x[0] for x in select_candidates(pool, controls, "cup_placement")], [1])
        controls["excluded"].clear()
        self.assertEqual(len(select_candidates(pool, controls, "cup_placement")), 2)
        controls["excluded"] = {"pi05-seed1": {}, "xiaomi-seed2": {}}
        self.assertEqual(select_candidates(pool, controls, "cup_placement"), [])

    def test_stale_control_and_pause_stop_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "health.json"
            c = dict(paused=False)
            self.assertEqual(control_ready(c, p)[1], "control_sync_stale")
            atomic_json(p, {"last_success": time.time()})
            self.assertTrue(control_ready(c, p)[0])
            atomic_json(p, {"last_success": time.time() - 46})
            self.assertFalse(control_ready(c, p)[0])
            self.assertEqual(control_ready({"paused": True})[1], "user_paused")

    def test_api_revision_persistence_origin_and_unknown_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            atomic_json(
                root / "tracking/reports/pi-cup-data-catalog.json",
                {
                    "episodes": [
                        {"id": "pi05-seed1", "eligible": True},
                        {"id": "pi05-seed2", "eligible": False},
                    ]
                },
            )
            args = argparse.Namespace(
                root=str(root / "review"),
                tracking=str(root / "tracking"),
                run="test",
                bind="127.0.0.1",
                port=0,
            )
            server = ReviewServer(args)
            server.origin = "http://127.0.0.1:" + str(server.server_port)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:

                def request(value, origin=None):
                    req = urllib.request.Request(
                        server.origin + "/api/control",
                        data=json.dumps(value).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "Origin": origin or server.origin,
                        },
                    )
                    try:
                        with urllib.request.urlopen(req) as r:
                            return r.status, json.load(r)
                    except urllib.error.HTTPError as e:
                        return e.code, json.load(e)

                code, r = request(
                    dict(
                        action="exclude",
                        id="pi05-seed1",
                        excluded=True,
                        reason="reviewed",
                        revision=0,
                    )
                )
                self.assertEqual(code, 200)
                self.assertIn("pi05-seed1", read_controls(server.control_path)["excluded"])
                self.assertEqual(
                    request(dict(action="exclude", id="pi05-seed1", excluded=False, revision=0))[0],
                    409,
                )
                self.assertEqual(
                    request(dict(action="exclude", id="pi05-seed2", excluded=True, revision=1))[0],
                    400,
                )
                self.assertEqual(
                    request(
                        dict(action="pause", paused=True, revision=1), origin="http://other.example"
                    )[0],
                    403,
                )
                self.assertEqual(request(dict(action="pause", paused=True, revision=1))[0], 200)
                self.assertEqual(
                    request(dict(action="exclude", id="pi05-seed1", excluded=False, revision=2))[0],
                    200,
                )
                self.assertEqual(read_controls(server.control_path)["excluded"], {})
                self.assertEqual(
                    len((server.root / "review-audit.jsonl").read_text().splitlines()), 3
                )
                self.assertTrue(read_controls(server.control_path)["paused"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
