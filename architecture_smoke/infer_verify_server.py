"""Combined GPU inference + VLM-verifier server (task t_fc5e73d5).

ONE model load on the GPU serves BOTH:
  * base-policy action inference -- a non-"op" request is handled EXACTLY like
    the stock ``upstream/deploy/server.py`` (``model(**data)`` -> raw
    ``outputs.actions`` tensor), so the unchanged ``rollout.EvalClient`` /
    ``Client`` drives it byte-identically; and
  * ``op="vlm_score"`` -- the obs-only skill-termination verifier's VQA forward
    (``model.vlm(...).logits`` -> P(yes)), reusing the env card's ``vlm_scorer``
    module (single source of truth for the VQA prompt + logit->prob math).

Why one load: the obs-only VLM verifier IS the policy's own frozen Qwen3-VL
backbone, so we must not load the 5B model twice (24GB card). A per-connection
thread + a single CUDA lock serialises the two request types (they are naturally
sequential in a rollout anyway), so the action client and the verifier client can
each hold their own persistent socket to the same model.

Runs in the xiaomi-cu121 image (CUDA torch + flash-attn + checkpoint). Wire
protocol = the stock length-prefixed pickle framing.
"""
from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import threading
import traceback
from pathlib import Path

import torch
from transformers import AutoModel

# vlm_scorer is the env card's single source of truth for the VQA prompt + math.
# It is mounted at /rl_env/src inside the container.
for _p in ("/rl_env/src", str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import vlm_scorer  # noqa: E402


class InferVerifyServer:
    def __init__(self, model_path, host, port):
        self.host, self.port = host, port
        self.model_path = model_path
        self._lock = threading.Lock()
        self._yn_ids = None
        print("Loading model (combined infer+verify server)...", flush=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            attn_implementation="flash_attention_2", dtype=torch.bfloat16,
        ).cuda().to(torch.bfloat16)
        self.model.eval()
        print("Model loaded.", flush=True)

    # ---- wire helpers ----------------------------------------------------- #
    def _recv_all(self, conn, n):
        data = b""
        while len(data) < n:
            pkt = conn.recv(n - len(data))
            if not pkt:
                return None
            data += pkt
        return data

    def _to_dev(self, v):
        if isinstance(v, torch.Tensor):
            return v.to(device=self.model.device, dtype=self.model.dtype) if v.is_floating_point() \
                else v.to(device=self.model.device)
        return v

    # ---- request handlers ------------------------------------------------- #
    def _yes_no_ids(self):
        if self._yn_ids is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._yn_ids = vlm_scorer.resolve_yes_no_ids(tok)
        return self._yn_ids

    def _op_vlm_score(self, req):
        """P(yes) from one VQA forward on the frozen backbone (obs-only verifier)."""
        data = {k: self._to_dev(v) for k, v in req["inputs"].items()}
        yes_ids, no_ids = self._yes_no_ids()
        with torch.no_grad():
            out = self.model.vlm(
                input_ids=data["input_ids"],
                attention_mask=data.get("attention_mask"),
                pixel_values=data.get("pixel_values"),
                image_grid_thw=data.get("image_grid_thw"),
            )
            logits_last = out.logits[0, -1, :]
            prob = vlm_scorer.answer_probability(logits_last, yes_ids, no_ids)
        return {"prob": float(prob), "question": req.get("question")}

    def _base_actions(self, input_data):
        """Stock action path: identical to upstream/deploy/server.py."""
        data = {
            k: self._to_dev(v) for k, v in input_data.items()
        }
        with torch.no_grad():
            outputs = self.model(**data)
        return outputs.actions.cpu()

    def handle(self, req):
        # An "op" key => structured verifier RPC (returns a dict).
        # No "op" key => stock base-policy action request (returns a raw tensor).
        if isinstance(req, dict) and req.get("op") == "vlm_score":
            with self._lock:
                return self._op_vlm_score(req)
        with self._lock:
            return self._base_actions(req)

    # ---- serving ---------------------------------------------------------- #
    def _serve_conn(self, conn):
        try:
            while True:
                ln = self._recv_all(conn, 4)
                if not ln:
                    break
                n = struct.unpack(">I", ln)[0]
                payload = self._recv_all(conn, n)
                if payload is None:
                    break
                resp = self.handle(pickle.loads(payload))
                out = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
                conn.sendall(struct.pack(">I", len(out)) + out)
        except Exception as e:  # noqa: BLE001
            print(f"conn error: {e}", flush=True)
            traceback.print_exc()
        finally:
            conn.close()

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(4)
            print(f"Infer+verify server running on {self.host}:{self.port}...", flush=True)
            while True:
                conn, _ = s.accept()
                threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10086)
    a = ap.parse_args()
    InferVerifyServer(a.model, a.host, a.port).serve()


if __name__ == "__main__":
    main()
