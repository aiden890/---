"""RL flow-SDE inference server (GPU side, runs in the xiaomi-cu121 image).

Drop-in sibling of upstream/deploy/server.py. Same length-prefixed pickle wire
protocol, same model load, but the request may ask for flow-SDE sampling and the
response carries the per-transition log-prob and (optionally) the executed-action
mask sum, so the RoboCasa client (CPU-torch image, no GPU) can drive skill-conditioned
RL rollouts and store exactly what GRPO needs.

Why server-side sampling: the client image has only CPU torch and cannot host the
model; the server image has CUDA torch + flash-attn + the checkpoint but no simulator.
Splitting sampling (GPU) from stepping (sim) across the socket is the existing,
verified architecture -- we only extend the server's per-request behaviour.

Wire protocol (unchanged framing):
    request  = pickle({ ...processor inputs..., "task_id": robot_type,
                        "rl": {"eta": float, "num_steps": int,
                               "replan_steps": int, "real_action_dim": int,
                               "seed": int|None} })       # "rl" absent -> deterministic
    response = pickle({ "actions": Tensor[B,L,A_full],     # raw model action space
                        "executed_logprob": [float]*B | None,
                        "chunk_logprob": [float]*B | None,
                        "x0_logprob": [float]*B | None,
                        "num_steps": int, "eta": float })

The deterministic path (no "rl" key) returns the same actions as upstream server.py
(model.forward) so the existing client keeps working unchanged.
"""
from __future__ import annotations

import argparse
import pickle
import socket
import struct
import sys
import traceback
from pathlib import Path

import torch
from transformers import AutoModel

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
from flow_policy import sample_action_chunk_sde  # noqa: E402


class RLServer:
    def __init__(self, model_path, host, port):
        self.host, self.port = host, port
        print("Loading model (RL flow-SDE server)...", flush=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16
        ).cuda().to(torch.bfloat16)
        self.model.eval()
        print("Model loaded.", flush=True)

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

    def handle(self, input_data):
        rl = input_data.pop("rl", None)
        data = {k: self._to_dev(v) for k, v in input_data.items()}
        state = data.pop("state")
        action_mask = data.pop("action_mask")
        data.pop("task_id", None)

        if rl is None or float(rl.get("eta", 0.0)) == 0.0 and not rl.get("force", False):
            # deterministic: identical to upstream server (model.forward Euler sampler)
            with torch.no_grad():
                out = self.model(state=state, action_mask=action_mask,
                                 num_steps=(rl or {}).get("num_steps", 5), **data)
            return {"actions": out.actions.cpu(), "executed_logprob": None, "chunk_logprob": None,
                    "x0_logprob": None, "num_steps": (rl or {}).get("num_steps", 5), "eta": 0.0}

        gen = None
        if rl.get("seed") is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(rl["seed"]))
        res = sample_action_chunk_sde(
            self.model, state, action_mask,
            num_steps=int(rl.get("num_steps", 5)), eta=float(rl["eta"]),
            replan_steps=rl.get("replan_steps"), real_action_dim=rl.get("real_action_dim"),
            generator=gen, grad=False, **data)
        return {
            "actions": res.actions.cpu(),
            "executed_logprob": [float(x) for x in res.executed_logprob().float().cpu()],
            "chunk_logprob": [float(x) for x in res.chunk_logprob.float().cpu()],
            "x0_logprob": [float(x) for x in res.x0_logprob.float().cpu()],
            "num_steps": res.num_steps, "eta": res.eta,
        }

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(1)
            print(f"RL server running on {self.host}:{self.port}...", flush=True)
            while True:
                conn, _ = s.accept()
                try:
                    while True:
                        ln = self._recv_all(conn, 4)
                        if not ln:
                            break
                        n = struct.unpack(">I", ln)[0]
                        payload = self._recv_all(conn, n)
                        if not payload:
                            break
                        resp = self.handle(pickle.loads(payload))
                        out = pickle.dumps(resp)
                        conn.sendall(struct.pack(">I", len(out)) + out)
                except Exception as e:
                    print(f"Error: {e}", flush=True)
                    traceback.print_exc()
                finally:
                    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10087)
    a = ap.parse_args()
    RLServer(a.model, a.host, a.port).serve()


if __name__ == "__main__":
    main()
