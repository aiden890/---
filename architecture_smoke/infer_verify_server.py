"""Combined GPU inference + VLM-verifier server (task t_fc5e73d5).

ONE model load on the GPU serves BOTH:
  * base-policy action inference -- a non-"op" request is handled EXACTLY like
    the stock ``upstream/deploy/server.py`` (``model(**data)`` -> raw
    ``outputs.actions`` tensor), so the unchanged ``rollout.EvalClient`` /
    ``Client`` drives it byte-identically; and
  * ``op="background_vlm"`` -- the canonical obs-only verifier/planner lane
    (``model.vlm(...).logits`` -> P(yes)), reusing the env card's ``vlm_scorer``
    module (single source of truth for the VQA prompt + logit->prob math).

Why one load: the obs-only VLM verifier IS the policy's own frozen Qwen3-VL
backbone, so we must not load the 5B model twice (24GB card). Per-connection
threads submit to a two-lane priority dispatcher: policy FIFO always wins over
the latest-only background slot, while one worker guarantees one CUDA forward.

Runs in the xiaomi-cu121 image (CUDA torch + flash-attn + checkpoint). Wire
protocol = the stock length-prefixed pickle framing.
"""
from __future__ import annotations

import argparse
import importlib
import pickle
import socket
import struct
import sys
import threading
import time
import traceback
from pathlib import Path

# vlm_scorer is the env card's single source of truth for the VQA prompt + math.
# It is mounted at /rl_env/src inside the container.
for _p in ("/rl_env/src", str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import vlm_scorer  # noqa: E402
from async_control import ControlRequest  # noqa: E402
from priority_dispatcher import PriorityDispatcher, SupersededError  # noqa: E402


class InferVerifyServer:
    DEFAULT_PLANNER_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
    DEFAULT_PLANNER_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

    @staticmethod
    def _load_model(model_path):
        torch = importlib.import_module("torch")
        AutoModel = importlib.import_module("transformers").AutoModel
        return AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            attn_implementation="flash_attention_2", dtype=torch.bfloat16,
        ).cuda().to(torch.bfloat16)

    @classmethod
    def _load_planner(cls, model_path, revision):
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        processor = transformers.AutoProcessor.from_pretrained(model_path, revision=revision)
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, revision=revision, dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").cuda().eval()
        return model, processor

    def __init__(self, model_path, host, port, *, model_loader=None,
                 planner_model_path=None, planner_revision=None, planner_loader=None):
        self.host, self.port = host, port
        self.model_path = model_path
        self._yn_ids = None
        self._tokenizer = None
        print("Loading model (combined infer+verify server)...", flush=True)
        if model_loader is None:
            model_loader = lambda: self._load_model(model_path)
        self.model = model_loader()
        if self.model is None:
            raise RuntimeError("model loader returned None")
        self.model_load_count = 1
        self.model.eval()
        self.planner_model = None
        self.planner_processor = None
        self.planner_model_path = planner_model_path
        if planner_model_path is not None:
            if planner_loader is None:
                planner_loader = lambda: self._load_planner(
                    planner_model_path, planner_revision or self.DEFAULT_PLANNER_REVISION)
            self.planner_model, self.planner_processor = planner_loader()
            if self.planner_model is None or self.planner_processor is None:
                raise RuntimeError("planner loader returned an incomplete model/processor pair")
            self.planner_model.eval()
            self.model_load_count += 1
        self.dispatcher = PriorityDispatcher()
        self._ready = True
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
        torch = importlib.import_module("torch")
        if isinstance(v, torch.Tensor):
            return v.to(device=self.model.device, dtype=self.model.dtype) if v.is_floating_point() \
                else v.to(device=self.model.device)
        return v

    # ---- request handlers ------------------------------------------------- #
    def _yes_no_ids(self):
        if self._yn_ids is None:
            self._yn_ids = vlm_scorer.resolve_yes_no_ids(self._get_tokenizer())
        return self._yn_ids

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True)
        return self._tokenizer

    def _op_vlm_score(self, req):
        """P(yes) from one VQA forward on the frozen backbone (obs-only verifier)."""
        torch = importlib.import_module("torch")
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

    def _op_planner(self, req):
        """Deterministic JSON generation on the dedicated generic Qwen planner."""
        if self.planner_model is None or self.planner_processor is None:
            raise RuntimeError("generic Qwen planner is not loaded")
        torch = importlib.import_module("torch")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": req["image"]},
            {"type": "text", "text": req["prompt"]},
        ]}]
        inputs = self.planner_processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        data = {}
        for key, value in dict(inputs).items():
            if isinstance(value, torch.Tensor):
                value = value.to(
                    device=self.planner_model.device,
                    dtype=(self.planner_model.dtype if value.is_floating_point() else None))
            data[key] = value
        input_len = int(data["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = self.planner_model.generate(
                **data, max_new_tokens=int(req.get("max_new_tokens", 384)),
                do_sample=False)
        text = self.planner_processor.tokenizer.decode(
            generated[0, input_len:], skip_special_tokens=True).strip()
        return {"text": text}

    def _base_actions(self, input_data):
        """Stock action path: identical to upstream/deploy/server.py."""
        torch = importlib.import_module("torch")
        data = {
            k: self._to_dev(v) for k, v in input_data.items()
        }
        with torch.no_grad():
            outputs = self.model(**data)
        return outputs.actions.cpu()

    @staticmethod
    def _control_request(req):
        required = ("episode_id", "skill_id", "observation_step", "request_id",
                    "request_kind", "payload")
        missing = [key for key in required if key not in req]
        if missing:
            raise ValueError(f"background request missing canonical fields: {missing}")
        return ControlRequest(**{key: req[key] for key in required})

    def _background_response(self, req):
        """Validate and execute the canonical control envelope."""
        control = self._control_request(req)
        started = time.monotonic()
        try:
            operation = control.payload.get("operation", "score")
            if control.request_kind.value != "planner" and operation == "planner":
                raise ValueError("planner operation requires request_kind=planner")
            function = (lambda: self._op_planner(control.payload)) if operation == "planner" \
                else (lambda: self._op_vlm_score(control.payload))
            result = self.dispatcher.submit_background(
                control.request_kind.value,
                function,
                timeout_s=req.get("timeout_s"),
            )
            status, error = "ok", None
        except SupersededError as exc:
            result, status, error = {}, "dropped", str(exc)
        except TimeoutError as exc:
            result, status, error = {}, "timeout", str(exc)
        except Exception as exc:  # preserve the connection after an isolated model fault
            result, status, error = {}, "error", repr(exc)
        return {
            "episode_id": control.episode_id, "skill_id": control.skill_id,
            "observation_step": control.observation_step,
            "request_id": control.request_id,
            "request_kind": control.request_kind.value,
            "status": status, "error": error, "result": result,
            "round_trip_ms": (time.monotonic() - started) * 1000.0,
        }

    def handle(self, req):
        # No "op" key remains the byte-compatible stock policy API.
        if not isinstance(req, dict) or "op" not in req:
            return self.dispatcher.submit_policy(lambda: self._base_actions(req))
        if req.get("op") == "background_vlm":
            return self._background_response(req)
        if req.get("op") == "health":
            return {"healthy": self.dispatcher.is_alive, "ready": self._ready,
                    "model_load_count": self.model_load_count,
                    "generic_planner_loaded": self.planner_model is not None,
                    "planner_model_path": self.planner_model_path,
                    "scheduler": self.dispatcher.snapshot()}
        raise ValueError(f"unknown structured operation: {req.get('op')!r}")

    def close(self):
        self._ready = False
        self.dispatcher.close(drain=False)

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
    ap.add_argument("--planner-model", default=InferVerifyServer.DEFAULT_PLANNER_MODEL)
    ap.add_argument("--planner-revision", default=InferVerifyServer.DEFAULT_PLANNER_REVISION)
    a = ap.parse_args()
    InferVerifyServer(
        a.model, a.host, a.port,
        planner_model_path=a.planner_model, planner_revision=a.planner_revision).serve()


if __name__ == "__main__":
    main()
