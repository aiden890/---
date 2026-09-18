#!/usr/bin/env python3
"""Single-model deterministic generation server used only by the planning probe."""
from __future__ import annotations

import argparse
import pickle
import socket
import struct
import time


def recv_all(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            return None
        buf += part
    return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/checkpoint")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10087)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True,
        attn_implementation="flash_attention_2", dtype=torch.bfloat16,
    ).cuda().to(torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("planning probe model ready", flush=True)

    def to_dev(value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(device=model.device, dtype=model.dtype)
            return value.to(device=model.device)
        return value

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        print(f"planning probe server listening on {args.host}:{args.port}", flush=True)
        while True:
            conn, _ = listener.accept()
            with conn:
                while True:
                    header = recv_all(conn, 4)
                    if header is None:
                        break
                    payload = recv_all(conn, struct.unpack(">I", header)[0])
                    if payload is None:
                        break
                    request = pickle.loads(payload)
                    if request.get("op") == "health":
                        response = {"ready": True, "model_load_count": 1,
                                    "device": str(model.device), "dtype": str(model.dtype)}
                    elif request.get("op") == "generate":
                        inputs = {k: to_dev(v) for k, v in request["inputs"].items()}
                        input_len = int(inputs["input_ids"].shape[-1])
                        started = time.monotonic()
                        with torch.no_grad():
                            output = model.vlm.generate(
                                **inputs,
                                max_new_tokens=int(request.get("max_new_tokens", 512)),
                                do_sample=False,
                            )
                        latency_ms = (time.monotonic() - started) * 1000.0
                        text = tokenizer.decode(
                            output[0, input_len:], skip_special_tokens=True
                        ).strip()
                        response = {"text": text, "latency_ms": latency_ms}
                    else:
                        response = {"error": f"unknown op: {request.get('op')!r}"}
                    data = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
                    conn.sendall(struct.pack(">I", len(data)) + data)


if __name__ == "__main__":
    main()
