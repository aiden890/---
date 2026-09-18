#!/usr/bin/env python3
"""MiBoT inference server that records exact, read-only attention summaries.

The policy forward remains the checkpoint implementation (FlashAttention/SDPA). Hooks
reconstruct probabilities from the same projected Q/K tensors without replacing model
layers or changing outputs. Only requested probe calls carrying ``attention_meta`` are
recorded; snapshot-generation calls remain ordinary policy inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import socket
import struct
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


SEMANTIC_LEXEMES = {
    "GRASP": ["pick", "grasp", "secure"],
    "MOVE": ["move", "above", "collid", "surround"],
    "PLACE": ["place", "plac", "top", "release"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_subsequence(sequence: list[int], needle: list[int], start: int = 0, end: int | None = None) -> list[int]:
    if not needle:
        return []
    end = len(sequence) if end is None else end
    for index in range(start, end - len(needle) + 1):
        if sequence[index : index + len(needle)] == needle:
            return list(range(index, index + len(needle)))
    return []


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


class AttentionRecorder:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, out_dir: Path, model_path: Path):
        self.model = model
        self.tokenizer = tokenizer
        self.out_dir = out_dir
        self.model_path = model_path
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.active: dict[str, Any] | None = None
        self.handles: list[Any] = []
        self._install_hooks()

    @property
    def text_layers(self):
        return self.model.vlm.model.language_model.layers

    @property
    def dit_layers(self):
        return self.model.dit.layers

    def _install_hooks(self) -> None:
        for layer_index, layer in enumerate(self.text_layers):
            self.handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._make_vlm_hook(layer_index), with_kwargs=True
                )
            )
        for layer_index, layer in enumerate(self.dit_layers):
            self.handles.append(
                layer.attn.register_forward_pre_hook(
                    self._make_dit_hook(layer_index), with_kwargs=True
                )
            )
        original = self.model.dit_forward

        def wrapped_dit_forward(module, *args, **kwargs):
            if self.active is not None:
                self.active["current_timestep"] = self.active["dit_call_count"]
                self.active["dit_call_count"] += 1
            return original(*args, **kwargs)

        self.model.dit_forward = MethodType(wrapped_dit_forward, self.model)

    def begin(self, input_ids: torch.Tensor, meta: dict[str, Any]) -> None:
        ids = input_ids[0].detach().cpu().tolist()
        instruction = str(meta["instruction"])
        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        instruction_positions = find_subsequence(ids, instruction_ids)
        if not instruction_positions:
            raise RuntimeError(f"instruction token span not found for {instruction!r}")

        image_id = int(self.model.config.vlm_config.image_token_id)
        video_id = int(self.model.config.vlm_config.video_token_id)
        visual_positions = [index for index, token_id in enumerate(ids) if token_id in (image_id, video_id)]
        if not visual_positions:
            raise RuntimeError("no image/video placeholder tokens found")

        semantic_positions: dict[str, list[int]] = {}
        for role, lexemes in SEMANTIC_LEXEMES.items():
            semantic_positions[role] = [
                position
                for position in instruction_positions
                if any(
                    lexeme in self.tokenizer.decode([ids[position]]).lower()
                    for lexeme in lexemes
                )
            ]

        layer_count = len(self.text_layers)
        vlm_heads = self.text_layers[0].self_attn.config.num_attention_heads
        dit_layer_count = len(self.dit_layers)
        dit_heads = self.dit_layers[0].attn.num_heads
        action_length = int(meta.get("action_length", 30))
        num_steps = int(meta.get("num_steps", 5))
        nan = np.float32(np.nan)
        self.active = {
            "meta": dict(meta),
            "ids": ids,
            "instruction_positions": instruction_positions,
            "visual_positions": visual_positions,
            "semantic_positions": semantic_positions,
            "vlm_instruction_to_image": np.full((layer_count, vlm_heads), nan, np.float32),
            "vlm_instruction_self": np.full((layer_count, vlm_heads), nan, np.float32),
            "vlm_image_to_instruction": np.zeros((layer_count, vlm_heads), np.float32),
            "dit_action_to_instruction": np.full((num_steps, dit_layer_count, dit_heads), nan, np.float32),
            "dit_action_to_image": np.full((num_steps, dit_layer_count, dit_heads), nan, np.float32),
            "dit_action_to_text_total": np.full((num_steps, dit_layer_count, dit_heads), nan, np.float32),
            "dit_query_to_instruction": np.full(
                (num_steps, dit_layer_count, dit_heads, action_length), nan, np.float32
            ),
            "semantic": {
                role: np.full((num_steps, dit_layer_count, dit_heads), nan, np.float32)
                for role in SEMANTIC_LEXEMES
            },
            "dit_call_count": 0,
            "current_timestep": 0,
        }

    @staticmethod
    def _add_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return logits
        mask = mask[..., : logits.shape[-2], : logits.shape[-1]]
        if mask.dtype == torch.bool:
            return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits + mask.to(logits.dtype)

    def _make_vlm_hook(self, layer_index: int):
        def hook(module, args, kwargs):
            if self.active is None:
                return
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            attention_mask = kwargs.get("attention_mask")
            if hidden is None or position_embeddings is None:
                raise RuntimeError("VLM attention hook did not receive hidden states/position embeddings")
            input_shape = hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
            checkpoint_module = sys.modules[module.__class__.__module__]
            query, key = checkpoint_module.apply_rotary_pos_emb(query, key, *position_embeddings)
            key = checkpoint_module.repeat_kv(key, module.num_key_value_groups)

            instruction = self.active["instruction_positions"]
            visual = self.active["visual_positions"]
            q = query[:, :, instruction, :]
            logits = torch.matmul(q, key.transpose(2, 3)) * module.scaling
            if attention_mask is not None and attention_mask.ndim == 4:
                qmask = attention_mask[:, :, instruction, : key.shape[-2]]
                logits = self._add_mask(logits, qmask)
            else:
                key_positions = torch.arange(key.shape[-2], device=key.device).view(1, 1, 1, -1)
                query_positions = torch.tensor(instruction, device=key.device).view(1, 1, -1, 1)
                allowed = key_positions <= query_positions
                if attention_mask is not None and attention_mask.ndim == 2:
                    allowed = allowed & attention_mask[:, None, None, : key.shape[-2]].bool()
                logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits.float(), dim=-1)
            self.active["vlm_instruction_to_image"][layer_index] = (
                probs[..., visual].sum(-1).mean(2)[0].detach().cpu().numpy()
            )
            self.active["vlm_instruction_self"][layer_index] = (
                probs[..., instruction].sum(-1).mean(2)[0].detach().cpu().numpy()
            )
            # Every visual token precedes the instruction in this causal decoder. The reverse
            # direction is therefore exactly zero; it is recorded explicitly rather than inferred.
        return hook

    def _make_dit_hook(self, layer_index: int):
        def hook(module, args, kwargs):
            if self.active is None:
                return
            hidden = kwargs.get("hidden_state", args[0] if args else None)
            past = kwargs.get("past_key_values", args[1] if len(args) > 1 else None)
            position_embeddings = kwargs.get("position_embeds", args[2] if len(args) > 2 else None)
            attention_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
            if hidden is None or past is None or position_embeddings is None:
                raise RuntimeError("DiT attention hook arguments missing")
            batch_size, query_length, _ = hidden.shape
            qkv = module.qkv_proj(hidden).view(batch_size, query_length, 3, module.num_heads, module.head_dim)
            query, key, _ = qkv.unbind(2)
            query = module.q_norm(query).transpose(1, 2)
            key = module.k_norm(key).transpose(1, 2)
            checkpoint_module = sys.modules[module.__class__.__module__]
            query, key = checkpoint_module.apply_rotary_pos_emb(query, key, *position_embeddings)
            cache_key = checkpoint_module.repeat_kv(past[0], module.num_key_value_groups)
            key = torch.cat([cache_key, key], dim=-2)

            action_length = self.active["dit_query_to_instruction"].shape[-1]
            action_q = query[:, :, -action_length:, :]
            logits = torch.matmul(action_q, key.transpose(2, 3)) / math.sqrt(module.head_dim)
            if attention_mask is not None:
                qmask = attention_mask[:, :, -action_length:, : key.shape[-2]]
                logits = self._add_mask(logits, qmask)
            probs = torch.softmax(logits.float(), dim=-1)
            instruction = self.active["instruction_positions"]
            visual = self.active["visual_positions"]
            timestep = int(self.active["current_timestep"])
            if timestep >= self.active["dit_action_to_instruction"].shape[0]:
                raise RuntimeError(f"unexpected DiT timestep {timestep}")
            instruction_mass = probs[..., instruction].sum(-1)[0]
            self.active["dit_query_to_instruction"][timestep, layer_index] = (
                instruction_mass.detach().cpu().numpy()
            )
            self.active["dit_action_to_instruction"][timestep, layer_index] = (
                instruction_mass.mean(-1).detach().cpu().numpy()
            )
            self.active["dit_action_to_image"][timestep, layer_index] = (
                probs[..., visual].sum(-1).mean(2)[0].detach().cpu().numpy()
            )
            prefix_length = cache_key.shape[-2]
            self.active["dit_action_to_text_total"][timestep, layer_index] = (
                probs[..., :prefix_length].sum(-1).mean(2)[0].detach().cpu().numpy()
            )
            for role, positions in self.active["semantic_positions"].items():
                if positions:
                    mass = probs[..., positions].sum(-1).mean(2)[0]
                else:
                    mass = torch.zeros(module.num_heads, device=probs.device)
                self.active["semantic"][role][timestep, layer_index] = mass.detach().cpu().numpy()
        return hook

    def finish(self, actions: torch.Tensor) -> dict[str, Any]:
        if self.active is None:
            raise RuntimeError("finish called without active recording")
        active = self.active
        self.active = None
        expected = int(active["meta"].get("num_steps", 5))
        if active["dit_call_count"] != expected:
            raise RuntimeError(f"expected {expected} DiT calls, observed {active['dit_call_count']}")
        missing = np.isnan(active["dit_action_to_instruction"]).sum()
        if missing:
            raise RuntimeError(f"DiT attention summary contains {missing} missing values")
        meta = active["meta"]
        stem = f"{meta['state_tag']}__{meta['label']}"
        arrays = {
            "vlm_instruction_to_image": active["vlm_instruction_to_image"],
            "vlm_image_to_instruction": active["vlm_image_to_instruction"],
            "vlm_instruction_self": active["vlm_instruction_self"],
            "dit_action_to_instruction": active["dit_action_to_instruction"],
            "dit_action_to_image": active["dit_action_to_image"],
            "dit_action_to_text_total": active["dit_action_to_text_total"],
            "dit_query_to_instruction": active["dit_query_to_instruction"],
            "actions": actions.detach().float().cpu().numpy(),
        }
        for role, values in active["semantic"].items():
            arrays[f"dit_action_to_{role.lower()}_tokens"] = values
        np.savez_compressed(self.out_dir / f"{stem}.npz", **arrays)

        token_records = []
        instruction_set = set(active["instruction_positions"])
        visual_set = set(active["visual_positions"])
        role_sets = {role: set(pos) for role, pos in active["semantic_positions"].items()}
        for index, token_id in enumerate(active["ids"]):
            roles = [role for role, positions in role_sets.items() if index in positions]
            token_records.append({
                "index": index,
                "id": token_id,
                "token": self.tokenizer.convert_ids_to_tokens(token_id),
                "decoded": self.tokenizer.decode([token_id]),
                "kind": "instruction" if index in instruction_set else "image" if index in visual_set else "other",
                "semantic_roles": roles,
            })
        mapping = {
            "meta": meta,
            "sequence_length": len(active["ids"]),
            "instruction_indices": active["instruction_positions"],
            "image_indices": active["visual_positions"],
            "semantic_indices": active["semantic_positions"],
            "tokens": token_records,
        }
        mapping_path = self.out_dir / f"{stem}.tokens.json"
        mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "state_tag": meta["state_tag"],
            "label": meta["label"],
            "instruction": meta["instruction"],
            "npz": f"{stem}.npz",
            "token_mapping": mapping_path.name,
            "sequence_length": len(active["ids"]),
            "instruction_tokens": len(active["instruction_positions"]),
            "image_tokens": len(active["visual_positions"]),
            "vlm_layers": len(self.text_layers),
            "vlm_heads": self.text_layers[0].self_attn.config.num_attention_heads,
            "dit_layers": len(self.dit_layers),
            "dit_heads": self.dit_layers[0].attn.num_heads,
            "timesteps": expected,
            "dtype": "float32 summaries reconstructed from checkpoint bf16 Q/K",
            "mean_action_to_instruction": float(arrays["dit_action_to_instruction"].mean()),
            "mean_action_to_image": float(arrays["dit_action_to_image"].mean()),
            "mean_instruction_to_image": float(arrays["vlm_instruction_to_image"].mean()),
        }
        with (self.out_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return summary


class Server:
    def __init__(self, model_path: Path, host: str, port: int, out_dir: Path):
        self.model_path = model_path
        self.host = host
        self.port = port
        print("Loading checkpoint model...", flush=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        ).cuda().to(torch.bfloat16).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.recorder = AttentionRecorder(self.model, tokenizer, out_dir, model_path)
        provenance = {
            "checkpoint": str(model_path),
            "config_sha256": sha256_file(model_path / "config.json"),
            "model_index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
            "checkpoint_model_class": self.model.__class__.__name__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "attention_backend": "flash_attention_2 (policy); exact selected QK probabilities reconstructed read-only",
            "server_source_sha256": sha256_file(Path(__file__)),
        }
        (out_dir / "server_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print("ATTENTION_SERVER_READY", flush=True)

    @staticmethod
    def recv_all(conn: socket.socket, length: int) -> bytes | None:
        data = b""
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            while True:
                conn, address = server_socket.accept()
                print(f"client={address}", flush=True)
                try:
                    while True:
                        size_bytes = self.recv_all(conn, 4)
                        if not size_bytes:
                            break
                        size = struct.unpack(">I", size_bytes)[0]
                        payload = self.recv_all(conn, size)
                        if not payload:
                            break
                        request = pickle.loads(payload)
                        meta = request.pop("attention_meta", None)
                        request.pop("task_id", None)
                        data = {
                            key: value.to(
                                device=self.model.device,
                                dtype=self.model.dtype if value.is_floating_point() else value.dtype,
                            ) if isinstance(value, torch.Tensor) else value
                            for key, value in request.items()
                        }
                        if meta is not None:
                            self.recorder.begin(data["input_ids"], meta)
                        started = time.time()
                        outputs = self.model(**data)
                        if meta is not None:
                            summary = self.recorder.finish(outputs.actions)
                            print(f"recorded {summary['state_tag']}/{summary['label']} in {time.time()-started:.2f}s", flush=True)
                        response = pickle.dumps(outputs.actions.cpu())
                        conn.sendall(struct.pack(">I", len(response)) + response)
                except Exception as error:
                    self.recorder.active = None
                    print(f"request failed: {error!r}", file=sys.stderr, flush=True)
                    raise
                finally:
                    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    args = parser.parse_args()
    Server(args.model, args.host, args.port, args.out).serve()


if __name__ == "__main__":
    main()
