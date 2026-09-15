"""Goal-3 SFT dataset + client: build masked per-skill CFM training batches from the
RoboCasa365 CloseBlenderLid LeRobot demos and drive the trainer server's sft_* ops.

Design (grounded in the verified checkpoint I/O):
  * robot_type = "robocasa365"; action space is 60-D but only dims 0-11 are ACTIVE
    (preprocessor std=1/mean=0 on 0-11, std=0 elsewhere). decode_action = a*std+mean is
    the identity on the active dims, so the demo raw action[12] IS the model-space target:
    x1[..., :12] = demo_action, x1[..., 12:] = 0.
  * conditioning inputs are built with the SAME processor.apply_chat_template used at
    rollout (3-camera video history + 16-D dataset state + skill instruction), so SFT and
    eval share one input pipeline (rl_rollout / deploy client).
  * obs history = 4 frames, interval = 2, crop = 0.95, replan/chunk L = 16 (eval config).

A training EXAMPLE = one action-chunk window inside a skill's [start,end) span of one
source demo. The loss mask is 1.0 only on active action dims AND on chunk steps that fall
inside the span (Goal-1 boundary); steps past the span end are masked out.

Videos: the LeRobot v3 mirror stores one concatenated mp4 per camera per data chunk; each
episode occupies [from_timestamp, to_timestamp) (from the episodes meta). We decode the
needed frame window with torchvision.io.read_video (available in xiaomi-cu121).
"""
from __future__ import annotations

import json
import os
import socket
import struct
import pickle
from pathlib import Path

import numpy as np

REPO = "ember-lab-berkeley/robocasa365-pretrain-atomic"
ROBOT_TYPE = "robocasa365"
CAMERA_KEYS = ("observation.images.robot0_agentview_left",
               "observation.images.robot0_agentview_right",
               "observation.images.robot0_eye_in_hand")
STATE_DIM = 60
ACTION_FULL = 60
ACTION_REAL = 12
OBS_HISTORY = 4
OBS_INTERVAL = 2
CROP = 0.95
FPS = 20

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data" / "skill_sft"


def dl(fn, cache):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset", local_dir=cache)


def center_crop_np(img, ratio):
    from PIL import Image
    if ratio >= 1.0:
        return img if isinstance(img, Image.Image) else Image.fromarray(img)
    # Match rollout.py center_crop EXACTLY: crop then resize BACK to the original
    # (H,W) so frames stay patch-divisible (256x256). Cropping without the resize
    # yields 243px frames that the Qwen3-VL video patchifier rejects.
    h, w = img.shape[:2]
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    l, t = (w - cw) // 2, (h - ch) // 2
    cropped = Image.fromarray(np.ascontiguousarray(img[t:t + ch, l:l + cw]))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((w, h), resampling)


class SFTData:
    """Loads segments + parquet + decodes video windows for CloseBlenderLid demos."""

    def __init__(self, cache):
        import pyarrow.parquet as pq
        self.cache = cache
        self.segments = {p["episode_index"]: p
                         for p in json.load(open(DATA_DIR / "skill_segments.json"))}
        self.manifest = json.load(open(DATA_DIR / "data_manifest.json"))
        self.episodes = {e["episode_index"]: e
                         for e in json.load(open(DATA_DIR / "closeblenderlid_episodes.json"))}
        # episodes meta (for video from/to timestamps)
        ep = pq.read_table(dl("meta/episodes/chunk-000/file-000.parquet", cache)).to_pydict()
        self.ep_meta = {}
        for i in range(len(ep["episode_index"])):
            self.ep_meta[int(ep["episode_index"][i])] = {k: ep[k][i] for k in ep}
        self._data_tbl = None
        self._video = {}

    @property
    def data_tbl(self):
        if self._data_tbl is None:
            import pyarrow.parquet as pq
            self._data_tbl = pq.read_table(dl("data/chunk-000/file-000.parquet", self.cache))
        return self._data_tbl

    def episode_rows(self, ei):
        tbl = self.data_tbl
        m = tbl.column("episode_index").to_numpy() == ei
        sub = tbl.filter(m)
        state = np.array(sub.column("observation.state").to_pylist(), dtype=np.float32)
        action = np.array(sub.column("action").to_pylist(), dtype=np.float32)
        return state, action

    def _read_camera(self, ei, cam):
        """Decode all frames of one episode from the concatenated per-chunk mp4."""
        import torchvision
        key = (ei, cam)
        if key in self._video:
            return self._video[key]
        m = self.ep_meta[ei]
        ck = m[f"videos/{cam}/chunk_index"]
        fi = m[f"videos/{cam}/file_index"]
        t0 = float(m[f"videos/{cam}/from_timestamp"])
        t1 = float(m[f"videos/{cam}/to_timestamp"])
        path = dl(f"videos/{cam}/chunk-{ck:03d}/file-{fi:03d}.mp4", self.cache)
        vid, _, _ = torchvision.io.read_video(path, start_pts=t0, end_pts=t1,
                                              pts_unit="sec", output_format="THWC")
        arr = vid.numpy()  # [T,H,W,C] uint8
        self._video[key] = arr
        return arr

    def obs_window(self, ei, step, state):
        """Build the 4-frame obs history ending at `step` (interval 2) for all 3 cameras
        plus the state window, matching the eval sampler."""
        idxs = [max(0, step - k * OBS_INTERVAL) for k in range(OBS_HISTORY - 1, -1, -1)]
        images = {}
        for cam in CAMERA_KEYS:
            frames = self._read_camera(ei, cam)
            n = len(frames)
            images[cam] = [center_crop_np(frames[min(i, n - 1)], CROP) for i in idxs]
        state_win = np.stack([state[min(i, len(state) - 1)] for i in idxs], axis=0)  # [4,16]
        return images, state_win

    def build_target(self, action, step, span, chunk_len=16):
        """x1 [L,60] and loss_mask [L,60] for a chunk starting at `step`.
        Active dims 0-11; steps within [span_start, span_end) unmasked."""
        n = len(action)
        x1 = np.zeros((chunk_len, ACTION_FULL), dtype=np.float32)
        mask = np.zeros((chunk_len, ACTION_FULL), dtype=np.float32)
        s0, s1 = span
        for j in range(chunk_len):
            t = step + j
            if t >= n:
                break
            x1[j, :ACTION_REAL] = action[t, :ACTION_REAL]
            if s0 <= t < s1:
                mask[j, :ACTION_REAL] = 1.0
        return x1, mask


class SFTClient:
    """Builds processor inputs + talks to the trainer server sft_* ops."""

    def __init__(self, model_path, host="127.0.0.1", port=10088):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.host, self.port = host, port
        self._connect()

    def _connect(self):
        import time
        for _ in range(600):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                return
            except OSError:
                time.sleep(1)
        raise ConnectionError(f"cannot reach trainer {self.host}:{self.port}")

    def _rpc(self, req):
        blob = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(blob)) + blob)
        ln = self.sock.recv(4)
        n = struct.unpack(">I", ln)[0]
        data = b""
        while len(data) < n:
            data += self.sock.recv(n - len(data))
        return pickle.loads(data)

    def build_inputs(self, images, state_win, instruction):
        state = np.zeros((1, state_win.shape[0], STATE_DIM), dtype=np.float32)
        state[0, :, :state_win.shape[-1]] = state_win
        videos = {k: images[k] for k in CAMERA_KEYS}
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Left camera: "}, {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                {"type": "text", "text": "\nRight camera: "}, {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                {"type": "text", "text": "\nWrist camera: "}, {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            do_resize=False, state=state, robot_type=ROBOT_TYPE)
        d = dict(inputs)
        d["task_id"] = ROBOT_TYPE
        return d

    def sft_update(self, inputs, x1, loss_mask, skill, seed=None, sft_steps=1):
        import torch
        return self._rpc({"op": "sft_update", "inputs": inputs,
                          "x1": torch.from_numpy(x1[None]), "loss_mask": torch.from_numpy(loss_mask[None]),
                          "skill": skill, "seed": seed, "sft_steps": sft_steps})

    def sft_val(self, inputs, x1, loss_mask, skill, seed=0, mc=8):
        import torch
        return self._rpc({"op": "sft_val", "inputs": inputs,
                         "x1": torch.from_numpy(x1[None]), "loss_mask": torch.from_numpy(loss_mask[None]),
                         "skill": skill, "seed": seed, "mc": mc})

    def rpc(self, req):
        return self._rpc(req)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
