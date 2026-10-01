# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Extracted unchanged from RLinf 034579cbfc4643f72c184ffc06458f788c09b3e6
# rlinf/data/storage/replay/buffer.py: TrajectoryCache
from typing import Optional
import torch

class TrajectoryCache:
    """FIFO cache for storing flattened trajectories."""

    def __init__(self, max_size: int = 5):
        self.cache: dict[int, int] = {}
        self.max_size = max_size
        self._buffer: Optional[dict] = None
        self._traj_num_samples: Optional[int] = None
        self._traj_key_lengths: dict[int, dict] = {}
        self._last_slot = 0
        self._slot_to_id: dict[int, int] = {}

    def _get_key_lengths(self, trajectory: dict) -> dict:
        lengths: dict = {}
        has_tensor = False
        for key, value in trajectory.items():
            if isinstance(value, torch.Tensor):
                lengths[key] = int(value.shape[0])
                has_tensor = True
            elif isinstance(value, dict):
                nested = self._get_key_lengths(value)
                if nested:
                    lengths[key] = nested
                    has_tensor = True
        if not has_tensor:
            raise ValueError("Trajectory contains no tensor fields.")
        return lengths

    def _get_max_num_samples(self, lengths: dict) -> int:
        max_len = 0
        for value in lengths.values():
            if isinstance(value, dict):
                max_len = max(max_len, self._get_max_num_samples(value))
            else:
                max_len = max(max_len, int(value))
        return max_len

    def _alloc_buffer_like(self, trajectory: dict, total_samples: int) -> dict:
        buffer: dict = {}
        for key, value in trajectory.items():
            if isinstance(value, torch.Tensor):
                shape = (total_samples, *value.shape[1:])
                buffer[key] = torch.empty(shape, dtype=value.dtype, device=value.device)
            elif isinstance(value, dict):
                buffer[key] = self._alloc_buffer_like(value, total_samples)
            else:
                buffer[key] = value
        return buffer

    def _insert_into_buffer(self, trajectory: dict, buffer: dict, start: int) -> None:
        for key, value in trajectory.items():
            if isinstance(value, torch.Tensor):
                end = start + value.shape[0]
                buffer[key][start:end] = value
            elif isinstance(value, dict):
                self._insert_into_buffer(value, buffer[key], start)
            else:
                buffer[key] = value

    def _slice_from_buffer(self, buffer: dict, slc: slice) -> dict:
        sliced: dict = {}
        for key, value in buffer.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[slc]
            elif isinstance(value, dict):
                sliced[key] = self._slice_from_buffer(value, slc)
            else:
                sliced[key] = value
        return sliced

    def _slice_from_buffer_with_lengths(
        self, buffer: dict, start: int, lengths: Optional[dict]
    ) -> dict:
        sliced: dict = {}
        for key, value in buffer.items():
            if isinstance(value, torch.Tensor):
                if lengths is None or key not in lengths:
                    end = start + self._traj_num_samples
                else:
                    end = start + int(lengths[key])
                sliced[key] = value[start:end]
            elif isinstance(value, dict):
                nested_lengths = None if lengths is None else lengths.get(key, None)
                sliced[key] = self._slice_from_buffer_with_lengths(
                    value, start, nested_lengths
                )
            else:
                sliced[key] = value
        return sliced

    def _copy_buffer_slice(
        self,
        src_buffer: dict,
        dst_buffer: dict,
        src_slc: slice,
        dst_slc: slice,
    ) -> None:
        for key, value in src_buffer.items():
            if isinstance(value, torch.Tensor):
                dst_buffer[key][dst_slc] = value[src_slc]
            elif isinstance(value, dict):
                self._copy_buffer_slice(value, dst_buffer[key], src_slc, dst_slc)
            else:
                dst_buffer[key] = value

    def _ensure_capacity(self, max_num_samples: int, trajectory: dict) -> None:
        if self._traj_num_samples is None:
            self._traj_num_samples = max_num_samples
            total_samples = self.max_size * self._traj_num_samples
            self._buffer = self._alloc_buffer_like(trajectory, total_samples)
            return
        if max_num_samples <= self._traj_num_samples:
            return

        # Grow slot length only when needed.
        old_slot_len = self._traj_num_samples
        new_slot_len = max_num_samples
        new_total_samples = self.max_size * new_slot_len
        new_buffer = self._alloc_buffer_like(trajectory, new_total_samples)

        if self._buffer is not None:
            for slot in self.cache.values():
                src_start = slot * old_slot_len
                src_end = src_start + old_slot_len
                dst_start = slot * new_slot_len
                dst_end = dst_start + old_slot_len
                self._copy_buffer_slice(
                    self._buffer,
                    new_buffer,
                    slice(src_start, src_end),
                    slice(dst_start, dst_end),
                )

        self._buffer = new_buffer
        self._traj_num_samples = new_slot_len

    def get(self, trajectory_id: int) -> Optional[dict]:
        if trajectory_id not in self.cache or self._buffer is None:
            return None
        slot = self.cache[trajectory_id]
        start = slot * self._traj_num_samples
        lengths = self._traj_key_lengths.get(trajectory_id)
        return self._slice_from_buffer_with_lengths(self._buffer, start, lengths)

    def get_buffer(self) -> Optional[dict]:
        return self._buffer

    def get_slot_length(self) -> Optional[int]:
        return self._traj_num_samples

    def put(self, trajectory_id: int, trajectory: dict):
        key_lengths = self._get_key_lengths(trajectory)
        max_num_samples = self._get_max_num_samples(key_lengths)
        self._ensure_capacity(max_num_samples, trajectory)

        if trajectory_id in self.cache:
            slot = self.cache[trajectory_id]
        else:
            slot = self._last_slot
            if slot in self._slot_to_id:
                evict_id = self._slot_to_id[slot]
                if evict_id in self.cache:
                    self.cache.pop(evict_id, None)
                self._traj_key_lengths.pop(evict_id, None)
            self._slot_to_id[slot] = trajectory_id
            self.cache[trajectory_id] = slot
            self._last_slot = (self._last_slot + 1) % self.max_size

        start = slot * self._traj_num_samples
        self._insert_into_buffer(trajectory, self._buffer, start)
        self._traj_key_lengths[trajectory_id] = key_lengths

    def clear(self):
        self.cache.clear()
        self._buffer = None
        self._traj_num_samples = None
        self._traj_key_lengths.clear()
        self._last_slot = 0
        self._slot_to_id.clear()


