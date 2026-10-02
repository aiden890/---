import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from online_bc.models.pi05_backend import Backend


class InferenceCheckpointTest(unittest.TestCase):
    def test_adapter_only_contract_and_training_restore(self):
        backend = Backend.__new__(Backend)
        state = types.SimpleNamespace(replace_by_pure_dict=lambda value: setattr(backend, "weights", value))
        backend.nnx = types.SimpleNamespace(state=lambda *args: state, update=lambda *args: None)
        backend.jnp = types.SimpleNamespace(asarray=np.asarray)
        backend.jax = types.SimpleNamespace(tree=types.SimpleNamespace(map=lambda fn, x: fn(x)))
        backend.model, backend.filter = object(), object()
        backend.opt_state = original_optimizer = object()
        backend.inference_only = False
        backend.infer_fn = original_cache = object()
        flax = types.ModuleType("flax")
        traverse = types.ModuleType("flax.traverse_util")
        traverse.unflatten_dict = lambda flat, sep: flat
        flax.traverse_util = traverse
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"flax": flax, "flax.traverse_util": traverse}
        ):
            path = Path(tmp)
            np.savez(path / "pi05-lora.npz", **{"layer/lora_b": np.ones((2, 2))})
            # Training restore remains strict and fails before changing parameters/cache.
            with self.assertRaises(FileNotFoundError):
                backend.load(path)
            self.assertFalse(hasattr(backend, "weights"))
            self.assertIs(backend.infer_fn, original_cache)
            backend.load(path, load_optimizer=False)
            self.assertIs(backend.opt_state, original_optimizer)
            self.assertIsNone(backend.infer_fn)
            np.testing.assert_array_equal(backend.weights["layer/lora_b"], np.ones((2, 2)))
            with self.assertRaisesRegex(RuntimeError, "optimizer"):
                backend.update_batch([])
            with self.assertRaisesRegex(RuntimeError, "training checkpoint"):
                backend.save(path, 50)
            # A full restore clears inference-only mode and restores the saved counter.
            with (path / "pi05-optimizer.pkl").open("wb") as handle:
                pickle.dump({"optimizer": np.array(500)}, handle)
            backend.load(path)
            self.assertFalse(backend.inference_only)
            self.assertEqual(int(backend.opt_state), 500)


if __name__ == "__main__":
    unittest.main()
