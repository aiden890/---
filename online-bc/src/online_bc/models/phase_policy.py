"""Explicit request routing between frozen base and adapted inference states."""


class PhasePolicy:
    def __init__(self, backend, base_snapshot):
        self.backend = backend
        self.base_snapshot = base_snapshot
        self.adapter_snapshot = None

    def load(self, path, *, load_optimizer=True):
        if load_optimizer:
            self.backend.load(path)
        else:
            self.backend.load(path, load_optimizer=False)
        self.adapter_snapshot = None

    def infer(self, sample, seed=0, variant="standard", phase=None):
        if variant not in ("standard", "base_prefix"):
            raise ValueError(f"Unsupported policy variant: {variant}")
        if variant == "base_prefix" and phase not in ("prefix", "cup_placement"):
            raise ValueError("base_prefix requires an explicit valid skill phase")
        use_base = variant == "base_prefix" and phase == "prefix"
        self.backend.infer_fn = self.base_snapshot if use_base else self.adapter_snapshot
        actions = self.backend.infer(sample, seed)
        if not use_base:
            self.adapter_snapshot = self.backend.infer_fn
        return actions, use_base
