from pathlib import Path
import numpy as np
import torch


class Backend:
    def __init__(self, checkpoint, lr=1e-4):
        from gr00t.model.policy import Gr00tPolicy
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from peft import LoraConfig, get_peft_model

        self.dc = DATA_CONFIG_MAP["panda_omron"]
        self.policy = Gr00tPolicy(
            checkpoint, "new_embodiment", self.dc.modality_config(), self.dc.transform()
        )
        model = self.policy.model
        # Native Beta sampling is fp32; bf16 checkpoint loading otherwise changes
        # the default distribution dtype, unsupported by torch's Dirichlet kernel.
        h = model.action_head
        h.beta_dist = torch.distributions.Beta(
            torch.tensor(h.config.noise_beta_alpha, dtype=torch.float32),
            torch.tensor(h.config.noise_beta_beta, dtype=torch.float32),
        )
        targets = [
            n
            for n, m in model.named_modules()
            if n.startswith("action_head.model.")
            and isinstance(m, torch.nn.Linear)
            and n.endswith(("to_q", "to_k", "to_v"))
        ]
        assert targets, "GR00T action-head LoRA targets not found"
        model.requires_grad_(False)
        self.model = get_peft_model(
            model,
            LoraConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules=targets, bias="none"),
        )
        self.policy.model = self.model
        self.policy.modality_transform.train()
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)

    def update(self, sample, seed=0):
        torch.manual_seed(seed)
        obs = sample["obs"]
        dc = self.dc
        raw = {k: obs[k][None].copy() for k in dc.video_keys + dc.state_keys}
        raw[dc.language_keys[0]] = np.array([sample["prompt"]])
        slices = [(0, 3), (3, 6), (6, 7), (7, 11), (11, 12)]
        for key, (a, b) in zip(dc.action_keys, slices):
            raw[key] = sample["actions"][:16, a:b].copy()
        transformed = self.policy.apply_transforms(raw)
        from gr00t.model.transforms import collate

        transform = self.policy.modality_transform.transforms[-1]
        batch = collate([transformed], transform.eagle_processor)
        batch["action_mask"] &= torch.from_numpy(sample["valid"][:16])[None, :, None]
        self.opt.zero_grad(set_to_none=True)
        self.model.eval()  # deterministic backbone; gradients remain enabled for LoRA
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = self.model(batch)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 0.5
        )
        self.opt.step()
        return dict(
            loss=float(loss.detach()),
            grad_norm=float(gn),
            peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
        )

    def save(self, path, step):
        self.model.save_pretrained(Path(path) / "groot-adapter")
        torch.save(
            dict(optimizer=self.opt.state_dict(), step=step), Path(path) / "groot-optimizer.pt"
        )

    def load(self, path):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(
            self.model, load_file(str(Path(path) / "groot-adapter/adapter_model.safetensors"))
        )
        self.opt.load_state_dict(
            torch.load(Path(path) / "groot-optimizer.pt", map_location="cuda")["optimizer"]
        )

    def parameters(self):
        return {
            n: p.detach().cpu().float().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def infer(self, sample, seed=0):
        torch.manual_seed(seed)
        dc = self.dc
        obs = sample["obs"]
        raw = {k: obs[k][None].copy() for k in dc.video_keys + dc.state_keys}
        raw[dc.language_keys[0]] = np.array([sample["prompt"]])
        self.policy.modality_transform.eval()
        result = self.policy.get_action(raw)
        self.policy.modality_transform.train()
        return np.concatenate([np.asarray(result[k]) for k in dc.action_keys], axis=-1)
