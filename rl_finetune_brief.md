# RL Fine-Tuning of a Rectified-Flow VLA with GRPO: Research Brief

**Target setup:** Xiaomi-Robotics-1 (MiBoT) — Qwen3-VL-4B backbone (frozen) + DiT action expert (1024 hidden, 36 layers, ~615M) using rectified-flow ODE (explicit Euler, 5 steps) → action chunk [L≈30, dim=12]. SDE conversion already built. Task: RoboCasa CloseBlenderLid, 3 skills (GRASP / MOVE_HOLDING / PLACE), group_size=4 on RTX 3090, simulator predicate reward, SGD + gradient checkpointing only.

---

## 1. Prior Art: RL Fine-Tuning of Flow / Diffusion Policies

### 1.1 DDPO — Denoising Diffusion Policy Optimization
**Paper:** Black et al., 2023. "Training Diffusion Models with Reinforcement Learning." arXiv:2305.13301  
**Core trick:** Frame the T-step DDPM denoising chain as a multi-step MDP. Each denoising transition `q(x_{t-1}|x_t)` is Gaussian with closed-form log-prob, making the policy gradient exact at every denoising step. Two variants: `DDPO_SF` (score-function / REINFORCE) and `DDPO_IS` (importance-sampling ratio). `DDPO_IS` is equivalent to PPO without the clip.  
**ODE→SDE conversion?** No — DDPM is already stochastic. The Gaussian transition is inherent to the DDPM forward process.  
**Key hyperparameters:** T=50 denoising steps (DDPM scheduler), batch of 128 images, 4 gradient updates per batch, clip ratio ε=0.1, LR≈1e-5 (Adam).

### 1.2 DPOK — Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models  
**Paper:** Fan et al., 2023. arXiv:2305.16381  
**Core trick:** Online RL on DDPM with KL regularization toward a frozen reference policy to prevent reward hacking. Adds a KL penalty `β·KL(π||π_ref)` to the per-denoising-step reward.  
**ODE→SDE?** No — inherits DDPM stochasticity.  
**Key hyperparameters:** β (KL) ∈ [0.001, 0.1], PPO clip ε=0.2, 50 denoising steps.

### 1.3 DPPO — Diffusion Policy Policy Optimization *(most directly relevant robotics prior)*  
**Paper:** Ren et al., 2024. "Diffusion Policy Policy Optimization." arXiv:2409.00588. **Published at ICLR 2025.**  
**Core trick:** Treats the K-step DDIM/DDPM denoising chain as a "Diffusion MDP" (inner MDP), nested within the environment MDP (outer MDP). Each step's Gaussian transition `p_θ(x_{k-1}|x_k)` gives exact log-probs. Fine-tunes the whole chain with PPO (clipped ratio + GAE). Key design decisions: (a) reduce denoising steps during training (e.g., 10→5 or 5→3) for efficiency; (b) freeze early denoising steps and only train the last few; (c) clip ratio ε=0.2; (d) per-step advantage = environment reward (no discounting within denoising chain — the reward arrives only after the full chunk).  
**ODE→SDE?** Not explicitly — uses DDPM's inherent stochasticity. For DDIM (deterministic), they add a small η>0 to re-introduce noise (η∈[0,1] in DDIM parameterization).  
**Key hyperparameters:** clip ε=0.2, LR 1e-4 to 1e-5 (Adam), K_train=5 denoising steps during RL (vs K_infer=10–100), sparse task reward, gradient checkpointing, LoRA not used — full fine-tune of noise network only.  
**Robotics results:** First RL algorithm to solve Transport (robomimic) >50% success; Furniture-Bench from 1%→86%.

### 1.4 Flow-GRPO — Flow Matching + GRPO for Image Generation *(closest match to our SDE strategy)*  
**Paper:** Liu et al., 2025. "Flow-GRPO: Training Flow Matching Models via Online RL." arXiv:2505.05470. **NeurIPS 2025.**  
**Code:** https://github.com/yifan123/flow_grpo  
**Core trick:** Two innovations: (1) **ODE-to-SDE conversion** — adds Gaussian noise `σ_t·ε` to each flow step, preserving marginals while making the process stochastic and giving exact per-step Gaussian log-probs. Specifically: `σ_t = a·√(t/(1-t))`, where `a` is a scalar hyperparameter controlling noise magnitude. (2) **Denoising Reduction** — use fewer steps during training (e.g., 5–10 steps) but keep the original count (e.g., 28) at inference. Then applies GRPO: sample a group of G completions per prompt, compute group-normalized advantage `A_i = (r_i - mean(r))/std(r)`, and optimize with clipped importance ratio.  
**ODE→SDE?** Yes — **this is exactly our strategy.** They verify that the SDE recovers the original ODE marginals.  
**Key hyperparameters:** G=16 (group size), ε=0.2 (clip), no explicit KL in the base version (optional), LR=1e-5 to 5e-6 (Adam), a (noise scale) ablated — too low → insufficient exploration, too high → distribution shift. Typical: a≈0.1–1.0.  
**Note:** Their context is image generation, not robotics — action space is image latent, not continuous robot actions.

### 1.5 ReinFlow — Flow Matching Policy RL for Robotics *(closest technical match)*  
**Paper:** Zhang et al., 2025. "ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning." arXiv:2505.22094. **NeurIPS 2025.**  
**Code:** https://github.com/ReinFlow/ReinFlow (MIT license; supports Pi0, Pi0.5, GR00T-N1.5)  
**Core trick:** Different from fixed-σ SDE — trains a **learnable noise injection network** (a small MLP) alongside the flow policy. This noise net outputs per-step σ_k, making the noise level adaptive rather than a fixed hyperparameter. The resulting transitions are still Gaussian with exact log-probs. Uses PPO (not GRPO) with clipped ratio.  
**ODE→SDE?** Yes — via learnable noise injection rather than fixed η. They show this is more robust to discretization error than fixed-η approaches, especially at very few steps (K=1–4).  
**Key hyperparameters:** K_train=4 steps (Rectified Flow), LR=3e-5 (Adam), clip ε=0.2, entropy bonus, no KL toward reference. Group size not applicable (uses PPO with GAE).  
**Results:** +135% reward on legged locomotion (Rectified Flow), +40% on visual manipulation; 82.6% faster than DPPO wall time.  
**Critical note:** ReinFlow explicitly notes that fixed-η (as in our design) works but the noise level must be tuned carefully to avoid (a) near-zero variance → gradient vanishing and (b) high variance → importance ratio explosion. Their learnable noise net sidesteps this.

### 1.6 πRL — Online RL for Flow-Based VLAs *(directly comparable architecture)*  
**Paper:** Chen et al., 2025. "πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models." arXiv:2510.25889. (v3 Jan 2026)  
**Core trick:** Two algorithms: **Flow-SDE** (ODE-to-SDE, exactly as we implement) and **Flow-Noise** (learnable noise network like ReinFlow). Targets π0/π0.5 (flow-based VLAs). **Flow-SDE** applies ODE→SDE conversion + two-layer MDP (inner: denoising steps; outer: env steps). Evaluates on LIBERO, MetaWorld, ManiSkill/Simpler benchmarks.  
**ODE→SDE?** Yes — explicitly validates the two-layer MDP formulation for flow VLAs.  
**Key findings:** Flow-SDE works but is sensitive to η choice; Flow-Noise (learnable) is more stable. Both outperform SFT baseline.

### 1.7 Smart-GRPO / SuperFlow  
**Papers:** Smart-GRPO: arXiv:2510.02654; SuperFlow: arXiv:2512.17951  
**Core trick:** Improvements over Flow-GRPO for image gen. Smart-GRPO selects high-reward noise seeds as curriculum. SuperFlow adds per-step advantage credit assignment (vs trajectory-level). Both confirm that (a) GRPO with flow-SDE works, (b) trajectory-level advantage (assigning the same `A` to all K denoising steps) introduces bias — SuperFlow proposes cosine-similarity-based per-step credit, though gains are marginal.  
**Relevance:** Confirms the credit assignment bias of using trajectory reward at every denoising step. For robotics with sparse reward, this bias is unavoidable without a learned critic.

### 1.8 VLA-Specific RL Methods

| Method | Base VLA | RL Approach | ODE→SDE? |
|--------|----------|-------------|-----------|
| **GRAPE** (arXiv:2411.19309) | OpenVLA (autoregressive) | Trajectory preference optimization (TPO), iterative DPO-style | N/A |
| **ConRFT** (arXiv:2502.05450) | pi0 (flow-based) | Offline BC + online RL with consistency policy; human-in-loop | No — uses consistency distillation |
| **TGRPO** (arXiv:2506.08440) | OpenVLA-OFT (autoregressive) | GRPO with trajectory-level + step-level advantage, dense LLM-generated reward | N/A |
| **RIPT-VLA** (arXiv:2505.17016) | Autoregressive VLA | Online RL with sparse binary reward; simple REINFORCE-style | N/A |
| **RLinf-VLA** (arXiv:2510.06710) | OpenVLA, OpenVLA-OFT | Unified PPO/GRPO framework; parallel sim | N/A |

**Key gap:** For *flow-based* VLAs (like MiBoT), only ReinFlow (arXiv:2505.22094) and πRL (arXiv:2510.25889) are directly applicable. Both validate the flow→SDE approach for robotics. GRAPE and TGRPO target autoregressive VLAs with discrete action tokens; their log-prob is trivially tractable.

---

## 2. Soundness of Our Approach: Pitfalls and Analysis

### 2.1 ODE→SDE + Gaussian Log-Probs: Is It Sound?
**Yes — with caveats.** Flow-GRPO (arXiv:2505.05470) and πRL (arXiv:2510.25889) have both published proofs that Euler-Maruyama SDE conversion preserves marginals at each timestep when `η = a·√(dt)` for an appropriate `a`. Our formula `x_{k+1} = μ_k + η·√(dt)·ε_k` with log-prob `log N(x_{k+1}; μ_k, η²·dt·I)` is mathematically correct.

The per-step log-prob for K=5 steps and action dim=12 is:
```
log p(x_1,...,x_K | x_0) = Σ_{k=0}^{K-1} log N(x_{k+1}; μ_k, η²·dt·I)
                         = -K·d/2·log(2π·η²·dt) - 1/(2η²·dt) · Σ||x_{k+1} - μ_k||²
```
where d=12 (action_dim per chunk step), K=5 (flow steps), and the chunk length L≈30 means this total log-prob is **summed over L env-steps-worth of actions** (or applied per-chunk depending on your action-chunking granularity).

**Known pitfalls:**

1. **Importance ratio explosion with K=5 steps.** The IS ratio in GRPO is `π_θ(τ)/π_θ_old(τ)` where τ is the full denoising trajectory. Since `log π = Σ_k log p_k`, the ratio is a **product of K per-step ratios**:  
   `r = exp(Σ_k [log p_θ(x_{k+1}|x_k) - log p_θ_old(x_{k+1}|x_k)])`  
   With K=5, even small per-step log-prob differences compound. This is the most dangerous failure mode. **Mitigation:** (a) tight clip ε=0.1–0.15 (tighter than the LLM standard of 0.2); (b) limit number of optimization epochs per batch to 1; (c) discard rollouts where ratio > 5–10 (DPPO uses this heuristic); (d) ReinFlow mitigates by making η adaptive.

2. **Log-prob variance at few steps.** With K=5, the Monte Carlo estimate of the log-prob has high variance per rollout (less averaging than K=50). The GRPO advantage normalization helps but doesn't eliminate this. **Mitigation:** normalize advantages within the group *and* across chunks of the same rollout; ensure group_size≥8 if hardware allows (currently 4 — this is tight, see §3).

3. **η choice.** Too small: near-deterministic → log-prob nearly infinite when on-policy, numerical instability when off-policy. Too large: marginal distribution diverges from the pretrained ODE, destroying the initialization. Flow-GRPO ablates `a` in their parameterization `σ_t = a·√(t/(1-t))`; their finding: performance peaks at moderate `a` (order of magnitude ~0.1–0.5 of the data std). For our setup, **start with `η ≈ 0.1·(action_std)`** of the BC dataset (the noise should make adjacent rollout samples distinguishable but not drastically different from the ODE output).

4. **Masking to executed timesteps.** DPPO recommends only backpropagating through denoising steps that are **actually executed** in the environment (not speculative). For action chunking (L=30 actions from one denoising pass), all K=5 denoising steps contribute to the chunk, so all 5 should be included in the log-prob. This is already what we compute.

5. **KL regularization.** Flow-GRPO offers both KL-free and KL-penalized variants. KL-free reaches higher reward but risks reward hacking. DPOK (arXiv:2305.16381) shows KL is critical when the reward model is imperfect. Since our reward is simulator ground truth (not a proxy), KL-free is defensible — but for a flow action-expert with only 5 steps, the policy can collapse quickly. **Recommendation:** include a light KL penalty (β≈0.001–0.01) or a reference policy log-prob auxiliary loss for the first 20–50 iterations to stabilize.

6. **Advantage normalization.** Within group_size=4, advantage normalization `A_i = (r_i - mean)/std` with n=4 is **very noisy** (std estimated from 4 samples). The advantage can be dominated by a single outlier reward. **Mitigation:** (a) increase group_size if possible; (b) clip advantages to [-3σ, +3σ]; (c) use running mean/std across multiple iterations rather than per-batch.

7. **Off-policy staleness.** GRPO uses the "old policy" (from rollout time) as the reference for IS ratios. With SGD (no momentum), each gradient step moves the policy further from the rollout policy. With only 4 rollouts per update, staleness accumulates quickly. **Recommendation:** use **1 gradient update per batch** (no epoch reuse), and regenerate rollouts every iteration. DPPO uses 1–4 epochs; for our setup (small group_size, SGD), 1 epoch is safer.

---

## 3. Recommended Hyperparameters

| Parameter | Published range | Recommendation for our setup |
|-----------|----------------|-------------------------------|
| **Group size G** | 8–64 (Flow-GRPO), 4–16 (robotics) | **4** (hardware-limited); compensate with gradient clipping and advantage clipping |
| **Clip ε** | 0.2 (PPO standard), 0.1–0.2 (DPPO) | **0.1–0.15** (tighter, because K=5 product amplifies ratio) |
| **Learning rate** | 1e-5 to 1e-4 (Adam, Flow-GRPO/DPPO) | **1e-3 to 5e-3 with SGD** (see note below) |
| **SGD vs Adam** | SGD matches AdamW in RLVR (arXiv:2602.07729) | **SGD without momentum** — Mukherjee et al. (2025) show momentum hurts in RLVR; consistent with hardware constraint |
| **η (noise scale)** | Ablated in Flow-GRPO: a≈0.1–1.0 | **Start η=0.1·action_std; grid-search {0.05, 0.1, 0.3}** |
| **K (flow steps, training)** | 5–10 (Flow-GRPO uses denoising reduction: 28→5) | **5** (same as inference; no reduction needed since inference is already 5) |
| **K (flow steps, inference)** | Same or higher | **5** (keep consistent to avoid distribution shift) |
| **KL coefficient β** | 0 (Flow-GRPO default) to 0.01 (DPOK) | **β=0.001–0.005 for first 30 iters, then 0** |
| **Rollouts per update** | 1 epoch (DDPO), 1–4 epochs (DPPO) | **1 epoch** (SGD + small group → staleness risk) |
| **Gradient clip norm** | 1.0 (standard) | **0.5** (tighter, to stabilize SGD with clipped ratio) |
| **Reward shaping** | Dense ≫ sparse for <50 iters (TGRPO finding) | **Per-skill milestone rewards + terminal sparse** |

**SGD LR note:** Mukherjee et al. (arXiv:2602.07729, "Do We Need Adam? Surprisingly Strong and Sparse Updates with SGD in RLVR", 2025) demonstrate that SGD (no momentum) matches or exceeds AdamW for RLVR across GRPO and PPO on Qwen and Llama models. For Qwen-3 8B with Adam LR=1e-6, the equivalent SGD LR is larger (typically 10×–100×), as SGD lacks per-parameter normalization. For LoRA parameters with rank r=8–16, safe SGD LR is **1e-3 to 5e-3** (pilot: try 1e-3 with cosine decay).

---

## 4. Per-Skill LoRA Adapter Architecture

### 4.1 Prior Art for Per-Task LoRA in Robot Policies

**CORAL** (Luo et al., 2025. "CORAL: Scalable Multi-Task Robot Learning via LoRA Experts." arXiv:2603.09298):  
The closest direct prior. Freezes a VLA backbone, attaches **one LoRA expert per task**, and routes at inference based on language instruction. Validated on LIBERO, WidowX, Galaxea R1. Prevents gradient interference between tasks. Key result: per-task LoRA outperforms joint fine-tuning significantly on fine-grained task distinctions. **This is strong validation for per-skill LoRA.**  
Their setup differs (tasks, not skills/options), but the principle is identical: strict parameter isolation → no negative transfer.

**Mixture of LoRA Experts (MoLE)** (OpenReview, 2024):  
Multiple LoRA experts with soft routing (attention-based gating). More flexible but requires learning a gating mechanism. For only 3 skills, hard routing (as in CORAL) is simpler and equally effective.

**Skill-to-LoRA** (arXiv:2606.16769, 2025):  
Assigns a separate LoRA per skill for LLM agents; activates the appropriate LoRA based on detected skill. Exact match to our per-skill (GRASP / MOVE_HOLDING / PLACE) design.

**Memory-Efficient Policy Libraries** (arXiv:2606.25700, 2025):  
LoRA-based policy libraries; notes 20–160× memory savings over full fine-tuning per policy. For 3 skills × rank-16 LoRA on a 615M DiT, total overhead is negligible (~3M params × 3 = ~9M parameters).

**GRAPE** (arXiv:2411.19309):  
Uses a single LoRA for all tasks with preference optimization. Does **not** do per-skill LoRA — works for multi-task but does not address per-skill specialization.

### 4.2 Risk Assessment

**Is per-skill LoRA on a flow action-expert unusual or risky?**

**Partially unusual, but theoretically sound.** The risks are:

1. **No published precedent for per-skill LoRA on a DiT action expert with flow-RL.** CORAL does per-task LoRA on VLA backbones (the language model part), not on a DiT action head. **This is an extrapolation.** ReinFlow and πRL fine-tune the entire flow policy without per-skill separation.

2. **Segment-only training.** Training each LoRA only on the action segment executed under that skill (e.g., GRASP LoRA trained only on GRASP-phase transitions) is correct in principle — this is exact option-MDP training. The risk is that **rare skill transitions and short segments** (especially PLACE, which may be brief) give the PLACE LoRA very few gradient steps. With group_size=4 and only partial success reaching the PLACE phase, PLACE LoRA may be severely data-starved.

3. **Shared DiT body.** With all three LoRAs sharing the frozen DiT body, the initialization point for each skill is the same pretrained flow policy, which is good. Cross-skill interference is zero by construction (separate parameters).

4. **Termination head.** A separate termination head per skill (deciding when to transition GRASP→MOVE_HOLDING etc.) is reasonable. This could be a 2-class linear head on top of the DiT's hidden state at each action step. However, training this head requires **skill-transition reward signal** — ensure your FSM exports ground-truth skill transition events as labels or intermediate rewards.

### 4.3 Closest Validated Alternative (If Per-Skill LoRA Fails)

**Single shared LoRA + skill-ID conditioning token** — the safer alternative:
- One LoRA (rank 16–32) on the DiT action expert, shared across all skills.
- Prepend a skill-ID embedding (learned or one-hot projected) to the DiT's conditioning input (concat with language embedding or add as cross-attention token).
- This is analogous to **FiLM conditioning** (Feature-wise Linear Modulation, Perez et al., 2018) and task-token conditioning in multi-task diffusion policies.
- Validated: DPPO fine-tunes a single policy across multiple tasks; T-GRPO (arXiv:2506.08440) uses a single shared policy with task instruction.
- **Advantage:** All skill data trains the same LoRA → no data-starvation problem; skill specialization comes from conditioning.
- **Disadvantage:** Possible inter-skill interference (but with rank ≤32 and frozen backbone, this is minor).

**Recommendation:** Start with the **single shared LoRA + skill-ID token** approach for the pilot (simpler, proven), then migrate to per-skill LoRA once the signal is confirmed. If going per-skill immediately, **ensure minimum 50 gradient updates per skill per LoRA** before evaluating success rates.

---

## 5. Concrete Recommended Recipe for Our Setup

### 5.1 What to Train

| Component | Frozen? | Adapter |
|-----------|---------|---------|
| Qwen3-VL-4B backbone | ✅ Frozen | None |
| DiT action expert (shared body) | ✅ Frozen | None |
| DiT — per-skill LoRA (or single shared LoRA) | ❌ Trainable | LoRA rank 8–16, α=32 |
| Skill termination head (linear, per-skill) | ❌ Trainable | 128→2 MLP |

**Why freeze the DiT body:** The DiT is 615M params. Even with gradient checkpointing, training it with SGD alongside a resident inference server on a single 3090 (24GB) will OOM. LoRA on the DiT's attention QKV or DiT block cross-attention is ~3–6M params — tractable.

**LoRA injection point:** Inject into the **DiT's self-attention Q, K, V projection matrices** (standard LoRA) and optionally the **cross-attention** between language conditioning and action tokens. The MoE-LoRA literature (arXiv:2501.15103) suggests Q+V is sufficient; K can be frozen.

**LoRA rank:** rank=8 for pilot (3M params/skill); rank=16 if more capacity needed. α=32 (scaling factor = α/r = 4× the update magnitude, standard initialization).

### 5.2 SDE Hyperparameters

- **η (fixed):** Set `η = 0.1`. This gives `std(noise) = η·√(dt)·I` per flow step. With dt = 1/K = 0.2 for K=5 steps, noise std per step = 0.1·√0.2 ≈ 0.045 in action space. Check that this is ~5–15% of the BC policy's per-step action variance — if smaller, exploration is insufficient; if larger, rollouts will diverge from the pretrained distribution.  
- **Verify η=0 determinism:** Already confirmed in your setup. Run a sanity check: `|x_rl_η0 - x_bc| < 1e-5` after every major code change.
- **Consider upgrading to ReinFlow-style learnable η** after the pilot: this is the most robust approach from the literature. The noise network is a 2-layer MLP mapping (t_k, DiT_hidden) → log_σ_k.

### 5.3 GRPO Objective

```
L_GRPO = E_{i=1}^{G} [ min(r_i · A_i, clip(r_i, 1-ε, 1+ε) · A_i) ]
       - β · KL(π_θ || π_ref)

where:
  r_i = exp(log π_θ(τ_i) - log π_θ_old(τ_i))   # product over K=5 SDE steps
  A_i = (reward_i - mean(reward_{1..G})) / (std(reward_{1..G}) + 1e-8)
  ε = 0.1  # clip range (tighter than default 0.2)
  β = 0.005 (first 30 iters), then 0
```

**Per-skill masking:** Compute `log π` only over the K=5 SDE steps of the **action chunks produced during that skill's active phase**. E.g., GRASP LoRA gradient flows only through chunks emitted while the FSM was in GRASP state.

### 5.4 Reward Shaping

```
R_total = R_terminal + Σ_skill R_milestone_skill

R_terminal = +1.0  (simulator predicate: lid closed)
R_milestone_GRASP = +0.3  (grasp predicate satisfied)
R_milestone_MOVE  = +0.3  (above target position predicate)  
R_milestone_PLACE = +0.4  (lid-on-blender predicate)
```

**Cautions:**
1. **No VLM reward.** This is correct — VLM reward models introduce proxy reward hacking. Simulator predicates are verifiable, consistent with RIPT-VLA (arXiv:2505.17016) which shows 97% success with binary success reward only.
2. **Avoid shaping that dominates the terminal reward.** If milestone rewards > terminal reward, the policy may learn to oscillate at the milestone boundary. Keep `R_terminal ≥ R_milestone_sum`.
3. **Skill-conditioned reward assignment:** Attribute rewards to the LoRA that was active during the transition producing them. Don't backpropagate PLACE reward through GRASP LoRA.
4. **Reward clipping:** Normalize rewards to [-1, 1] before advantage computation to stabilize with SGD.

### 5.5 Minimal Pilot Protocol (Tens of GRPO Iterations)

**Goal:** Confirm gradient signal exists before committing to full training.

**Phase 0 — Sanity checks (before any RL):**
- [ ] `η=0` recovers the BC checkpoint output bit-for-bit ✅ (already done)
- [ ] `η=0.1` produces visually distinct but plausible rollouts
- [ ] Per-step log-probs are finite for η=0.1
- [ ] IS ratio `π_θ/π_θ_old` is 1.0 ± 0.01 before any gradient step

**Phase 1 — Reward distribution check (10 iterations, no training):**
- Sample G=4 rollouts for 10 episodes each skill.
- Log reward histograms per skill. **If reward variance is 0** (all successes or all failures), GRPO advantage is 0 — no signal. Adjust initial policy or environment difficulty.
- Aim for 20–70% success rate at rollout time (the "learning zone").

**Phase 2 — Single-skill pilot with GRASP LoRA only (20–30 iterations):**
- Fix MOVE_HOLDING and PLACE skills to BC policy (no LoRA, no gradient).
- Train only GRASP LoRA with GRPO using GRASP milestone reward.
- Metrics to track: (a) GRASP success rate ↑; (b) IS ratio clipping frequency (should be <20% of steps); (c) advantage std (should be >0.01 — if near 0, group_size problem); (d) LoRA weight norm (should grow slowly — explosion = LR too high).
- **Signal detected if** GRASP success rate improves >5% absolute within 20 iterations.

**Phase 3 — Full 3-skill training (50+ iterations if signal confirmed):**
- Activate all 3 LoRAs + termination heads.
- Use curriculum: first 20 iters GRASP only, next 20 iters GRASP+MOVE, then all three.
- Monitor end-to-end CloseBlenderLid success rate.

**Hyperparameter pivot points:**
- If IS ratio clipping >50%: lower ε to 0.05, or reduce gradient updates per batch.
- If reward doesn't improve after 20 iters: increase η (more exploration) or check reward signal.
- If LoRA weights collapse to near-zero: LR is too low for SGD — try 5e-3.
- If OOM: reduce chunk length L or enable more aggressive gradient checkpointing.

---

## Risks and Conflicts with Current Plan

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **IS ratio explosion (K=5 product)** | High | ε=0.1 (tighter clip); 1 epoch per rollout batch |
| **group_size=4 → high variance advantage** | High | Clip advantages; use running stats; consider G=8 if any VRAM headroom (smaller batch or shorter L) |
| **Per-skill LoRA on DiT — no direct prior** | Medium | Start with shared LoRA + skill token; validate before per-skill |
| **PLACE LoRA data-starved** | Medium | Curriculum: unlock PLACE LoRA only after GRASP+MOVE converge |
| **η choice sensitivity** | Medium | Grid search {0.05, 0.1, 0.3}; pilot 5 iters each |
| **SGD + GRPO + K=5 steps: no exact prior** | Medium | SGD is validated for RLVR in LLMs (arXiv:2602.07729); untested for flow robotics RL — watch for training instability |
| **Simulator predicate: binary and sparse** | Low | Per-skill milestones provide dense signal; RIPT-VLA shows binary reward alone can work |
| **Reward hacking (lid predicate exploited)** | Low | Monitor action diversity; add entropy bonus if diversity collapses |
| **Off-policy staleness with SGD** | Medium | 1 epoch per batch; regenerate rollouts every iteration |

**Biggest unresolved question:** Whether the flow-SDE + GRPO with K=5 steps and G=4 is numerically stable enough for meaningful updates. The evidence from Flow-GRPO and ReinFlow is encouraging but was developed with larger group sizes (G=16) and Adam optimizer. The SGD + small-G regime is scientifically valid (supported by arXiv:2602.07729 for LLMs) but not yet validated for flow policy RL specifically. **The pilot protocol in §5.5 is the critical gating step.**

---

## Quick Reference: Key Papers

| Paper | arXiv ID | Venue | Key contribution |
|-------|----------|-------|-----------------|
| DDPO | 2305.13301 | ICLR 2024 | Denoising as multi-step MDP; exact per-step log-probs |
| DPOK | 2305.16381 | NeurIPS 2023 | KL regularization for diffusion RL |
| DPPO | 2409.00588 | ICLR 2025 | Best practices for diffusion policy RL in robotics |
| Flow-GRPO | 2505.05470 | NeurIPS 2025 | ODE→SDE + GRPO for flow matching; image gen |
| ReinFlow | 2505.22094 | NeurIPS 2025 | Learnable noise for flow policy RL; robotics |
| πRL | 2510.25889 | Preprint | Flow-SDE for flow-based VLAs (pi0, pi0.5) |
| GRAPE | 2411.19309 | Preprint | Preference-based RL for autoregressive VLA |
| ConRFT | 2502.05450 | Preprint | Consistency policy RL for VLA manipulation |
| TGRPO | 2506.08440 | Preprint | Trajectory-wise GRPO for VLA |
| CORAL | 2603.09298 | Preprint | Per-task LoRA experts for multi-task VLA |
| SGD for RLVR | 2602.07729 | Preprint | SGD matches AdamW for RL fine-tuning of LLMs |
| Smart-GRPO | 2510.02654 | Preprint | Noise seed curriculum for flow GRPO |
| SuperFlow | 2512.17951 | Preprint | Per-step credit assignment for flow GRPO |

---

*Brief prepared: September 2026. Findings reflect literature through January 2026. Distinguish well-established results (DDPO, DPPO, ReinFlow, Flow-GRPO — peer-reviewed) from inference (per-skill LoRA on DiT, SGD for flow robot RL — extrapolated from related work).*
