#!/usr/bin/env python3
"""wandb 워크스페이스에 지표를 중요도순으로 패널 배치.

섹션 순서(위→아래):
  1) 최종 목표 — eval delta (before/after/heldout grasp+official)
  2) 학습 신호 — gated, succ_rate, composition
  3) 업데이트 메커니즘 — adapter_dL2, loss, grad_norm
  4) 안정성 — post_kl, post_clip, post_ratio
"""
import os
os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2.interface as wr

ENTITY = "aiden-lab-desktop"
PROJECT = "robocasa-grpo"

def line(*keys, title=None):
    return wr.LinePlot(title=title or ", ".join(keys), x="iter", y=list(keys))

workspace = ws.Workspace(
    name="RL 지표 (중요도순)",
    entity=ENTITY,
    project=PROJECT,
    sections=[
        ws.Section(
            name="1. 최종 목표 — 진짜 개선됐나 (eval delta)",
            panels=[
                line("eval_before_grasp", "eval_after_grasp", title="GRASP 성공률 (before→after)"),
                line("eval_before_official", "eval_after_official", title="Official 성공률 (before→after)"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="2. 학습 신호 — 신호가 있나",
            panels=[
                line("succ_rate", title="그룹 성공률 (추세가 올라야 학습됨)"),
                line("gated", title="GATED (신호없어 스킵 — 낮아야 좋음)"),
                line("n_succ", title="n_succ / group"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="3. 업데이트 메커니즘 — 파라미터가 움직이나",
            panels=[
                line("adapter_dL2", title="어댑터 이동량 (0이면 정지)"),
                line("loss", title="GRPO loss"),
                line("grad_norm", title="grad norm"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="4. 안정성 — 발산 안 하나",
            panels=[
                line("post_kl", title="KL (한 스텝 정책 변화)"),
                line("post_clip", title="clip_fraction (>0.9면 포화)"),
                line("post_ratio", title="importance ratio (1 근처)"),
            ],
            is_open=False,
        ),
    ],
)
saved = workspace.save()
print("워크스페이스 저장됨:", getattr(saved, "url", "ok"))
