#!/usr/bin/env python3
"""Create the curated W&B view for the active Arm A GRPO run.

Only decision-useful per-update metrics are shown. W&B still retains every
logged metric; this view merely hides low-value diagnostics from the main page.
"""
import os

os.environ["WANDB_BASE_URL"] = "http://localhost:8080"

import wandb_workspaces.reports.v2.interface as wr
import wandb_workspaces.workspaces as ws

ENTITY = "aiden-lab-desktop"
PROJECT = "robocasa-grpo"
RUN_NAME = "armA_v4_lr3e7_mg8x8_block20_from_u2_a08e168"
RUN_ID = f"grpo-{RUN_NAME}"
WORKSPACE_NAME = "Arm A GRPO 핵심 지표"


def line(*keys: str, title: str, y_range=(None, None)) -> wr.LinePlot:
    return wr.LinePlot(
        title=title,
        x="iter",
        y=list(keys),
        range_y=y_range,
        title_x="optimizer update",
        smoothing_type="none",
        max_runs_to_show=1,
        legend_position="south",
    )


workspace = ws.Workspace(
    name=WORKSPACE_NAME,
    entity=ENTITY,
    project=PROJECT,
    auto_generate_panels=False,
    settings=ws.WorkspaceSettings(
        x_axis="iter",
        smoothing_type="none",
        sort_panels_alphabetically=False,
        max_runs=1,
        remove_legends_from_panels=False,
    ),
    runset_settings=ws.RunsetSettings(
        query=RUN_NAME,
        pinned_runs=[RUN_ID],
        pinned_columns=["Name", "State", "iter", "mean_return", "post_step_mean_kl"],
    ),
    sections=[
        ws.Section(
            name="1. 업데이트별 리워드와 성공률",
            panels=[
                line("mean_return", "max_return", title="리워드 평균·최댓값 / update"),
                line("reward_std", title="리워드 표준편차 / update", y_range=(0, None)),
                wr.LinePlot(
                    title="GRASP rollout 성공률 / update",
                    x="iter",
                    y=[],
                    custom_expressions=["${n_success_group} / ${trajectories_collected}"],
                    range_y=(0, 1),
                    title_x="optimizer update",
                    title_y="success rate",
                    smoothing_type="none",
                    max_runs_to_show=1,
                    legend_position="south",
                ),
            ],
            is_open=True,
            pinned=True,
            layout_settings=ws.SectionLayoutSettings(columns=3, rows=1),
        ),
        ws.Section(
            name="2. 정책 업데이트가 실제로 일어났나",
            panels=[
                line("adapter_delta_l2", title="LoRA adapter 이동량 L2", y_range=(0, None)),
                line("loss", title="GRPO loss"),
                line("grad_norm", title="gradient norm", y_range=(0, None)),
            ],
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=3, rows=1),
        ),
        ws.Section(
            name="3. PPO 안정성",
            panels=[
                line("post_step_mean_kl", title="업데이트 후 KL", y_range=(0, None)),
                line("post_step_clip_fraction", title="업데이트 후 clip fraction", y_range=(0, 1)),
                line("post_step_ess", "post_step_mean_ratio", title="ESS·importance ratio (1 근처가 안정)", y_range=(0, None)),
            ],
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=3, rows=1),
        ),
        ws.Section(
            name="4. 시간과 학습 데이터량",
            panels=[
                line("collect_seconds", "optimizer_seconds", title="rollout 수집·optimizer 시간 / update", y_range=(0, None)),
                line("trajectories_collected", "trainable_chunks_collected", title="rollout·학습 chunk 수 / update", y_range=(0, None)),
            ],
            is_open=False,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=1),
        ),
    ],
)

saved = workspace.save()
print(getattr(saved, "url", saved))
