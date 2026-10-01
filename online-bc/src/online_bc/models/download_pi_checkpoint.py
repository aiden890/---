import argparse
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("directory")
a = p.parse_args()
snapshot_download(
    "robocasa/robocasa365_checkpoints",
    local_dir=a.directory,
    allow_patterns=[
        "pi05_pretrain_human300/multitask_learning/75000/params/**",
        "pi05_pretrain_human300/multitask_learning/75000/assets/**",
    ],
)
