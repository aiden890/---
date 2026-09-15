from huggingface_hub import snapshot_download
sub='pi05_pretrain_human300/multitask_learning/75000'
p=snapshot_download(repo_id='robocasa/robocasa365_checkpoints',
    allow_patterns=[f'{sub}/params/*', f'{sub}/assets/*', f'{sub}/_CHECKPOINT_METADATA'],
    local_dir='/ckpt')
print('DONE', p)
