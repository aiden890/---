"""Validate research artifacts without importing RoboCasa or running policy."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
selection = json.loads((root/'selection.json').read_text())
manifest = json.loads((root/'source_manifest.json').read_text())
report = (root/'INDEX.md').read_text()
tasks = selection['selected_tasks']
ids = [t['task_id'] for t in tasks]
primary = [t for t in tasks if t['priority']=='primary']
assert len(ids)==len(set(ids)) and len(ids)>=5
assert len(primary)>=5
assert not set(ids)&set(selection['excluded_tasks'])
assert sum(t['category']=='composite_unseen' for t in primary)>=2
registry=ast.parse((root/'sources/robocasa/utils/dataset_registry.py').read_text())
assignments={t.id:n.value for n in registry.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
categories={k.arg:ast.literal_eval(k.value) for k in assignments['TARGET_TASKS'].keywords}
configs={k.arg:k.value for k in assignments['COMPOSITE_TASK_DATASETS'].keywords}
for t in tasks:
    assert t['task_id'] in categories[t['category']]
    cfg=configs[t['task_id']]
    assert t['horizon']==next(ast.literal_eval(k.value) for k in cfg.keywords if k.arg=='horizon')
    assert t['expected_episode_seed_num_trials_1']==7+categories[t['category']].index(t['task_id'])
    assert any(k.arg=='target' for k in cfg.keywords)
    assert '### '+t['task_id'] in report
    assert t['source_path'] in report
    assert 'get_ep_meta' in t['methods'] and '_check_success' in t['methods']
    section=report.split('### '+t['task_id']+'\n',1)[1].split('\n##',1)[0]
    assert 'Instruction' in section and 'evidence:' in section and '후보' in section
for item in manifest:
    data=(root/item['snapshot']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==item['sha256']
    assert Path(item['local_original']).read_bytes()==data, item['local_original']
    if 'path' in item:
        repo=Path(item['local_original']).parents[len(Path(item['path']).parts)-1]
        git_data=subprocess.check_output(['git','-C',str(repo),'show',selection['robocasa_commit']+':'+item['path']])
        assert git_data==data
result={'passed':True,'unique_selected_tasks':len(ids),'primary_tasks':len(primary),'primary_unseen':sum(t['category']=='composite_unseen' for t in primary),'category_counts':dict(Counter(t['category'] for t in tasks)),'source_snapshots_verified':len(manifest),'checks':['unique IDs and exclusion set','official TARGET_TASKS categories','original registry horizons and target availability','seed computation','per-task instruction/evidence/source sections','all snapshots SHA256 and original byte identity','official source bytes equal pinned git commit'],'policy_executed':False,'remote_files_transferred':False,'limitations':['Static source audit only; remote runtime results are not validated here.']}
(root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
