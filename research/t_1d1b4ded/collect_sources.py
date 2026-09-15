"""Static source collection only: no environment imports or policy execution."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[2]
REPO = LAB / 'robocasa-docker/vendor/robocasa'
OLD = LAB / 'robocasa-docker/rollouts/xiaomi-robotics-1'
COMMIT = '4f8a2980def75a55dff96b990745b83540425f09'
S = '/home/aiden/.hermes/skills/research/grounded-citations/scripts/sources.py'
TASKS = ['BreadSelection', 'CategorizeCondiments', 'PackIdenticalLunches', 'SteamInMicrowave', 'WashLettuce', 'PreSoakPan', 'MakeIceLemonade', 'WashFruitColander']
registry = REPO / 'robocasa/utils/dataset_registry.py'
tree = ast.parse(registry.read_text())
def assignment(name):
    return next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
def kw(node, name):
    return next(k.value for k in node.keywords if k.arg == name)
categories = {k.arg: ast.literal_eval(k.value) for k in assignment('TARGET_TASKS').keywords}
composites = assignment('COMPOSITE_TASK_DATASETS')
old = json.loads((OLD / 'summary.json').read_text())
excluded = sorted({a['task'] for a in old['attempts']} | {'CloseBlenderLid'})
assert len(excluded) == 6 and not set(TASKS) & set(excluded)
assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
subprocess.run(['python3', S, '--ledger', str(ROOT / 'ledger.json'), 'reset'], check=True)
files = [registry, REPO/'robocasa/utils/dataset_registry_utils.py', REPO/'robocasa/utils/object_utils.py', REPO/'robocasa/models/fixtures/sink.py', REPO/'robocasa/models/fixtures/microwave.py']
records = []
for task in TASKS:
    candidates = []
    for p in (REPO/'robocasa/environments/kitchen/composite').rglob('*.py'):
        if 'class '+task+'(' in p.read_text():
            candidates.append(p)
    assert len(candidates) == 1
    p = candidates[0]
    files.append(p)
    cls = next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.ClassDef) and n.name == task)
    methods = {n.name: {'start': n.lineno, 'end': n.end_lineno} for n in cls.body if isinstance(n, ast.FunctionDef)}
    cfg = kw(composites, task)
    category = next(k for k,v in categories.items() if task in v)
    records.append({'task_id': task, 'priority': 'primary' if len(records)<5 else 'reserve', 'category': category, 'split': 'target', 'horizon': ast.literal_eval(kw(cfg,'horizon')), 'category_index': categories[category].index(task), 'base_seed': 7, 'expected_episode_seed_num_trials_1': 7+categories[category].index(task), 'source_path': str(p.relative_to(REPO)), 'registry_lines': [cfg.lineno,cfg.end_lineno], 'methods': methods})
manifest=[]
for p in files:
    rel = p.relative_to(REPO)
    dest=ROOT/'sources'/rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p,dest)
    url=f'https://github.com/robocasa/robocasa/blob/{COMMIT}/{rel}'
    output=subprocess.check_output(['python3', S, '--ledger', str(ROOT/'ledger.json'), 'add', url], text=True).strip()
    manifest.append({'path':str(rel),'local_original':str(p),'snapshot':str(dest.relative_to(ROOT)),'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ledger_result':output})
for p in [OLD/'INDEX.md',OLD/'summary.json',OLD/'run-t_5af7225b.sh',LAB/'robocasa-docker/xiaomi-cu121/rollout.py']:
    dest=ROOT/'sources/prior'/p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p,dest)
    manifest.append({'local_original':str(p),'snapshot':str(dest.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
result={'task_id':'t_1d1b4ded','robocasa_commit':COMMIT,'versions':old['versions'],'previous_artifact_root':str(OLD),'previous_remote_root':'/home/v4/rollouts-xiaomi-t_5af7225b','in_progress_remote_root':'/home/v4/rollouts-xiaomi-t_460aea68','excluded_tasks':excluded,'selected_tasks':records}
(ROOT/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(records,indent=2))
print('Static collection passed; no policy/environment run.')
