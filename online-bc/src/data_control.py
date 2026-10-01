"""Durable optional review controls; applied before each optimizer update."""
import json,time
from pathlib import Path

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def episode_id(manifest):
    return f"{manifest['model']}-seed{int(manifest['seed'])}"

def read_controls(path):
    if not path:return {'revision':0,'excluded':{},'paused':False}
    d=json.loads(Path(path).read_text())
    if type(d.get('revision')) is not int or not isinstance(d.get('excluded'),dict) or type(d.get('paused',False)) is not bool:raise ValueError('Invalid data controls')
    return d

def select_candidates(episodes,controls,skill=None):
    result=[]
    for tid,(path,m) in episodes.items():
        if episode_id(m) in controls['excluded']:continue
        indices=[i for i,r in enumerate(m['samples']) if skill is None or r['skill']==skill]
        if indices:result.append((tid,path,m,indices))
    return result

def control_ready(controls,health=None,max_age=45):
    if controls.get('paused'):return False,'user_paused'
    if health:
        try:ok=time.time()-json.loads(Path(health).read_text())['last_success']<=max_age
        except (OSError,ValueError,KeyError):ok=False
        if not ok:return False,'control_sync_stale'
    return True,None
