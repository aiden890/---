"""Launch the verified native learner environment on an existing worker host."""
import argparse,json,subprocess
from pathlib import Path
from docker_worker import run_args
p=argparse.ArgumentParser();p.add_argument('--model',default='pi05');p.add_argument('--name',required=True);p.add_argument('command',choices=['train','infer']);p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args()
t=json.loads((Path(__file__).parent/'configs'/f'{a.model}-containers.json').read_text())[0];cmd=t['Config']['Cmd'];checkpoint=cmd[cmd.index('--checkpoint')+1]
script={'train':'train_online_bc.py','infer':'verify_inference.py'}[a.command]
subprocess.run(run_args(t)+['-d','--name',a.name,'-w','/results/coffee-online-bc',t['Config']['Image'],cmd[0],script,'--model',a.model,'--checkpoint',checkpoint,*a.argv],check=True)
