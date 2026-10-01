import argparse,subprocess,sys
from pathlib import Path
from learner_service import unpack
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--transport-python',required=True);p.add_argument('--token-file',required=True);a=p.parse_args()
root=Path(a.root)
for model in ['pi05','xiaomi']:
    incoming=root/'incoming'/model
    subprocess.run([a.transport_python,str(Path(__file__).with_name('hf_transfer.py')),'download',str(incoming),'pi05-cup-online-bc-20261001/bootstrap/'+model,'--token-file',a.token_file],check=True)
    unpack(incoming,root/'data'/model)
