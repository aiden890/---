"""Lab coordinator; dataset archives upload directly from each rollout host."""
import argparse,json,subprocess,time,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--rounds',type=int,default=100);ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    c=json.loads(Path(args.config).read_text());run=c['run'];root=Path(c['root']);root.mkdir(parents=True,exist_ok=True)
    def sync(direction,folder,prefix):
        return subprocess.run([c['transport_python'],str(Path(__file__).with_name('hf_transfer.py')),direction,str(folder),prefix,'--token-file',c['token_file']],capture_output=True,text=True)
    status=root/'status.json';start=json.loads(status.read_text())['next_round'] if status.exists() else 1
    if start==1 and c.get('bootstrap_weights') and not args.dry_run:
        for model,w in c['workers'].items():
            adapter=root/f'adapters/{model}/round-0000'
            while True:
                result=sync('download',adapter,f'{run}/weights/{model}/round-0000')
                if result.returncode==0 and (adapter/'metadata.json').exists():break
                time.sleep(10)
            subprocess.run([x.format(round=0,model=model,run=run,adapter=str(adapter)) for x in w['reload_argv']],check=True)
    for round_index in range(start,args.rounds+1):
        if args.dry_run:
            print(json.dumps(dict(round=round_index,workers=list(c['workers']),steps=c['steps_per_round'],run=run)));break
        # Worker commands wait for all of their episodes, compress, and upload locally.
        workers=[]
        for model,w in c['workers'].items():
            cmd=[x.format(round=round_index,run=run,model=model) for x in w['collect_argv']]
            log=(root/f'{model}-round-{round_index:04d}.log').open('w');workers.append((model,subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
        for model,p,log in workers:
            code=p.wait();log.close()
            if code:raise RuntimeError(f'{model} collection/upload failed; see coordinator logs. No learner job queued.')
        if c.get('tracking_root'):
            subprocess.run([sys.executable,str(Path(__file__).with_name('review_catalog.py')),'--config',args.config,'--round',str(round_index)],check=True)
        prefixes=[dict(model=m,round=round_index,prefix=f'{run}/data/{m}/round-{round_index:04d}') for m in c['workers']]
        for model in c['workers']:
            folder=root/f'jobs/{model}/round-{round_index:04d}';folder.mkdir(parents=True,exist_ok=True)
            (folder/'job.json').write_text(json.dumps(dict(model=model,round=round_index,data_prefixes=prefixes,steps=c['steps_per_round'],skills=c.get('skills',['cup_placement','button_press'])),indent=2))
            r=sync('upload',folder,f'{run}/jobs/{model}/round-{round_index:04d}')
            if r.returncode:raise RuntimeError(r.stderr[-1000:])
        for model,w in c['workers'].items():
            adapter=root/f'adapters/{model}/round-{round_index:04d}'
            while True:
                r=sync('download',adapter,f'{run}/weights/{model}/round-{round_index:04d}')
                if r.returncode==0 and (adapter/'metadata.json').exists():break
                time.sleep(10)
            subprocess.run([x.format(adapter=str(adapter),round=round_index,model=model,run=run) for x in w['reload_argv']],check=True)
        (root/'status.json').write_text(json.dumps(dict(completed_round=round_index,next_round=round_index+1),indent=2))

if __name__=='__main__':main()
