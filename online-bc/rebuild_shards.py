"""Re-score completed episodes and rebuild masks without rerunning physics."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/results')
from coffee_metrics import score
from bc_capture import Capture

ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args()
for p in Path(args.root).glob('*/result.json'):
    result=json.loads(p.read_text());trace=json.loads((p.parent/'trace.json').read_text());result.update(score(trace));p.write_text(json.dumps(result,indent=2))
    if not (p.parent/'bc').exists():continue
    cap=Capture(p.parent);cap.starts=sorted(int(f.stem.split('-')[1]) for f in cap.root.glob('obs-*.npz'))
    cap.finalize(np.load(p.parent/'actions.npy'),trace,result['model'],result['seed'],result['instruction'])
print(json.dumps({'rebuilt':str(args.root)}))
