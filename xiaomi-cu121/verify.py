"""Acceptance test; fails unless one real policy episode and MP4 exist."""
import json
import hashlib
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

root=Path('/output')
inference=json.loads((root/'inference.json').read_text())
assert inference['stage']=='REAL_OBSERVATION_INFERENCE_OK'
assert inference['shape']==[16,12]
summary=json.loads((root/'rollout/summary.json').read_text())
assert summary['num_episodes']==1 and summary['num_tasks']==1
stats=summary['tasks']['CloseBlenderLid']
assert stats['horizon']==900
assert len(stats['episodes'])==1
episode=stats['episodes'][0]
assert episode['instruction'] and episode['steps']>0
assert episode['termination_reason'] in ('success','environment_done','environment_truncated','horizon')
if episode['termination_reason']=='horizon':
    assert episode['steps']==900 and not episode['success']
if episode['termination_reason']=='success':
    assert episode['success']
videos=list((root/'rollout/CloseBlenderLid').glob('*.mp4'))
assert len(videos)==1
video=videos[0]
reader=imageio.get_reader(video)
meta=reader.get_meta_data()
count=0
first=last=None
for frame in reader:
    assert frame.shape==(256,768,3)
    if first is None:
        first=frame.copy()
    last=frame.copy()
    count+=1
reader.close()
assert count>=2 and float(np.std(first))>1
imageio.imwrite(root/'rollout-first.png',first)
imageio.imwrite(root/'rollout-last.png',last)
report={'verified':True,'video':str(video.relative_to(root)),'bytes':video.stat().st_size,'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'decoded_frames':count,'fps':meta['fps'],'frame_shape':list(first.shape),'first_last_mean_abs_difference':float(np.abs(first.astype(float)-last.astype(float)).mean()),'episode':episode,'inference':inference}
(root/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
