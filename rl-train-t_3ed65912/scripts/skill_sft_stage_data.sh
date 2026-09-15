#!/usr/bin/env bash
# Pre-stage Goal-3 SFT deps + the ONLY needed dataset files into $train so the
# sft client can run under the trainer's --network none isolation (no pip/hf at run time).
# Staged: pylibs/ (pyarrow av huggingface_hub) and hf_cache/ (CBL parquet + meta +
# the 3 distinct video chunk files that ALL 106 CloseBlenderLid episodes share).
# Idempotent; ~640MB. Reproducible: only CBL episodes (data/skill_sft/closeblenderlid_episodes.json).
set -euo pipefail
train=/home/v4/rl-train-t_3ed65912
server_image=xiaomi-cu121:t_9f03a613
mkdir -p $train/pylibs $train/hf_cache
docker run --rm -v $train:/train --entrypoint bash $server_image -lc '
set -e
python3 -m pip install --quiet --no-deps --target /train/pylibs pyarrow av 2>&1 | tail -1  # hf_hub stays image-provided (0.36.2)
PYTHONPATH=/train/pylibs python3 - <<PY
import json
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
REPO="ember-lab-berkeley/robocasa365-pretrain-atomic"; cache="/train/hf_cache"
epf=hf_hub_download(repo_id=REPO, filename="meta/episodes/chunk-000/file-000.parquet", repo_type="dataset", local_dir=cache)
t=pq.read_table(epf).to_pydict()
cbl=set(x["episode_index"] for x in json.load(open("/train/data/skill_sft/closeblenderlid_episodes.json")))
cams=["observation.images.robot0_agentview_left","observation.images.robot0_agentview_right","observation.images.robot0_eye_in_hand"]
idx={e:i for i,e in enumerate(t["episode_index"])}
need=set()
for e in cbl:
    i=idx[e]
    for cam in cams:
        need.add((cam,int(t[f"videos/{cam}/chunk_index"][i]),int(t[f"videos/{cam}/file_index"][i])))
files=["data/chunk-000/file-000.parquet","meta/info.json","meta/tasks.parquet"]
files+=[f"videos/{c}/chunk-{a:03d}/file-{b:03d}.mp4" for c,a,b in sorted(need)]
for f in files:
    hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset", local_dir=cache)
print("staged", len(files), "files;", len(need), "distinct video chunks")
PY'
echo STAGED
