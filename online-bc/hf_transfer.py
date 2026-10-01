"""File transport in a separate huggingface_hub>=2 environment; never log tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent/"transport-cache"))
from huggingface_hub import sync_bucket

p = argparse.ArgumentParser()
p.add_argument('direction', choices=['upload', 'download'])
p.add_argument('directory')
p.add_argument('prefix')
p.add_argument('--bucket', default='khmin101/vla-rollout-transfer')
p.add_argument('--token-file', required=True)
a = p.parse_args()
assert a.prefix and all(x not in ('', '.', '..') for x in a.prefix.split('/'))
directory = Path(a.directory)
directory.mkdir(parents=True, exist_ok=True)
remote = f'hf://buckets/{a.bucket}/{a.prefix}'
token = Path(a.token_file).read_text().strip()
assert token
source, destination = (str(directory), remote) if a.direction == 'upload' else (remote, str(directory))
sync_bucket(source, destination, token=token, quiet=True)
files = {}
for f in sorted(directory.rglob('*')):
    if f.is_file():
        h = hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
        files[str(f.relative_to(directory))] = {'bytes': f.stat().st_size, 'sha256': h.hexdigest()}
print(json.dumps({'direction': a.direction, 'prefix': a.prefix, 'files': files}))
