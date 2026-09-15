"""Read-only progress inspection for the task's extension build."""
from pathlib import Path
import json
for root in Path('/tmp').glob('pip-install-*/flash*'):
    logs=list(root.rglob('.ninja_log'))
    objects=list(root.rglob('*.o'))
    print(json.dumps({'root':str(root),'object_count':len(objects),'object_bytes':sum(p.stat().st_size for p in objects),'ninja_logs':[str(p) for p in logs]}))
    for p in root.rglob('build.ninja'):
        print('total build edges',sum(line.startswith('build ') for line in p.read_text().splitlines()))
    for p in logs:
        lines=p.read_text().splitlines()
        print('completed commands',len(lines)-1)
        print('\n'.join(lines[-4:]))
