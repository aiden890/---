"""Verify the final scoped package without changing upstream artifacts."""
from pathlib import Path
import hashlib
import json
import math
import cv2

root = Path('/home/aiden/Desktop/lab/robot/robocasa-docker/rollouts/closeblenderlid-final-t_de8b0c1e')
summary = json.loads((root / 'summary.json').read_text())
rows = []
assert [e['seed'] for e in summary['episodes']] == list(range(7, 13))
for e in summary['episodes']:
    p = root / e['video']
    assert p.stat().st_size == e['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == e['sha256']
    stats = json.loads((root / e['stats']).read_text())
    assert stats['env_name'] == 'CloseBlenderLid' and stats['horizon'] == 900
    assert len(stats['episodes']) == 1
    ep = stats['episodes'][0]
    for k in ('seed', 'success', 'steps'):
        assert ep[k] == e[k]
    cap = cv2.VideoCapture(str(p))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (256, 768, 3)
        n += 1
    cap.release()
    assert n == e['frames'] == 1 + math.ceil(e['steps'] / 2)
    assert fps == e['fps'] == 20
    rows.append({'seed': e['seed'], 'success': e['success'], 'steps': e['steps'], 'frames': n, 'video': str(p)})
assert sum(r['success'] for r in rows) == 1
assert next(r for r in rows if r['success'])['seed'] == 9
result = {'task_id': 't_460aea68', 'scope': summary['scope'], 'verified_episodes': rows, 'additional': {'episodes': len(rows[1:]), 'successes': sum(r['success'] for r in rows[1:])}, 'combined': {'episodes': len(rows), 'successes': sum(r['success'] for r in rows)}, 'checks': ['final package SHA256 and size 6/6', 'stats metadata 6/6', 'full local MP4 decode 6/6'], 'limitations': ['No new episodes or fresh remote verification', 'No causal planning-failure analysis; cancelled by user', 'Shared server preserved; live state not checked']}
Path(__file__).with_name('verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
