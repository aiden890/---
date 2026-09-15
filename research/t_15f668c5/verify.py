"""Read-only verification of existing CloseBlenderLid results; no rollouts."""
from pathlib import Path
import hashlib
import json
import math
import cv2

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'rollouts/xiaomi-robotics-1-t_f22ea8b3/remote/closeblenderlid'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
checks = []
for line in (BASE / 'SHA256SUMS').read_text().splitlines():
    expected, relative = line.split(maxsplit=1)
    p = BASE / relative
    assert sha(p) == expected, str(p)
    checks.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': expected})
assert len(checks) == 20
remote = json.loads((BASE / 'validation_remote.json').read_text())
commands = [json.loads(s) for s in (BASE / 'commands.jsonl').read_text().splitlines()]
assert [c['seed'] for c in commands] == list(range(8, 13))
assert all(c['rc'] == 0 for c in commands)
rows = []
for seed in range(7, 13):
    folder = ROOT / 'xiaomi-cu121/output/rollout/CloseBlenderLid' if seed == 7 else BASE / f'seed{seed}/rollout/CloseBlenderLid'
    stats = json.loads((folder / 'stats.json').read_text())
    assert stats['env_name'] == 'CloseBlenderLid' and stats['horizon'] == 900
    assert len(stats['episodes']) == stats['num_episodes'] == 1
    ep = stats['episodes'][0]
    assert ep['seed'] == seed
    assert ep['success'] == (seed == 9)
    assert ep['steps'] == (885 if seed == 9 else 900)
    videos = list(folder.glob('*.mp4'))
    assert len(videos) == 1
    video = videos[0]
    cap = cv2.VideoCapture(str(video))
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
    assert fps == 20 and n == 1 + math.ceil(ep['steps'] / 2)
    digest = sha(video)
    if seed != 7:
        record = remote['/cbl/' + str(video.relative_to(BASE))]
        assert record['decoded_frames'] == n and record['sha256'] == digest
        assert record['bytes'] == video.stat().st_size
    rows.append(dict(ep, horizon=900, stats_path=str(folder / 'stats.json'), video=str(video), decoded_frames=n, fps=fps, sha256=digest))
result = {'task_id': 't_15f668c5', 'scope': 'Read-only verification after user cancellation of planning-candidate reruns; no new episodes', 'new_rollouts': 0, 'manifest_checks': checks, 'manifest_matches': len(checks), 'episodes': rows, 'additional': {'episodes': len(rows[1:]), 'successes': sum(r['success'] for r in rows[1:])}, 'combined': {'episodes': len(rows), 'successes': sum(r['success'] for r in rows)}, 'remote_comparison': 'Compared local bytes to preserved upstream remote manifest and decode report; no fresh SSH query', 'limitations': ['No behavior or causal failure classification from integrity checks.', 'Horizon termination does not establish planning failure.', 'Original multi-task planning-candidate reruns cancelled by user, not completed and not a no-candidate finding.']}
(OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('manifest_checks', 'episodes')}, indent=2))
print('seeds/steps/frames:', [(r['seed'], r['steps'], r['decoded_frames']) for r in rows])
