"""Validate collected real integration evidence, using only Python stdlib."""
import json
from pathlib import Path

_here = Path(__file__).resolve().parent
# Results/configs live at the project root; this script sits in scripts/ (with a
# root compat symlink). Resolve the project root either way.
root = _here.parent if _here.name == "scripts" else _here
results = []
for mode in ('smoke', 'cpu', 'physics'):
    result = json.loads((root / 'output' / mode / 'result.json').read_text())
    assert result['status'] == 'passed', mode
    assert result['environment'] == 'Kitchen' and result['steps'] == 5, mode
    assert result['action_dim'] == 12 and result['simulation_time'] > 0, mode
    assert result['rendered'] == (mode != 'physics'), mode
    if mode != 'physics':
        assert result['image_shape'] == [128, 128, 3] and result['image_std'] > 1
        assert (root / 'output' / mode / 'frame.png').read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    if mode == 'smoke':
        assert 'NVIDIA GeForce RTX 3090' in result['gl_renderer']
    if mode == 'cpu':
        assert result['backend'] == 'osmesa' and 'llvmpipe' in result['gl_renderer']
    results.append({'mode': mode, 'status': result['status'],
                    'renderer': result.get('gl_renderer'), 'steps': result['steps']})
lock = set((root / 'requirements.lock').read_text().splitlines())
freeze = {line for line in (root / 'logs' / 'pip-freeze.txt').read_text().splitlines() if '==' in line}
assert lock == freeze, {'missing': sorted(lock-freeze), 'unexpected': sorted(freeze-lock)}
print(json.dumps({'verified_runs': len(results), 'locked_dependencies': len(lock), 'runs': results}, indent=2))
