"""Read-only directory-organization acceptance test (Python stdlib)."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.parse

ROOT = Path('/home/aiden/Desktop/lab/robot/robocasa-docker')
AUDIT = ROOT / 'archives/organization-audit'
baseline_path = AUDIT / 'before.json'
if not baseline_path.exists():
    baseline_path = Path('/tmp/robocasa-organization-audit/before.json')
before = json.loads(baseline_path.read_text())
assert (ROOT / 'DIRECTORY_GUIDE.md').is_file(), '한글 디렉토리 진입점 누락'
moves = {m['old']: m['new'] for m in before['moves']}
for row in before['entries']:
    path = ROOT / moves.get(row['path'], row['path'])
    assert path.exists(), f'원본 누락: {row["path"]}'
    if 'sha256' in row:
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        assert actual == row['sha256'], f'원본 내용 변경: {row["path"]}'
    else:
        assert os.readlink(path) == row['symlink']
for old, new in moves.items():
    assert not (ROOT / old).exists(), f'분류되지 않은 보관물: {old}'
    assert (ROOT / new).is_file()
links = []
markdown_links = []
shell_scripts = []
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in {'vendor', '.git', 'checkpoint', 'checkpoints', 'models', '.venv'}]
    for name in dirs + files:
        path = Path(base) / name
        if path.is_symlink():
            assert not os.path.isabs(os.readlink(path)), str(path)
            assert path.exists(), f'깨진 심볼릭 링크: {path}'
            links.append(str(path.relative_to(ROOT)))
    for name in files:
        path = Path(base) / name
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^\s)]+)', path.read_text()):
                if '://' not in target and not target.startswith(('#', '/', 'mailto:')):
                    resolved = path.parent / urllib.parse.unquote(target.split('#')[0])
                    assert resolved.exists(), f'깨진 Markdown 링크: {path}: {target}'
                    markdown_links.append({'file':str(path.relative_to(ROOT)), 'target':target})
        if path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
            shell_scripts.append(str(path.relative_to(ROOT)))
original_videos = [r['path'] for r in before['entries'] if r['path'].endswith('.mp4')]
video_links = list((ROOT / 'videos').rglob('*.mp4'))
assert len(video_links) == len(original_videos), '영상 링크 수 불일치'
assert {p.resolve() for p in video_links} == {ROOT / p for p in original_videos}
report = {'preserved_files_sha256':sum('sha256' in r for r in before['entries']),
          'moved_archives':len(moves), 'relative_symlinks':len(links),
          'video_links':len(video_links), 'markdown_relative_links':len(markdown_links),
          'bash_n_passed':shell_scripts, 'excluded':before['excluded'], 'status':'passed'}
print(json.dumps(report, ensure_ascii=False, indent=2))
