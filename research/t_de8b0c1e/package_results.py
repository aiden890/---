"""Package only the user-rescoped CloseBlenderLid results, without rollout or SSH."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import cv2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'rollouts/closeblenderlid-final-t_de8b0c1e'
UP = ROOT / 'rollouts/xiaomi-robotics-1-t_f22ea8b3'
BASE = UP / 'remote/closeblenderlid'
OLD = ROOT / 'research/t_15f668c5'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not OUT.exists(), 'Refuse to overwrite an existing package'
    source = json.loads((OLD / 'verification.json').read_text())
    assert [e['seed'] for e in source['episodes']] == list(range(7, 13))
    OUT.mkdir(parents=True)
    copies = []
    def copy(src, relative):
        dst = OUT / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest, size = sha(src), src.stat().st_size
        assert not dst.exists()
        shutil.copy2(src, dst)
        assert sha(dst) == sha(src) == digest and dst.stat().st_size == size
        copies.append(dict(source=str(src), destination=str(dst), bytes=size, sha256=digest))
        return dst
    for src in sorted(BASE.rglob('*')):
        if src.is_file():
            copy(src, Path('episodes/additional') / src.relative_to(BASE))
    for name in ('REPORT.md', 'verification.json'):
        copy(OLD / name, 'provenance/upstream-' + name)
    copy(UP / 'REPORT.md', 'provenance/execution-REPORT.md')
    copy(UP / 'remote/versions.json', 'provenance/versions.json')
    copy(UP / 'remote/tools/run-closeblenderlid.sh', 'provenance/run-closeblenderlid.sh')
    task_path = Path('robocasa/environments/kitchen/atomic/kitchen_blender.py')
    task_src = ROOT / 'vendor/robocasa' / task_path
    commit = json.loads((UP / 'remote/versions.json').read_text())['robocasa_commit']
    pinned = subprocess.check_output(['git', '-C', str(ROOT / 'vendor/robocasa'), 'show', f'{commit}:{task_path}'])
    assert pinned == task_src.read_bytes()
    copy(task_src, 'provenance/kitchen_blender.py')
    rows = []
    remote = json.loads((BASE / 'validation_remote.json').read_text())
    for ep in source['episodes']:
        seed = ep['seed']
        if seed == 7:
            video = copy(Path(ep['video']), Path('episodes/baseline') / Path(ep['video']).name)
            stats = copy(Path(ep['stats_path']), 'episodes/baseline/stats.json')
        else:
            video = OUT / 'episodes/additional' / Path(ep['video']).relative_to(BASE)
            stats = OUT / 'episodes/additional' / Path(ep['stats_path']).relative_to(BASE)
        data = json.loads(stats.read_text())
        actual = data['episodes'][0]
        assert data['horizon'] == 900 and actual['seed'] == seed
        assert actual['success'] == ep['success'] and actual['steps'] == ep['steps']
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        fps, n = cap.get(cv2.CAP_PROP_FPS), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (256, 768, 3)
            n += 1
        cap.release()
        assert n == ep['decoded_frames'] == 1 + math.ceil(actual['steps'] / 2)
        assert fps == 20 and sha(video) == ep['sha256']
        remote_path = None
        if seed != 7:
            relative = str(Path(ep['video']).relative_to(BASE))
            record = remote['/cbl/' + relative]
            assert record['sha256'] == sha(video) and record['bytes'] == video.stat().st_size
            assert record['decoded_frames'] == n
            remote_path = '/home/v4/rollouts-xiaomi-t_f22ea8b3/closeblenderlid/' + relative
        rows.append(dict(seed=seed, role='baseline' if seed == 7 else 'additional', success=actual['success'], steps=actual['steps'], horizon=900, termination=actual['termination_reason'], frames=n, fps=fps, duration_seconds=n/fps, video=str(video.relative_to(OUT)), stats=str(stats.relative_to(OUT)), original_local_video=ep['video'], remote_video=remote_path, sha256=sha(video), bytes=video.stat().st_size))
    checks = 0
    for line in (OUT / 'episodes/additional/SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha(OUT / 'episodes/additional' / name) == digest
        checks += 1
    assert checks == 20 and len(rows) == 6
    assert len(list(OUT.rglob('*.mp4'))) == 6
    additional = dict(episodes=len(rows[1:]), successes=sum(r['success'] for r in rows[1:]))
    combined = dict(episodes=len(rows), successes=sum(r['success'] for r in rows))
    assert additional == {'episodes': 5, 'successes': 1}
    versions = json.loads((OUT / 'provenance/versions.json').read_text())
    summary = dict(task_id='t_de8b0c1e', scope='CloseBlenderLid additional seeds8-12 plus existing seed7 only', new_rollouts=0, additional=additional, combined=combined, environment=versions, episodes=rows, instruction=source['episodes'][0]['instruction'], original_horizon=900, success_predicate='lid_on_blender AND gripper_fxtr_far(lid main body, th=0.15); no historical ordering check in task _check_success', official_source=f'https://github.com/robocasa/robocasa/blob/{commit}/{task_path}#L45-L91', commands='provenance/run-closeblenderlid.sh (archival; do not rerun cancelled scope)', command_results='episodes/additional/commands.jsonl', cancelled=['5 new tasks / 2 Composite-Unseen acceptance', 'planning candidate two-seed reproduction', 'candidate clips'], limitations=['No fresh SSH inspection: additional videos compared with preserved remote hashes/sizes/decode report.', 'Baseline verified against upstream local hash and full local decode; no remote baseline comparison in this card.', 'Stats and integrity checks do not establish planning/manipulation/perception causality or internal reasoning.', 'No per-step predicate trace for these episodes; final stats are the outcome evidence.', 'Seeds vary simulator initialization; success on seed9 is not a same-seed reproducibility claim.'], cleanup='No remote resources created or modified in this card. Shared server preserved; live state not rechecked.', superseded_partial='Upstream lunches-s8 was stopped early; preserved outside this package, not a valid episode and not analysed.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = ['seed','role','success','steps','horizon','termination','frames','duration_seconds','video','stats']
    with (OUT / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    lines = ['# CloseBlenderLid 최종 결과', '', '사용자 변경 범위: 추가 5회(seed 8-12)와 기존 seed7만 정리했습니다. 다른 task 탐색/재실행/후보 클립은 사용자 취소이며 완료로 간주하지 않습니다.', '', f"추가 {additional['successes']}/{additional['episodes']} 성공; 기존 포함 {combined['successes']}/{combined['episodes']} 성공. seed9만 885 steps 성공, 나머지는 900-step horizon 종료입니다.", '', '| Seed | 역할 | 성공 | Steps | Frames | 원본 영상 | 최종 상태 |', '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['role']} | {r['success']} | {r['steps']} | {r['frames']} | [MP4]({r['video']}) | [stats]({r['stats']}) |")
    lines += ['', '## 실행 및 판정 근거', '', summary['instruction'], '', '원래 horizon=900, target50/pretrain, 실제 seed=base seed. CUDA12.1, 동일 공식 체크포인트와 공유 서버를 사용한 upstream 실행 기록입니다.', '', f"모델: {versions['model']}; checkpoint revision: {versions['checkpoint_revision']}", f"RoboCasa: {versions['robocasa_commit']}; Xiaomi code: {versions['xiaomi_code']}", '', '[환경/이미지 ID](provenance/versions.json), [실제 runner](provenance/run-closeblenderlid.sh), [seed별 시간/종료코드](episodes/additional/commands.jsonl), [실행 보고서](provenance/execution-REPORT.md).', '', f"[공식 task 코드]({summary['official_source']}), [동일 바이트 스냅샷](provenance/kitchen_blender.py).", '성공은 lid_on_blender와 gripper_fxtr_far(th=0.15)의 AND입니다. task 판정에는 과거 subgoal 순서 검사가 없습니다. 이 패키지는 최종 stats를 결과 근거로 삼으며 프레임별 predicate/내부 reasoning을 추정하지 않습니다.', '', '## 무결성 및 한계', '', '복사 전후 크기/SHA256 대조, manifest 20/20 일치, MP4 6개 전체 로컬 decode 및 기대 프레임 수 검증을 통과했습니다. 추가 5개는 보존된 원격 decode 보고서의 크기/해시/프레임 수와 대조했습니다. 실시간 SSH 재검증은 아닙니다. seed7은 기존 로컬 해시와 대조했습니다.', '', '20 fps 영상의 재생 길이는 CSV에 기록했습니다. 클립은 생성하지 않았으며 원본을 그대로 제공합니다. 종료 step은 stats 근거이고 영상 길이를 시뮬레이터 경과시간 또는 정확한 종료 프레임 timestamp로 혼동하지 않습니다.', '', 'seed9 성공은 항상 실패한다는 주장에 대한 반례입니다. 실패 원인이나 동일-seed 재현성은 검증하지 않았습니다. planning 후보가 없다고 판정한 것이 아니라 분석 요구가 취소되었습니다.', '', '취소된 타 task lunches-s8의 부분 실행은 upstream에 보존되어 있으며 유효 episode로 집계하지 않았습니다. 이 패키지의 CloseBlenderLid 영상에는 프레임 부족/손상 징후가 검출되지 않았습니다.', '', '이 카드에서 서버/컨테이너를 만들거나 변경하지 않았습니다. 공유 서버는 보존했으며 현재 GPU 상태를 재조회하지 않았습니다.', '', '[비교표](comparison.csv), [기계 요약](summary.json), [무결성 검증](integrity.json), [이전 독립 검증](provenance/upstream-verification.json).', '패키지는 기존 산출물을 덮어쓰지 않는 별도 보존 사본입니다.']
    (OUT / 'INDEX.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'REPORT.md').write_text('# 최종 전달 보고서\n\n[INDEX](INDEX.md)를 기준으로 영상, 판정, 실행 근거, 비교표를 확인하세요.\n\n추가 1/5 성공, 기존 포함 1/6 성공. 사용자 취소 범위는 완료 주장에 포함하지 않았습니다. 새 rollout 및 원격 변경 없음.\n')
    for link in re.findall(r'\]\(([^)]+)\)', (OUT / 'INDEX.md').read_text()):
        if '://' not in link and link != 'integrity.json':
            assert (OUT / link).is_file(), link
    result = dict(passed=True, manifest_matches=checks, copied_files=len(copies), videos_decoded=len(rows), copies=copies, decode=rows, remote_comparison='preserved records, not live SSH', original_sources_unchanged=all(sha(Path(c['source'])) == c['sha256'] for c in copies))
    assert result['original_sources_unchanged']
    (OUT / 'integrity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('copies','decode')}, indent=2))
    print(OUT)

if __name__ == '__main__':
    main()
