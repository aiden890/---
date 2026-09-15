# RoboCasa 작업 폴더 안내

## 여기서 시작

- **[영상 모아보기 — Task / 실제 seed별](videos/README.md)**
- **[실험별 결과·보고서](experiments/README.md)**
- [이전 tar / 전송 자료 보관함](archives/README.md)
- [중복 코드 조사·보존 이유·경로 의존성](archives/organization-audit/DEPENDENCIES_AND_DEDUP.md)

## 폴더 구조

| 경로 | 역할 / 주의 |
|---|---|
| [videos/](videos/) | `Task/seed-N/실험__episode…mp4` 상대 링크. 실제 영상은 원래 실험 폴더에 보존 |
| [experiments/](experiments/) | 최신·이전·최초 실험, simulation, research, preflight 탐색 진입점 |
| [archives/](archives/) | 과거 전송 tar, 로그/결과 묶음, 정리 전 목록 및 무결성 검증 |
| [rollouts-xiaomi-t_460aea68/](rollouts-xiaomi-t_460aea68/) | 최신 실험 원본. 작업 중인 worker의 파일·경로는 수정하지 않음 |
| [rollouts/xiaomi-robotics-1/](rollouts/xiaomi-robotics-1/) | 이전 실험, 실행 기록과 고정 출처 자료 |
| [xiaomi-cu121/](xiaomi-cu121/) | 최초 추론/rollout 코드, CUDA 12.1 Docker 컨텍스트, checkpoint/upstream, 원본 output |
| [output/](output/), [logs/](logs/) | 최초 시뮬레이터 결과·로그. `run.sh`가 그대로 사용 |
| [research/](research/), [preflight/](preflight/) | 태스크 연구/출처 스냅샷 및 사전 점검 |
| [vendor/](vendor/) | upstream 참조 checkout. 이동·해싱·중복 제거하지 않음 |
| 루트 Dockerfile, run.sh, requirements*, 10_nvidia.json, *.py | 기존 빌드/실행 경로 유지. 재사용 파일을 tar와 함께 옮기지 않음 |

## 기존 문서 해석

[README](README.md)는 초기 simulator 사용법이고, [XIAOMI_STATUS](XIAOMI_STATUS.md)는 당시 blocker 기록입니다. 초기 NOT COMPLETE 문구를 최신 영상 부재로 해석하지 마세요. [실험별 보고서](experiments/README.md)가 각 시점의 결과를 구분합니다. 이번 작업은 폴더 정리이며 정책 실행/성공 여부를 새로 평가하지 않았습니다.

## 안전성 / 유지보수

- 원본 코드·설정·로그·영상은 변경하지 않았습니다. tar류만 실제 `archives/` 하위로 이동했습니다.
- 기존 실행 경로와 Docker bind mount/build context를 유지했습니다. Git 저장소는 아닙니다.
- 정확히 같은 코드 3개 그룹은 worker 복사본/고정 증거/배포 의존성이 있어 보존했습니다. 무리한 wrapper 공통화는 하지 않았습니다.
- 영상 링크는 정리 시점의 스냅샷입니다. 새로운 결과가 생기면 원본을 옮기지 말고 같은 task/seed 구조에 상대 링크를 추가하세요.
- 검증 재실행(로컬 읽기 전용):

```bash
python3 /home/aiden/Desktop/lab/robot/robocasa-docker/archives/organization-audit/verify_organization.py
```

진행 중 worker가 기존 로그/보고서를 갱신하면 원본 SHA-256 검사는 변경을 보고할 수 있습니다. 무조건 되돌리지 말고 변경 소유자를 확인하세요.
