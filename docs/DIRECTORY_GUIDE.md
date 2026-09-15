# RoboCasa 작업 폴더 안내

이 문서의 정식 위치는 `docs/DIRECTORY_GUIDE.md`이며, 루트의 `DIRECTORY_GUIDE.md`는 호환 심볼릭 링크입니다.

## 여기서 시작

- **[영상 모아보기 — Task / 실제 seed별](../videos/README.md)**
- **[실험별 결과·보고서](../experiments/README.md)**
- [이전 tar / 전송 자료 보관함](../archives/README.md)
- [중복 코드 조사·보존 이유·경로 의존성](../archives/organization-audit/DEPENDENCIES_AND_DEDUP.md)
- [구조 개편 이력·롤백·구경로 폐기 계획](DEPRECATIONS.md) · [개편 상세](MIGRATION.md)

## Canonical 구조 (2차 개편)

재사용 코드·설정·결과·문서·보관물을 최상위에서 분리했습니다. 이전 경로는 모두 루트 호환 심볼릭 링크로 계속 resolve 됩니다.

| 경로 | 역할 |
|---|---|
| [scripts/](../scripts/) | 실행·CLI 도구: `run.sh`, `remote.sh`, `smoke_test.py`, `verify_results.py`, `diagnose_egl.py` |
| [configs/](../configs/) | 실험 설정·lock: `requirements-sim.txt`, `requirements.lock`, `10_nvidia.json` |
| [docker/sim/](../docker/sim/) | 시뮬레이터 이미지 Dockerfile (build context는 프로젝트 루트) |
| [results/sim/](../results/sim/) | 최초 시뮬레이터 결과·로그 (`output/`, `logs/`) |
| [docs/](.) | 문서: DIRECTORY_GUIDE, VERIFICATION, XIAOMI_STATUS, DEPRECATIONS, MIGRATION |
| [videos/](../videos/) | `Task/seed-N/실험__episode…mp4` 상대 링크. 실제 영상은 원래 실험 폴더에 보존 |
| [experiments/](../experiments/) | 최신·이전·최초 실험, simulation, research, preflight 탐색 진입점(상대 심볼릭 링크) |
| [archives/](../archives/) | 과거 전송 tar, 로그/결과 묶음, 정리 전 목록·무결성·개편 검증 |

## 제자리 유지된 원자적 모듈 (분해 금지)

아래는 Docker COPY/bind-mount 루트 또는 고정 provenance 경로가 결합되어 있어 **이동하지 않았습니다**. 이동 시 실행 계약·증거 해시가 깨집니다.

| 경로 | 유지 사유 |
|---|---|
| [xiaomi-cu121/](../xiaomi-cu121/) | 최초 추론/rollout 코드 + CUDA 12.1 Docker 컨텍스트 + `-v $root:/work`·`checkpoint`·`upstream` 마운트 + `source.sha256` 고정 매니페스트 |
| [rollouts-xiaomi-t_460aea68/](../rollouts-xiaomi-t_460aea68/) | 최신·활성 실험 원본. 원격 `/home/v4/...` 러너 계약. worker 파일·경로 미수정 |
| [rollouts/xiaomi-robotics-1/](../rollouts/xiaomi-robotics-1/) | 이전 실험 실행 기록·고정 출처 자료 |
| [research/](../research/), [preflight/](../preflight/) | 태스크 연구/출처 스냅샷 및 사전 점검 |
| [vendor/](../vendor/) | upstream 참조 checkout. 이동·해싱·중복 제거하지 않음 |

모델 checkpoint는 `xiaomi-cu121/checkpoint/`에 그대로 두었습니다. 별도 `models/`로 옮기면 `xiaomi-cu121/run.sh`의 `-v "$root/checkpoint:/checkpoint:ro"` 마운트가 깨지고 큰 트리를 복제하므로 신설하지 않았습니다.

## 기존 문서 해석

[README](../README.md)는 초기 simulator 사용법이고, [XIAOMI_STATUS](XIAOMI_STATUS.md)는 당시 blocker 기록입니다. 초기 NOT COMPLETE 문구를 최신 영상 부재로 해석하지 마세요. [실험별 보고서](../experiments/README.md)가 각 시점의 결과를 구분합니다. 이번 작업은 폴더 정리이며 정책 실행/성공 여부를 새로 평가하지 않았습니다.

## 안전성 / 유지보수

- 원본 코드·설정·로그·영상의 바이트는 변경하지 않았습니다. 관리용 실행 코드(`scripts/run.sh`, `scripts/verify_results.py`, `docker/sim/Dockerfile`, `.dockerignore`)만 경로 참조를 갱신했습니다. 상세는 [MIGRATION.md](MIGRATION.md).
- 기존 실행 경로와 Docker bind mount/build context를 유지했습니다. Git 저장소는 아닙니다.
- 정확히 같은 코드 3개 그룹은 worker 복사본/고정 증거/배포 의존성이 있어 보존했습니다. 무리한 wrapper 공통화는 하지 않았습니다.
- 영상 링크는 정리 시점의 스냅샷입니다. 새로운 결과가 생기면 원본을 옮기지 말고 같은 task/seed 구조에 상대 링크를 추가하세요.
- 검증 재실행(로컬 읽기 전용):

```bash
python3 /home/aiden/Desktop/lab/robot/robocasa-docker/archives/organization-audit/verify_organization.py
python3 /home/aiden/Desktop/lab/robot/robocasa-docker/archives/organization-audit/migration/test_layout.py
```

진행 중 worker가 기존 로그/보고서를 갱신하면 원본 SHA-256 검사는 변경을 보고할 수 있습니다. 무조건 되돌리지 말고 변경 소유자를 확인하세요.
