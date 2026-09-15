# 2차 구조 개편 상세 (MIGRATION)

/ home/aiden/Desktop/lab/robot/robocasa-docker 의 최상위 정리를 코드·설정·결과·문서·보관물 분리까지 확장했습니다. 로컬 전용, non-git, 정적+부분 런타임 검증. SSH/서버/GPU/Docker 빌드 실행은 이 작업 범위 밖입니다.

## before / after (최상위)

개편 전 최상위에 흩어져 있던 실행 스크립트·설정·Dockerfile·결과·문서를 목적별 디렉토리로 이동했습니다. 구경로는 전부 루트 호환 심볼릭 링크로 유지됩니다.

| 항목 | before (루트) | after (canonical) |
|---|---|---|
| 실행/CLI | `run.sh` `remote.sh` `smoke_test.py` `verify_results.py` `diagnose_egl.py` | `scripts/` |
| 설정/lock | `requirements-sim.txt` `requirements.lock` `10_nvidia.json` | `configs/` |
| 이미지 정의 | `Dockerfile` | `docker/sim/Dockerfile` |
| 시뮬레이터 결과 | `output/` `logs/` | `results/sim/` |
| 문서 | `DIRECTORY_GUIDE.md` `VERIFICATION.md` `XIAOMI_STATUS.md` | `docs/` (+`DEPRECATIONS.md` `MIGRATION.md`) |

이동·중복 통합 목록과 재실행 명령은 [movemap.json](../archives/organization-audit/migration/movemap.json)과 아래 "재실행"을 참조하세요.

## 바이트 보존과 관리 코드 변경 구분

- **바이트 보존 이동(내용 무변경):** `smoke_test.py`, `remote.sh`, `diagnose_egl.py`, `requirements-sim.txt`, `requirements.lock`, `10_nvidia.json`, `VERIFICATION.md`, `XIAOMI_STATUS.md`, `output/`·`logs/` 전체. 이동 전후 SHA-256 대조 통과.
- **관리 코드 경로 참조 변경(diff 기록):**
  - `docker/sim/Dockerfile`: `COPY` 소스를 `configs/…`, `scripts/smoke_test.py`로 갱신(build context는 여전히 프로젝트 루트).
  - `scripts/run.sh`: 루트 해석 로직 추가(루트 심볼릭 링크/직접 실행 모두 지원), `docker build -f docker/sim/Dockerfile .`.
  - `scripts/verify_results.py`: `scripts/`에서 실행 시 프로젝트 루트를 해석하도록 `root` 계산 수정.
  - `.dockerignore`: 이동한 `results/`와 `scripts/remote.sh`를 build context에서 제외(기존 `logs/output/remote.sh` 제외 의미 보존).
  - `docs/DIRECTORY_GUIDE.md`: 새 위치 기준 상대 링크 및 canonical/원자적 모듈 구조 문서화.
  - 각 파일의 개편 전 원본은 `archives/organization-audit/migration/pre-migration/`에 SHA-256 대조된 사본으로 보관.

## 이동하지 않은 원자적 모듈과 사유

`xiaomi-cu121/`, `rollouts*/`, `research/`, `preflight/`, `vendor/`, `videos/`, `experiments/`, `archives/`는 Docker COPY/bind-mount 루트, 원격 러너 계약, 고정 provenance(예: `xiaomi-cu121/source.sha256`가 `robocasa-docker/xiaomi-cu121/...` 경로를 해시), 활성 worker 결과이므로 이동 시 실행/증거 계약이 깨집니다. 모델 checkpoint를 별도 `models/`로 옮기지 않은 이유도 동일(마운트 `-v $root/checkpoint` 파손 + 대형 트리 복제).

## 중복 코드 조사 (통합 보류)

정확 SHA-256 중복 3그룹(runner 3, `dataset_registry.py` 2, `rollout.py` 3)은 worker 소유본·고정 증거 스냅샷·배포 의존(`__file__`/`dirname $0`/Docker COPY)로 안전 치환 근거가 없어, 사용하지 않는 공통 stub을 만들지 않고 경계를 명시한 채 보류했습니다. 상세: [DEPENDENCIES_AND_DEDUP.md](../archives/organization-audit/DEPENDENCIES_AND_DEDUP.md), [code-duplicates.json](../archives/organization-audit/code-duplicates.json). readonly vs writable 마운트 등 실제 의미 차이는 각 runner에 명시적으로 유지됩니다.

## 검증 (재실행)

```bash
cd /home/aiden/Desktop/lab/robot/robocasa-docker
# 레이아웃 불변식(구경로 resolve, 심볼릭 링크·md 링크·영상 집합·bash -n·Docker COPY·보존 해시)
python3 archives/organization-audit/migration/test_layout.py
# 1차 정리 검증(원본 SHA-256, 상대 링크, 영상 대응)
python3 archives/organization-audit/verify_organization.py
# 결과 증거 검증(로컬, GPU 불필요 — 저장된 result.json/frame.png/lock 대조)
python3 verify_results.py
# 셸 구문
for f in scripts/run.sh scripts/remote.sh xiaomi-cu121/run.sh \
  rollouts-xiaomi-t_460aea68/run-t_460aea68.sh rollouts-xiaomi-t_460aea68/parent-runner.sh \
  rollouts/xiaomi-robotics-1/run-t_5af7225b.sh rollouts/xiaomi-robotics-1/run-readonly-initial.sh; do bash -n "$f"; done
```

## 미검증 (범위 밖)

Docker 이미지 빌드, 원격 `/home/v4/...` 러너 실제 실행, GPU rollout은 이 카드에서 실행하지 않았습니다. Dockerfile `COPY` 경로는 정적으로만 무결성 확인했습니다(빌드 미수행). 원격 경로 갱신은 별도 서버 작업 필요.
