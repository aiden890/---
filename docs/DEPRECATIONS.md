# 구경로 폐기(Deprecation) 계획 및 롤백

2차 구조 개편에서 최상위의 재사용 코드·설정·결과·문서를 목적별 디렉토리로 이동하고, **모든 구경로를 루트 호환 심볼릭 링크로 유지**했습니다. 기존 참조(고정 provenance 매니페스트 `xiaomi-cu121/source.sha256`, 원격 러너, 문서 링크 등)가 전부 계속 resolve 됩니다.

## 호환 심볼릭 링크 (구경로 → canonical)

| 구경로(루트) | canonical |
|---|---|
| `run.sh` | `scripts/run.sh` |
| `remote.sh` | `scripts/remote.sh` |
| `smoke_test.py` | `scripts/smoke_test.py` |
| `verify_results.py` | `scripts/verify_results.py` |
| `diagnose_egl.py` | `scripts/diagnose_egl.py` |
| `requirements-sim.txt` | `configs/requirements-sim.txt` |
| `requirements.lock` | `configs/requirements.lock` |
| `10_nvidia.json` | `configs/10_nvidia.json` |
| `Dockerfile` | `docker/sim/Dockerfile` |
| `VERIFICATION.md` | `docs/VERIFICATION.md` |
| `XIAOMI_STATUS.md` | `docs/XIAOMI_STATUS.md` |
| `DIRECTORY_GUIDE.md` | `docs/DIRECTORY_GUIDE.md` |
| `output` | `results/sim/output` |
| `logs` | `results/sim/logs` |

`experiments/simulation` 심볼릭 링크는 `../output` → `../results/sim/output`로 재지정했습니다(대상 실체 동일).

## 폐기 조건 (구경로 심볼릭 링크 제거 시점)

호환 심볼릭 링크는 아래를 **모두** 만족하기 전에는 제거하지 않습니다.

1. 원격 `/home/v4/...` 러너와 재현 스크립트가 새 경로(또는 새 구조로 재동기화된 컨텍스트)를 사용하도록 갱신됨.
2. `xiaomi-cu121/source.sha256` 등 고정 provenance 매니페스트가 새 경로 기준으로 재발급되거나, 해당 실험이 아카이브로 동결됨.
3. 외부(문서·타 프로젝트·tmux 세션)에서 구 루트 경로를 직접 참조하지 않음이 확인됨.

현재 이 세 조건은 미충족(원격 v4/Docker는 이 카드 범위 밖)이므로 **모든 구경로 심볼릭 링크를 유지**합니다.

## 롤백

이동은 같은 파일시스템 내 rename이며 바이트가 보존됩니다(이동 전후 SHA-256 대조 완료). 되돌리려면:

```bash
python3 archives/organization-audit/migration/rollback.py
```

`rollback.py`는 `movemap.json`을 역으로 적용해 호환 심볼릭 링크를 제거하고 canonical 파일을 원래 경로로 되돌린 뒤, 관리용 편집 파일(run.sh, verify_results.py, docker/sim/Dockerfile, .dockerignore, docs/DIRECTORY_GUIDE.md)의 개편 전 사본은 `archives/organization-audit/migration/pre-migration/`에 보관된 원본으로 복원합니다.
