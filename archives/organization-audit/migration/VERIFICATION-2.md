# 2차 개편 검증 요약 (VERIFICATION-2)

로컬 전용, non-git. SSH/서버/GPU/Docker 빌드 미실행(범위 밖).

## 결과 (모두 통과)

| 검증 | 명령 | 결과 |
|---|---|---|
| 레이아웃 불변식 (RED→GREEN) | `python3 archives/organization-audit/migration/test_layout.py` | GREEN (구경로 resolve, 32 symlink, md링크, 영상집합, bash -n, Docker COPY, 보존해시) |
| 바이트 무결성 diff | `python3 archives/organization-audit/migration/integrity_diff.py` | preserved_byte_identical=388, vanished=0, undeclared_changes=0 |
| 결과 증거 (로컬, GPU불필요) | `python3 verify_results.py` (루트 심볼릭·`scripts/` 둘 다) | passed (smoke/cpu/physics 3런) |
| 셸 구문 | 8개 runner `bash -n` | 전부 OK |
| Python 구문 | 편집·신규 .py `py_compile` | OK |
| Docker COPY 정적 무결성 | 컨텍스트=루트, `.dockerignore` 반영 | 4개 소스 모두 존재·미제외 |

## 변경 범위

- 바이트 보존 이동: 실행 스크립트/설정/Dockerfile/문서/`output`/`logs` (이동 전후 SHA-256 대조 통과).
- 관리 코드 경로 참조만 편집(5개, diff는 `docs/MIGRATION.md`): `docker/sim/Dockerfile`, `scripts/run.sh`, `scripts/verify_results.py`, `.dockerignore`, `docs/DIRECTORY_GUIDE.md`. 개편 전 원본은 `migration/pre-migration/`에 SHA-256 대조 사본 보관.
- 원자적 모듈(xiaomi-cu121, rollouts*, research, preflight, vendor, videos, experiments, archives) 및 모델 checkpoint 미이동 — Docker COPY/mount·원격 러너·고정 provenance 계약 보존.

## 롤백

```bash
python3 archives/organization-audit/migration/rollback.py   # 심볼릭 링크 제거 + 원위치 복원 + 원본 관리코드 복원
```

## 미검증 (범위 밖)

Docker 이미지 빌드, 원격 `/home/v4/...` 러너 실제 실행, GPU rollout. Dockerfile COPY 경로는 정적으로만 확인(빌드 미수행). 원격 경로 갱신은 별도 서버 작업 필요.
