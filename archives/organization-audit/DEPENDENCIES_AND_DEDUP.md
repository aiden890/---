# 경로 의존성 및 중복 코드 판단

## 유지한 실행 경계

- 루트 `run.sh`는 `dirname "$0"`로 이동한 뒤 Docker build context `.` 및 `output/$mode`를 사용한다. Dockerfile이 COPY하는 requirements, 10_nvidia.json, smoke_test.py 위치를 유지했다.
- `xiaomi-cu121/run.sh`는 자기 디렉토리를 `/work`로 마운트한다. Dockerfile이 COPY하는 download.py, probe.py 및 checkpoint/upstream/build context를 이동하지 않았다.
- `rollout.py`는 `Path(__file__).resolve().parent / "upstream"`를 사용한다. 공통 위치로 이동하거나 단순 symlink로 치환하면 import 기준과 Docker bind mount 범위가 달라진다.
- `verify_results.py`, `build_summary.py`, research 스크립트는 `__file__` 기준 결과/소스 경로를 사용한다. 모두 수정하지 않았다.
- 기존 runner들은 원격 `/home/v4/...` 경로와 실험별 컨테이너/볼륨을 명시한다. 실행하지 않았으며 서버/SSH 작업은 하지 않았다.

## 중복 조사 결과와 공통화 보류 이유

[SHA-256별 정확한 중복 목록](code-duplicates.json)에 3개 그룹을 기록했다.

1. `run-t_5af7225b.sh`, 최신 실험 `parent-runner.sh`, research `sources/prior/run-t_5af7225b.sh`: 실행용 기존 경로, worker 소유 복사본, 고정 증거 스냅샷이다. worker 파일이나 해시 고정 스냅샷을 wrapper로 바꾸지 않았다.
2. 두 `dataset_registry.py`: 실험 재현 및 research 출처 증거이다. vendor와 함께 중복 제거 대상에서 제외했다.
3. 세 `rollout.py`: 최초 실행의 `/work/upstream` 상대 import 및 고정 source manifest, 최신 worker 복사본에 걸쳐 있다. 호환 wrapper도 worker 파일 편집과 배포 컨텍스트 변경을 요구하므로 그대로 보존했다.

내용상 유사한 `run-readonly-initial.sh`와 `run-t_5af7225b.sh`도 비교했다. 전자는 `robocasa-assets-t_9f03a613:...:ro`, 후자는 별도 `robocasa-assets-t_5af7225b` writable volume으로 동작이 다르다. 초기 실패/수정 재현 근거이므로 합치지 않았다. 최신 runner는 root/server/client 식별자가 다르다. 영상 검증 스크립트들은 첫/마지막 프레임 검증과 contact sheet 생성으로 출력 계약·마운트 위치가 서로 다르다.

**안전하게 치환 가능한 중복 실행 코드가 없어 이번 정리에서 공통 구현이나 wrapper를 도입하지 않았다.** 중복 제거 완료라고 주장하지 않는다. 코드 변경보다 worker·재현 증거·빌드 경로 보존 조건을 우선했다.

## 보관물 이동 근거

- 프로젝트 코드·문서·JSON의 archive 파일명 참조 조사 및 상위 robot 트리의 `*.{py,sh,md,json,yaml,yml}` 검색에서 이동 대상 참조를 발견하지 못했다.
- tar의 멤버 목록을 읽어 전송/빌드/로그/결과의 과거 묶음임을 확인했다. `context.tar`는 Docker가 직접 사용하는 build context 경로가 아니라 과거 전송 묶음이다. `.bin`도 원본 그대로 보관했다.
- 로컬 `/proc`에서 접근 가능한 cwd/fd를 점검했으며 열린 프로젝트 원본 파일은 발견되지 않았다. 다른 머신이나 미래 작업의 참조까지 보장하는 검사는 아니다.
- 파일명 및 압축 내용은 변경하지 않고 같은 파일시스템 안에서 rename했다. 이동 전후 SHA-256을 확인했다.
- 실행 코드, 활성 결과, 로그, 연구 snapshot 및 기존 README는 편집하지 않았다. 원본 이름의 호환 symlink는 필요성이 확인되지 않은 과거 archive에 남기지 않았다.

## 검증 범위

[before.json](before.json): 기존 top-level 전체 목록, 검사 대상 원본 파일 경로/크기/모드/SHA-256, 이동 대응표, tar 멤버, 제외 경로. 큰 vendor/checkpoint 및 models 하위 디렉토리는 이동·해싱하지 않았다.

[verify_organization.py](verify_organization.py)는 네트워크·Docker·SSH 없이 읽기 전용으로 원본 SHA-256, 이동, 상대 symlink, Markdown 파일 상대 링크, 영상 전체 대응 및 모든 비-vendor shell script의 `bash -n`을 검증한다. `bash -n`은 셸 구문 검사일 뿐 실제 GPU 실행 검증은 아니다. 최초 실행은 DIRECTORY_GUIDE 부재로 실패(RED)했으며 정리 후 재실행한다.

루트는 Git 저장소가 아니므로 git diff 및 Git 기반 독립 리뷰 단계는 적용하지 않았다. 코드 리팩터링은 하지 않았다.
