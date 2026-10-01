#!/usr/bin/env bash
# Canonical Lab entry point. Runtime state stays outside the source repository.
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
state_dir="${PI_REVIEW_ROOT:-/home/aiden/Desktop/lab/robot/pi-cup-data-review}"
tracking_dir="${PI_TRACKING_ROOT:-/home/aiden/Desktop/lab/robot/robocasa-docker/tracking}"
transport_py="${PI_TRANSPORT_PYTHON:-/home/aiden/Desktop/lab/robot/skku-vla-20260930/transport-venv/bin/python}"
token_file="${HF_TOKEN_FILE:-/home/aiden/Desktop/lab/robot/skku-vla-20260930/secrets/hf-token}"
export PYTHONPATH="$code_dir/src${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-status}" in
  review)
    mkdir -p "$state_dir"
    if test -f "$state_dir/server.pid" && kill -0 "$(cat "$state_dir/server.pid")" 2>/dev/null; then printf 'Review server already running.\n'; exit 0; fi
    nohup python3 -m online_bc.review.review_server --root "$state_dir" --tracking "$tracking_dir" --transport-python "$transport_py" --token-file "$token_file" > "$state_dir/server.log" 2>&1 < /dev/null &
    echo "$!" > "$state_dir/server.pid"
    printf 'Review: http://100.86.183.64:8897/\n'
    ;;
  status) python3 -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("http://100.86.183.64:8897/api/state")); print(json.dumps({"controls":d["controls"],"sync":d["sync"],"learner":d["learner"],"eligible":sum(e["eligible"] for e in d["episodes"])},indent=2,ensure_ascii=False))';;
  collect) python3 -m online_bc.orchestration.online_rounds --config "$code_dir/configs/pi05-cup-coordinator.json" --rounds "${2:-5}" ;;
  test-review) python3 -m unittest discover -s "$code_dir/tests" -p test_review.py ;;
  *) printf 'Usage: bash scripts/manage.sh review|status|collect [rounds]|test-review\n' >&2;exit 2;;
esac
