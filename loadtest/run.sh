#!/usr/bin/env bash
# usage: loadtest/run.sh [users] [seconds]   -> results/locust_*.csv
set -euo pipefail
U=${1:-20}; T=${2:-60}
python loadtest/serve.py 8077 & PID=$!; trap "kill $PID" EXIT
for i in $(seq 50); do curl -sf localhost:8077/healthz >/dev/null && break; sleep 0.2; done
locust -f loadtest/locustfile.py --headless -u "$U" -r "$U" -t "${T}s" --host http://127.0.0.1:8077 \
  --csv "results/locust_u${U}" --only-summary
