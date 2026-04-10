#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of invocation location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${REPO_ROOT}/logs"
exec > >(tee "${REPO_ROOT}/logs/stdout.local.log")
exec 2> >(tee "${REPO_ROOT}/logs/stderr.local.log" >&2)

echo "Pretraining..."
python3 scripts/1-pretraining.py

echo "Training..."
if command -v gtimeout >/dev/null 2>&1; then
  gtimeout -k 10m 30m python3 scripts/2-training.py
elif command -v timeout >/dev/null 2>&1; then
  timeout -k 10m 30m python3 scripts/2-training.py
else
  echo "No timeout binary found (install coreutils for gtimeout). Running without timeout."
  python3 scripts/2-training.py
fi
rc=$?
echo "2-training.py exited with ${rc}"
sync
sleep 2

echo "Posttraining..."
python3 scripts/3-posttraining.py

exit "${rc}"
