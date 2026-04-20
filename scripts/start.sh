#!/usr/bin/env bash
set -u # Set no fail because otherwise the script fails and is restarted by runpod; the pod is never closed.

mkdir -p /secrets
if [ -n "${RUNPOD_SECRET_RCLONE_GDRIVE_CONF:-}" ]; then
  echo "$RUNPOD_SECRET_RCLONE_GDRIVE_CONF" | base64 -d > /secrets/rclone.conf
  chmod 600 /secrets/rclone.conf
fi

mkdir -p /workspace/logs
exec > >(tee "/workspace/logs/stdout.log")
exec 2> >(tee "/workspace/logs/stderr.log" >&2)

echo "Pretraining..."
python3 scripts/1-pretraining.py

echo "Training..."
python3 scripts/2-training.py
#if timeout -k 10m 30m python3 scripts/2-training.py; then
#  rc=0
#else
#  rc=$?
#fi
#echo "train_model.py exited with $rc"
#sync
#sleep 2

echo "Posttraining..."
python3 scripts/3-posttraining.py

tar -C /workspace -czf /workspace/aim_repo_${RUNPOD_POD_ID}.tar.gz .aim

echo "Uploading results to Google Drive..."

experiment_name="$(python3 - <<'PY'
import yaml
from pathlib import Path
p = Path("/workspace/configs/context.yaml")
with p.open("r") as f:
    d = yaml.safe_load(f)
print(d["experiment_name"])
PY
)"
  
aim_run_hash="$(cat "/workspace/data/checkpoints/aim_run_hash.txt")"
base_remote="Gdrive:runpod-uploads/${experiment_name}/${aim_run_hash}"

rclone --config /secrets/rclone.conf copy /workspace/logs/ "${base_remote}/logs/" --create-empty-src-dirs --retries 3
rclone --config /secrets/rclone.conf copy /workspace/configs/ "${base_remote}/configs/" --create-empty-src-dirs --retries 3
rclone --config /secrets/rclone.conf copy /workspace/data/checkpoints/ "${base_remote}/checkpoints/" --create-empty-src-dirs --retries 3
rclone --config /secrets/rclone.conf copy /workspace/aim_repo_${RUNPOD_POD_ID}.tar.gz "${base_remote}/.aim/${RUNPOD_POD_ID}/" --retries 3

echo "Stopping pod $RUNPOD_POD_ID..."
runpodctl remove pod "$RUNPOD_POD_ID"
exit $rc