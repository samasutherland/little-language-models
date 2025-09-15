#!/usr/bin/env bash
set -u

mkdir -p /secrets
if [ -n "${RUNPOD_SECRET_RCLONE_GDRIVE_CONF:-}" ]; then
  echo "$RUNPOD_SECRET_RCLONE_GDRIVE_CONF" | base64 -d > /secrets/rclone.conf
  chmod 600 /secrets/rclone.conf
fi

exec > >(tee "stdout.log")
exec 2> >(tee "stderr.log" >&2)

timeout -k 30s 5m python3 train_model.py
rc=$?
echo "train_model.py exited with $rc"

echo "Uploading results to Google Drive..."

rclone --config /secrets/rclone.conf copy /workspace/checkpoints/ckpt_latest.pt "Gdrive:runpod-uploads/${RUNPOD_POD_ID}" --create-empty-src-dirs --retries 3 
rclone --config /secrets/rclone.conf copy /workspace/stdout.log "Gdrive:runpod-uploads/${RUNPOD_POD_ID}" --create-empty-src-dirs --retries 3 
rclone --config /secrets/rclone.conf copy /workspace/stderr.log "Gdrive:runpod-uploads/${RUNPOD_POD_ID}" --create-empty-src-dirs --retries 3 

echo "Stopping pod $RUNPOD_POD_ID..."
runpodctl remove pod "$RUNPOD_POD_ID"
exit $rc