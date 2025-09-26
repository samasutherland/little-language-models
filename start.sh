#!/usr/bin/env bash
set -u

mkdir -p /secrets
if [ -n "${RUNPOD_SECRET_RCLONE_GDRIVE_CONF:-}" ]; then
  echo "$RUNPOD_SECRET_RCLONE_GDRIVE_CONF" | base64 -d > /secrets/rclone.conf
  chmod 600 /secrets/rclone.conf
fi

exec > >(tee "/workspace/experiment/stdout.log")
exec 2> >(tee "/workspace/experiment/stderr.log" >&2)

python3 find_batch_size_and_step_number.py
timeout -k 10m 30m python3 train_model.py
rc=$?
echo "train_model.py exited with $rc"
sync
sleep 2

tar -C /workspace -czf /workspace/aim_repo_${RUNPOD_POD_ID}.tar.gz .aim

echo "Uploading results to Google Drive..."

rclone --config /secrets/rclone.conf copy /workspace/experiment/ "Gdrive:runpod-uploads/${RUNPOD_POD_ID}/" --create-empty-src-dirs --retries 3
rclone --config /secrets/rclone.conf copy /workspace/aim_repo_${RUNPOD_POD_ID}.tar.gz "Gdrive:runpod-uploads/.aim/${RUNPOD_POD_ID}/" --retries 3

echo "Stopping pod $RUNPOD_POD_ID..."
runpodctl remove pod "$RUNPOD_POD_ID"
exit $rc