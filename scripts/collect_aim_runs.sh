#!/usr/bin/env bash
aim init
DEST="$(pwd)"

for d in ./aim_repos_chinchilla_sweep/*; do
  td=$(mktemp -d)
  if [ -d "$d" ] && compgen -G "$d"/*.tar.gz > /dev/null; then
    tar -C "$td" -xzf "$(ls -1 "$d"/*.tar.gz | head -n1)"
    (cd "$td" && aim runs cp '*' --destination "$DEST")
    rmdir "$td"
  elif [ -d "$d/.aim" ]; then
    cp -a "$d/.aim" "$td/.aim"
    (cd "$td" && aim runs cp '*' --destination "$DEST")
    rm -rf "$td"
  elif [ -f "$d" ] && [[ "$d" == *.tar.gz ]]; then
    tar -C "$td" -xzf "$d"
    (cd "$td" && aim runs cp '*' --destination "$DEST")
    rmdir "$td"
  else
    rmdir "$td"
  fi
done
