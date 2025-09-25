#!/usr/bin/env bash
if [ ! -d ".aim" ]; then
  aim init
fi
DEST="$(pwd)"

for d in ./aim_repos/*;
do
  td=$(mktemp -d)
  if [ -d "$d" ]; then
    mv "$d" "$td/.aim"
    (cd "$td" && aim runs cp '*' --destination "$DEST")
    mv "$td/.aim" "$d"
    rmdir "$td"
  elif [ -f "$d" ] && [[ "$d" == *.tar.gz ]]; then
    tar -C "$td" -xzf "$d"
    (cd "$td" && aim runs cp '*' --destination "$DEST")
    rmdir "$td"
  else
    rmdir "$td"
  fi
done
