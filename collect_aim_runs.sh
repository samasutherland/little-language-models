#!/usr/bin/env bash
aim init
DEST="$(pwd)"

for d in ./aim_repos/*;
do
  td=$(mktemp -d);
  mv "$d" "$td/.aim";
  (cd "$td" && aim runs cp '*' --destination "$DEST");
  mv "$td/.aim" "$d";
  rmdir "$td";
done
