#!/usr/bin/env bash
aim init
DEST="$(pwd -P)"
AIM_REPOS="$DEST/aim_repos"
TMP_WORK="${TMPDIR:-/tmp}"
shopt -s nullglob

# Work dirs are always under TMP_WORK; nothing under aim_repos is ever removed.
mktemp_workdir() {
  local td resolved
  td=$(mktemp -d "$TMP_WORK/aim_collect.XXXXXX")
  resolved=$(cd "$td" && pwd -P)
  case "$resolved" in
    "$AIM_REPOS"|"$AIM_REPOS"/*)
      echo "collect_aim_runs: TMPDIR must not be inside aim_repos ($AIM_REPOS)" >&2
      exit 1
      ;;
  esac
  printf '%s\n' "$td"
}

remove_workdir() {
  local td="$1" resolved
  [ -n "$td" ] && [ -d "$td" ] || return 0
  resolved=$(cd "$td" && pwd -P 2>/dev/null) || return 0
  case "$resolved" in
    "$AIM_REPOS"|"$AIM_REPOS"/*)
      echo "collect_aim_runs: refusing to remove path under aim_repos: $resolved" >&2
      exit 1
      ;;
  esac
  rm -rf "$td"
}

for d in ./aim_repos/*/; do
  d="${d%/}"
  td=$(mktemp_workdir)
  if [ -d "$d" ] && compgen -G "$d"/*.tar.gz > /dev/null; then
    tar -C "$td" -xzf "$(ls -1 "$d"/*.tar.gz | head -n1)"
    (cd "$td" && aim runs cp '*' --destination "$DEST")
    remove_workdir "$td"
  elif [ -d "$d/.aim" ]; then
    cp -a "$d/.aim" "$td/.aim"
    (cd "$td" && aim runs cp '*' --destination "$DEST")
    remove_workdir "$td"
  else
    remove_workdir "$td"
  fi
done

for f in ./aim_repos/*.tar.gz; do
  td=$(mktemp_workdir)
  tar -C "$td" -xzf "$f"
  (cd "$td" && aim runs cp '*' --destination "$DEST")
  remove_workdir "$td"
done
