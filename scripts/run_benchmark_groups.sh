#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

groups=()
benchmark_args=()

if [ "$#" -gt 0 ]; then
  separator_seen=0
  for arg in "$@"; do
    if [ "${arg}" = "--" ] && [ "${separator_seen}" -eq 0 ]; then
      separator_seen=1
      continue
    fi

    if [ "${separator_seen}" -eq 0 ]; then
      groups+=("${arg}")
    else
      benchmark_args+=("${arg}")
    fi
  done
fi

if [ "${#groups[@]}" -eq 0 ]; then
  mapfile -t groups < <(python -c "from imcts.benchmarks.registry import load_bundled_registry; print('\n'.join(load_bundled_registry().list_groups()))")
fi

for group in "${groups[@]}"; do
  printf '\n=== Running benchmark group: %s ===\n' "${group}"
  python -m imcts.benchmarks --group "${group}" --split-by-case "${benchmark_args[@]}"
done
