#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

while [ ! -d "${REPO_ROOT}/imcts" ] && [ "${REPO_ROOT}" != "/" ]; do
  REPO_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
done

if [ ! -d "${REPO_ROOT}/imcts" ]; then
  echo "Unable to locate repository root from ${SCRIPT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

groups=()
benchmark_args=()
report_args=()
parsing_benchmark_args=0

for arg in "$@"; do
  if [ "${parsing_benchmark_args}" -eq 0 ] && [ "${arg}" = "--" ]; then
    parsing_benchmark_args=1
    continue
  fi

  if [ "${parsing_benchmark_args}" -eq 0 ]; then
    groups+=("${arg}")
  else
    benchmark_args+=("${arg}")
  fi
done

if [ "${#groups[@]}" -eq 0 ]; then
  mapfile -t groups < <(python -c "from imcts.benchmarks.registry import load_bundled_registry; print('\n'.join(load_bundled_registry().list_groups()))")
fi

index=0
while [ "${index}" -lt "${#benchmark_args[@]}" ]; do
  arg="${benchmark_args[${index}]}"
  case "${arg}" in
    --config|--results-dir)
      report_args+=("${arg}")
      index=$((index + 1))
      if [ "${index}" -lt "${#benchmark_args[@]}" ]; then
        report_args+=("${benchmark_args[${index}]}")
      fi
      ;;
    --config=*|--results-dir=*)
      report_args+=("${arg}")
      ;;
  esac
  index=$((index + 1))
done

for group in "${groups[@]}"; do
  printf '\n=== Running benchmark group: %s ===\n' "${group}"
  python -m imcts.benchmarks --group "${group}" "${benchmark_args[@]}"
done

printf '\n--- Summary for benchmark groups ---\n'
python -m imcts.benchmarks.report "${groups[@]}" "${report_args[@]}" --level group || \
  printf 'Summary skipped: no reportable CSV output found yet.\n'
