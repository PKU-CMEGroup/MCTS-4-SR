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

benchmark_args=("$@")

for config in imcts/benchmarks/experiments/ablations/model_*.yaml; do
  model="${config##*_}"
  model="${model%.yaml}"
  model="${model^^}"

  printf '\n=== Running ablation Model %s ===\n' "${model}"
  python -m imcts.benchmarks --group Nguyen --config "${config}" "${benchmark_args[@]}"

  printf '\n--- Summary for Model %s ---\n' "${model}"
  python -m imcts.benchmarks.report Nguyen --config "${config}" --level group || \
    printf 'Summary skipped: no reportable CSV output found yet.\n'
done
