#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 framework|full [JOBS] [DEPENDENCY_DIRECTORY]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 3 ]] || usage

mode=$1
jobs=${2:-2}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")
deps_root=$(realpath -m "${3:-$repo_root/.deps}")

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || {
  echo "JOBS must be a positive integer" >&2
  exit 2
}

case "$mode" in
  framework)
    build_dir="$repo_root/build/framework"
    cmake -S "$repo_root" -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release
    ;;
  full)
    env_file="$deps_root/env.sh"
    [[ -f "$env_file" ]] || {
      echo "Missing $env_file; run scripts/bootstrap_dependencies.sh first" >&2
      exit 2
    }
    # shellcheck disable=SC1090
    source "$env_file"
    build_dir="$repo_root/build/full"
    cmake -S "$repo_root" -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DHERA_ENABLE_FULL_RUNTIME=ON \
      -DSEAL_DIR="$SEAL_DIR" \
      -DSCI_DIR="$SCI_DIR" \
      -DEigen3_DIR="$Eigen3_DIR"
    ;;
  *) usage ;;
esac

cmake --build "$build_dir" --parallel "$jobs"
ctest --test-dir "$build_dir" --output-on-failure -j 1

echo "HERA_BUILD_STATUS=PASS"
echo "HERA_BUILD_MODE=$mode"
