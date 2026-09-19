#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")
jobs=${1:-2}

"$script_dir/bootstrap_dependencies.sh" "$repo_root/.deps" "$jobs"
"$script_dir/build_and_test.sh" full "$jobs" "$repo_root/.deps"

echo "HERA_REPRODUCE_STATUS=PASS"
