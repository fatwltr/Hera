#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [DEPENDENCY_DIRECTORY] [JOBS]" >&2
  exit 2
}

[[ $# -le 2 ]] || usage

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")
deps_root=$(realpath -m "${1:-$repo_root/.deps}")
jobs=${2:-2}

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || {
  echo "JOBS must be a positive integer" >&2
  exit 2
}

"$script_dir/check_prerequisites.sh" full

# shellcheck source=../dependencies.lock
source "$repo_root/dependencies.lock"

src_root="$deps_root/src"
build_root="$deps_root/build"
mkdir -p "$src_root" "$build_root"

"$script_dir/download_dependencies.sh" "$deps_root"

seal_output="$build_root/seal-$SEAL_VERSION"
seal_compression=${HERA_SEAL_COMPRESSION:-zstd}
seal_dir="$seal_output/install/lib/cmake/SEAL-4.4"
if [[ ! -f "$seal_dir/SEALConfig.cmake" ]]; then
  [[ ! -e "$seal_output" ]] || {
    echo "Incomplete SEAL output exists: $seal_output" >&2
    echo "Move it aside and rerun; the script never overwrites dependency builds." >&2
    exit 2
  }
  "$script_dir/prepare_seal.sh" "$src_root/seal" "$seal_output" \
    "$jobs" "$seal_compression"
fi

seal_manifest="$seal_output/install/share/hera-seal/manifest.txt"
grep -qx "SEAL_VERSION=$SEAL_VERSION" "$seal_manifest"
grep -qx "SEAL_COMMIT=$SEAL_COMMIT" "$seal_manifest"
if ! grep -qx "SEAL_COMPRESSION=$seal_compression" "$seal_manifest"; then
  if [[ "$seal_compression" != none ]] ||
     grep -q '^SEAL_COMPRESSION=' "$seal_manifest"; then
    echo "Existing SEAL build has a different compression mode" >&2
    exit 2
  fi
fi

# A new recipe gets a new directory; never mix headers with older SCI archives.
sci_output="$build_root/$HERA_SCI_BUILD"
sci_dir="$sci_output/install/lib/cmake/SCI"
if [[ ! -f "$sci_dir/SCIConfig.cmake" ]]; then
  [[ ! -e "$sci_output" ]] || {
    echo "Incomplete HERA-SCI output exists: $sci_output" >&2
    echo "Move it aside and rerun; the script never overwrites dependency builds." >&2
    exit 2
  }
  "$script_dir/prepare_hera_sci.sh" \
    "$src_root/blb/SCI" "$src_root/bolt/SCI" "$seal_dir" \
    "$sci_output" "$jobs"
fi

sci_manifest="$sci_output/install/share/hera-sci/manifest.txt"
grep -qx "HERA_SCI_ABI=$HERA_SCI_ABI" "$sci_manifest"
grep -qx "SEAL_VERSION=$SEAL_VERSION" "$sci_manifest"
grep -qx "OT_BACKEND=VOLE_SILENT_OT" "$sci_manifest"
grep -qx "OT_EXPANSION=UNROLLED_AES_PREFIX_PAD" "$sci_manifest"
grep -qx "MPC_MEMORY=REUSABLE_SLAB_V1" "$sci_manifest"
grep -qx "MPC_COMPARISON=CORRELATED_AND_V1" "$sci_manifest"
grep -qx "MPC_OPENMP=DIRECTION_TASKS_V2" "$sci_manifest"
grep -qx "RANDOM_SOURCE=OS_RANDOM_DEVICE" "$sci_manifest"
grep -qx "EMP_SOURCE=VENDORED_HEADERS" "$sci_manifest"
grep -qx "BLB_COMMIT=$BLB_COMMIT" "$sci_manifest"
grep -qx "BOLT_COMMIT=$BOLT_COMMIT" "$sci_manifest"

emp_header="$sci_output/install/include/emp-tool/emp-tool.h"
[[ -f "$emp_header" ]] || { echo "Incomplete HERA-SCI package: $emp_header" >&2; exit 2; }

env_file="$deps_root/env.sh"
printf 'export SEAL_DIR=%q\nexport SCI_DIR=%q\nexport Eigen3_DIR=%q\n' \
  "$seal_dir" "$sci_dir" "$sci_output/SCI/extern/eigen/build" > "$env_file"

echo "HERA_DEPENDENCIES_STATUS=PASS"
echo "DEPENDENCY_ENV=$env_file"
echo "Run: source '$env_file'"
