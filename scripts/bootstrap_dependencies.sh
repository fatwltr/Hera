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

for tool in cmake git patch rsync; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Missing required tool: $tool" >&2
    exit 2
  }
done

# shellcheck source=../dependencies.lock
source "$repo_root/dependencies.lock"

src_root="$deps_root/src"
build_root="$deps_root/build"
mkdir -p "$src_root" "$build_root"

checkout_pinned() {
  local name=$1
  local repository=$2
  local commit=$3
  local destination="$src_root/$name"

  if [[ ! -d "$destination/.git" ]]; then
    [[ ! -e "$destination" ]] || {
      echo "Refusing non-git dependency path: $destination" >&2
      exit 2
    }
    git clone --filter=blob:none "$repository" "$destination"
  fi

  if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    echo "Dependency checkout has local changes: $destination" >&2
    exit 2
  fi
  # A repeat build with the pinned source already present must not require
  # GitHub access. This is also useful when dependencies were staged on a
  # server before its compute node was disconnected from the network.
  if [[ "$(git -C "$destination" rev-parse HEAD)" != "$commit" ]]; then
    git -C "$destination" fetch --depth 1 origin "$commit"
    git -C "$destination" checkout --detach "$commit"
  fi
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]] || {
    echo "Failed to select pinned revision for $name" >&2
    exit 1
  }
}

checkout_pinned seal "$SEAL_REPOSITORY" "$SEAL_COMMIT"
checkout_pinned blb "$BLB_REPOSITORY" "$BLB_COMMIT"
checkout_pinned bolt "$BOLT_REPOSITORY" "$BOLT_COMMIT"

git -C "$src_root/bolt" submodule update --init --depth 1 SCI/extern/eigen

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

# Keep the previous ABI-compatible build intact for reproducible comparisons.
sci_output="$build_root/hera-sci-abi-$HERA_SCI_ABI-correlated"
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
grep -qx "BLB_COMMIT=$BLB_COMMIT" "$sci_manifest"
grep -qx "BOLT_COMMIT=$BOLT_COMMIT" "$sci_manifest"

# Older installations with the same ABI/commit manifest omitted this
# header-only transitive dependency. Repair that exact packaging omission
# without rebuilding SCI or touching its pinned source checkout.
emp_header="$sci_output/install/include/emp-tool/emp-tool.h"
if [[ ! -f "$emp_header" ]]; then
  emp_source="$sci_output/SCI/deps/emp-tool/emp-tool"
  [[ -f "$emp_source/emp-tool.h" ]] || {
    echo "Incomplete HERA-SCI package: emp-tool headers are absent" >&2
    echo "Move $sci_output aside and rerun dependency bootstrap." >&2
    exit 2
  }
  mkdir -p "$(dirname "$emp_header")"
  rsync -a --exclude '*.rej' "$emp_source/" "$(dirname "$emp_header")/"
fi

env_file="$deps_root/env.sh"
cat > "$env_file" <<EOF
export SEAL_DIR='$seal_dir'
export SCI_DIR='$sci_dir'
export Eigen3_DIR='$sci_output/SCI/extern/eigen/build'
EOF

echo "HERA_DEPENDENCIES_STATUS=PASS"
echo "DEPENDENCY_ENV=$env_file"
echo "Run: source '$env_file'"
