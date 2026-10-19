#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [DEPENDENCY_DIRECTORY]" >&2
  exit 2
}

[[ $# -le 1 ]] || usage

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")
deps_root=$(realpath -m "${1:-$repo_root/.deps}")
src_root="$deps_root/src"

command -v git >/dev/null 2>&1 || {
  echo "Missing required tool: git" >&2
  exit 2
}

# shellcheck source=../dependencies.lock
source "$repo_root/dependencies.lock"
mkdir -p "$src_root"

# Older HERA recipes applied this exact patch to the download itself. Only
# undo a byte-for-byte match of that generated change; preserve user edits.
restore_legacy_seal_patch() {
  local destination=$1
  local expected_status=$' M native/src/seal/evaluator.cpp\n M native/src/seal/evaluator.h'
  [[ "$(git -C "$destination" status --porcelain)" == "$expected_status" &&
     "$(git -C "$destination" rev-parse HEAD)" == "$SEAL_COMMIT" ]] || return 0
  local comparison
  comparison=$(mktemp -d)
  mkdir -p "$comparison/native/src/seal"
  for file in evaluator.cpp evaluator.h; do
    git -C "$destination" show "HEAD:native/src/seal/$file" \
      > "$comparison/native/src/seal/$file"
  done
  if patch --silent -d "$comparison" -p1 < "$script_dir/seal_rotation_hoisting.patch" &&
     cmp -s "$comparison/native/src/seal/evaluator.cpp" "$destination/native/src/seal/evaluator.cpp" &&
     cmp -s "$comparison/native/src/seal/evaluator.h" "$destination/native/src/seal/evaluator.h"; then
    echo "Restoring the pinned SEAL checkout from the legacy generated HERA patch"
    git -C "$destination" apply --reverse "$script_dir/seal_rotation_hoisting.patch"
  fi
  rm -rf -- "$comparison"
}

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
    git init --quiet "$destination"
    git -C "$destination" remote add origin "$repository"
  fi

  if [[ "$name" == seal ]]; then
    restore_legacy_seal_patch "$destination"
  fi
  if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    echo "Dependency checkout has local changes: $destination" >&2
    exit 2
  fi
  if [[ "$(git -C "$destination" rev-parse --verify HEAD 2>/dev/null || true)" != "$commit" ]]; then
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

# HERA-SCI copies Eigen from this submodule. BLB's bundled SEAL submodule is
# intentionally omitted because HERA builds the pinned SEAL 4.4.5 checkout.
git -C "$src_root/bolt" submodule update --init --depth 1 SCI/extern/eigen

[[ -f "$src_root/seal/CMakeLists.txt" &&
   -f "$src_root/blb/SCI/src/BuildingBlocks/aux-protocols.cpp" &&
   -f "$src_root/bolt/SCI/src/FloatingPoint/fp-math.cpp" &&
   -f "$src_root/bolt/SCI/extern/eigen/CMakeLists.txt" ]] || {
  echo "Downloaded dependency sources are incomplete" >&2
  exit 1
}

echo "HERA_DEPENDENCY_DOWNLOAD_STATUS=PASS"
echo "DEPENDENCY_SOURCE_DIRECTORY=$src_root"
echo "To build dependencies later: '$script_dir/bootstrap_dependencies.sh' '$deps_root' 2"
