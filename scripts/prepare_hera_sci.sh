#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 BLB_SCI_SOURCE BOLT_SCI_SOURCE SEAL_DIR OUTPUT_DIRECTORY [JOBS]" >&2
  exit 2
}

[[ $# -ge 4 && $# -le 5 ]] || usage

blb_source=$(realpath "$1")
bolt_source=$(realpath "$2")
seal_dir=$(realpath "$3")
output_parent=$(realpath -m "$(dirname "$4")")
output="$output_parent/$(basename "$4")"
jobs=${5:-2}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")

[[ -f "$blb_source/src/BuildingBlocks/aux-protocols.cpp" ]] || {
  echo "Not a BLB-SCI source tree: $blb_source" >&2
  exit 2
}
[[ -f "$bolt_source/src/FloatingPoint/fp-math.cpp" ]] || {
  echo "Not a BOLT-SCI source tree: $bolt_source" >&2
  exit 2
}
[[ -f "$seal_dir/SEALConfig.cmake" ]] || {
  echo "SEAL_DIR does not contain SEALConfig.cmake: $seal_dir" >&2
  exit 2
}
seal_version_file="$seal_dir/SEALConfigVersion.cmake"
[[ -f "$seal_version_file" ]] || seal_version_file="$seal_dir/SEALConfig.cmake"
grep -Eq '(PACKAGE_VERSION|SEAL_VERSION)[[:space:]]+"?4\.4\.5"?' \
  "$seal_version_file" || {
  echo "HERA-SCI requires Microsoft SEAL exactly 4.4.5" >&2
  exit 2
}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || {
  echo "JOBS must be a positive integer" >&2
  exit 2
}
[[ ! -e "$output" ]] || {
  echo "Refusing to overwrite existing path: $output" >&2
  exit 2
}

mkdir -p "$output/SCI" "$output/SCI/extern/eigen"

# Copy source, headers and EMP dependencies from BLB. Its bundled SEAL and
# Phantom trees are deliberately excluded; the unified stack uses the
# caller's SEAL 4.4.5. Eigen is configured below in the copied tree so this
# recipe also works from a fresh recursive clone (no pre-existing build dir).
rsync -a --exclude build --exclude install --exclude extern \
  "$blb_source/" "$output/SCI/"
eigen_source="$blb_source/extern/eigen"
if [[ ! -f "$eigen_source/CMakeLists.txt" ]]; then
  eigen_source="$bolt_source/extern/eigen"
fi
[[ -f "$eigen_source/CMakeLists.txt" ]] || {
  echo "Eigen submodule is missing; initialize BOLT/SCI/extern/eigen" >&2
  exit 2
}
rsync -a --exclude .git --exclude build \
  "$eigen_source/" "$output/SCI/extern/eigen/"
cmake -S "$output/SCI/extern/eigen" \
  -B "$output/SCI/extern/eigen/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN_BUILD_DOC=OFF \
  -DBUILD_TESTING=OFF

# BOLT's src tree is the numerically validated nonlinear implementation. It is
# migrated onto the BLB copy, then the small BLB prime-wrap extension is added
# back by a reviewable repository patch.
rsync -a "$bolt_source/src/" "$output/SCI/src/"
patch -d "$output/SCI" -p1 < "$repo_root/third_party/hera-sci.patch"

cmake -S "$output/SCI" -B "$output/build" \
  -DBUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DSEAL_DIR="$seal_dir" \
  -DEigen3_DIR="$output/SCI/extern/eigen/build" \
  -DCMAKE_INSTALL_PREFIX="$output/install"
cmake --build "$output/build" --parallel "$jobs"
cmake --install "$output/build"

test -f "$output/install/lib/libSCI-FloatingPoint.a"
test -f "$output/install/lib/libSCI-BuildingBlocks.a"
test -f "$output/install/lib/cmake/SCI/SCIConfig.cmake"
nm -C "$output/install/lib/libSCI-BuildingBlocks.a" | \
  grep 'AuxProtocols::wrap_computation_prime' >/dev/null

blb_commit=unknown
bolt_commit=unknown
eigen_commit=unknown
if git -C "$blb_source" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  blb_commit=$(git -C "$blb_source" rev-parse HEAD)
fi
if git -C "$bolt_source" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  bolt_commit=$(git -C "$bolt_source" rev-parse HEAD)
fi
if git -C "$eigen_source" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  eigen_commit=$(git -C "$eigen_source" rev-parse HEAD)
fi
mkdir -p "$output/install/share/hera-sci"
printf '%s\n' \
  'HERA_SCI_ABI=2' \
  'SEAL_VERSION=4.4.5' \
  "BLB_COMMIT=$blb_commit" \
  "BOLT_COMMIT=$bolt_commit" \
  "EIGEN_COMMIT=$eigen_commit" \
  'BASE=BLB-SCI' \
  'NONLINEAR=BOLT-SCI-src' \
  'EXTENSION=wrap_computation_prime' \
  > "$output/install/share/hera-sci/manifest.txt"

echo "HERA_SCI_PREPARE_STATUS=PASS"
echo "SCI_DIR=$output/install/lib/cmake/SCI"
echo "Eigen3_DIR=$output/SCI/extern/eigen/build"
echo "SEAL_DIR=$seal_dir"
