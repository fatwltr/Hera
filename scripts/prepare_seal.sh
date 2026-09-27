#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 SEAL_SOURCE OUTPUT_DIRECTORY [JOBS] [none|zstd|zlib]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 4 ]] || usage

seal_source=$(realpath "$1")
output_parent=$(realpath -m "$(dirname "$2")")
output="$output_parent/$(basename "$2")"
jobs=${3:-2}
compression=${4:-none}

[[ -f "$seal_source/CMakeLists.txt" ]] || {
  echo "Not a Microsoft SEAL source tree: $seal_source" >&2
  exit 2
}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || {
  echo "JOBS must be a positive integer" >&2
  exit 2
}
[[ "$compression" == none || "$compression" == zstd ||
   "$compression" == zlib ]] || usage

use_zstd=OFF
use_zlib=OFF
if [[ "$compression" == zstd ]]; then use_zstd=ON; fi
if [[ "$compression" == zlib ]]; then use_zlib=ON; fi
grep -Eq 'project\(SEAL VERSION 4\.4\.5([[:space:]]|\))' \
  "$seal_source/CMakeLists.txt" || {
  echo "Microsoft SEAL source must be exactly version 4.4.5" >&2
  exit 2
}
[[ ! -e "$output" ]] || {
  echo "Refusing to overwrite existing path: $output" >&2
  exit 2
}

mkdir -p "$output/build" "$output/install"
cmake -S "$seal_source" -B "$output/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$output/install" \
  -DSEAL_BUILD_DEPS=ON \
  -DSEAL_BUILD_EXAMPLES=OFF \
  -DSEAL_BUILD_TESTS=OFF \
  -DSEAL_BUILD_BENCH=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DSEAL_USE_ZLIB="$use_zlib" \
  -DSEAL_USE_ZSTD="$use_zstd"
cmake --build "$output/build" --parallel "$jobs"
cmake --install "$output/build"

seal_dir="$output/install/lib/cmake/SEAL-4.4"
[[ -f "$seal_dir/SEALConfig.cmake" ]] || {
  echo "SEALConfig.cmake was not installed at $seal_dir" >&2
  exit 1
}

commit=unknown
if git -C "$seal_source" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  commit=$(git -C "$seal_source" rev-parse HEAD)
fi
mkdir -p "$output/install/share/hera-seal"
printf '%s\n' \
  'HERA_SEAL_ABI=1' \
  'SEAL_VERSION=4.4.5' \
  "SEAL_COMMIT=$commit" \
  'BUILD_TYPE=Release' \
  'BUILD_SHARED=OFF' \
  "SEAL_COMPRESSION=$compression" \
  > "$output/install/share/hera-seal/manifest.txt"

echo "HERA_SEAL_PREPARE_STATUS=PASS"
echo "SEAL_DIR=$seal_dir"
