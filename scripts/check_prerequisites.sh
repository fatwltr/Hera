#!/usr/bin/env bash
set -euo pipefail

mode=${1:-full}
[[ $# -le 1 && ( "$mode" == full || "$mode" == framework ) ]] || {
  echo "Usage: $0 [framework|full]" >&2
  exit 2
}
for tool in cmake c++ cc git python3; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Missing required tool: $tool; install the packages listed in README.md" >&2
    exit 2
  }
done
python3 - "$(cmake --version)" <<'PY'
import re, sys
version = tuple(map(int, re.search(r'version (\d+)\.(\d+)', sys.argv[1]).groups()))
if version < (3, 21):
    sys.exit('CMake >= 3.21 is required; see README.md')
PY
if [[ "$mode" == full ]]; then
  for tool in patch rsync nm sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "Missing required tool: $tool; see README.md" >&2
      exit 2
    }
  done
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    echo "The cryptographic runtime requires Linux x86-64; framework-only builds are available." >&2
    exit 2
  }
  for feature in aes avx2 sse4_1; do
    if ! grep -Eq "^flags[[:space:]]*:.*[[:space:]]$feature([[:space:]]|$)" /proc/cpuinfo; then
      echo "CPU feature $feature is required by SCI; choose a cloud VM exposing AES-NI, AVX2 and SSE4.1." >&2
      exit 2
    fi
  done
fi
printf 'HERA_PREREQUISITES_STATUS=PASS\n'
