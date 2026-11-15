#!/bin/bash
# Runs the built CLI on a PDF and writes the result to out/. Arguments: PDF path, output name, then CLI flags.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export DPLUS_PYTHON="${DPLUS_PYTHON:-$HOME/.venvs/vllm-rocm10/bin/python}"
unset PYTHONPATH
mkdir -p "$root/out"
input=$1
name=$2
shift 2
before=$(sha256sum "$input")
"$root/dist/dplus" translate -output "$root/out/$name" "$@" "$input"
after=$(sha256sum "$input")
if [ "$before" != "$after" ]; then
  echo "the input file changed during the run" >&2
  exit 1
fi
