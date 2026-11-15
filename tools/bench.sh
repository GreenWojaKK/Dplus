#!/bin/bash
# Measures generation on the paragraphs of a report in out/, once at each batch size. Arguments: report name, then CLI
# flags. The runs are appended to out/<name>.bench.jsonl.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export DPLUS_PYTHON="${DPLUS_PYTHON:-$HOME/.venvs/vllm-rocm10/bin/python}"
unset PYTHONPATH
name=$1
shift
"$root/dist/dplus" bench "$@" "$root/out/$name.json"
