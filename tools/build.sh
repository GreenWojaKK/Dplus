#!/bin/bash
# Checks formatting, vets, builds and tests the CLI, then builds the device kernel library. Runs under Linux or WSL.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/tools/go/bin:$HOME/.local/bin:$PATH"
export CGO_ENABLED=1
export GOFLAGS=-mod=mod
# Under WSL the Windows module cache serves modules when DNS is unreliable.
windows_cache="/mnt/c/Users/$(id -un)/go/pkg/mod/cache/download"
if [ -d "$windows_cache" ]; then
  export GOPROXY="file://$windows_cache,https://proxy.golang.org,direct"
fi
unformatted=$(gofmt -l cmd internal)
if [ -n "$unformatted" ]; then
  echo "not formatted:" >&2
  echo "$unformatted" >&2
  exit 1
fi
go vet ./...
go build -o dist/dplus ./cmd/dplus
go test ./...
# The device kernel library comes from rdna4-w4a16, the folder beside this repository unless RDNA4_W4A16 names
# another. It is built with the HIP compiler of the model worker's Python environment and copied next to the CLI.
library="${RDNA4_W4A16:-$(pwd)/../rdna4-w4a16}"
RDNA4_W4A16_PYTHON="${DPLUS_PYTHON:-$HOME/.venvs/vllm-rocm10/bin/python}" "$library/tools/build_kernels.sh"
"${DPLUS_PYTHON:-$HOME/.venvs/vllm-rocm10/bin/python}" "$library/tools/generate_binding.py" --check
cp "$library/dist/librdna4_w4a16.so" "$library/rdna4_w4a16.py" "$library/_rdna4_w4a16_abi.py" dist/
