# Dplus

Dplus is a local PDF translator that reconstructs reading order and paragraph boundaries, then writes the source text and translations as a continuous Markdown document. Paragraphs can continue across columns and pages. Figures, tables, captions and footnotes are placed between paragraphs by the decision model.

Heron identifies document regions and their types. A shared d3-ultra-INT4 backbone determines document structure and generates translations, using the Qwen3.5-27B vocabulary head for generation. See [architecture.md](architecture.md) for the processing stages and model interfaces.

## Translate a document

After [setting up the runtime and models](#setup), run:

```sh
dist/dplus translate -to ko -pages 1-3 paper.pdf
```

Omit `-pages` to process the entire document. The default target language is Korean.

Output paths are based on the input filename:

| Output | Contents |
|---|---|
| `paper.dplus.md` | Source paragraphs and translations in reading order, with block types and inserted elements marked |
| `paper.dplus.json` | Page classifications, blocks, reading order, paragraph sources, image paths, translation results and checks, and runtime statistics |
| `paper.dplus.pictures/` | Figure regions cropped from pages rendered at 200 DPI and saved as PNG files |

The Markdown document links each figure at its assigned position. Use `-embed-pictures` to embed the images as base64 data. Tables retain their extracted text. Pages classified as scanned, corrupt or blank are skipped and identified in the output. Reading orders that fail validation after a second pass are flagged for review in the JSON report. After translation, the CLI prints and the JSON report keeps a check of each translation against its source: the reference marks it lacks, its relative length and its stop reason, tallied by paragraph type and source length.

### Translation options

| Flag | Meaning |
|---|---|
| `-to` | Target language; default: `ko` |
| `-from` | Source language; default: `auto`, detected for each paragraph |
| `-pages` | Pages to process, such as `1-3,7`; default: all pages |
| `-output` | Output path without an extension; default: the input path with `.pdf` replaced by `.dplus` |
| `-lexicon` | JSON file of Korean terms and their required spacing, such as `{"안전보건": "안전 보건"}` |
| `-batch` | Maximum number of concurrently generated paragraphs; default: `16` |
| `-no-translate` | Write the reconstructed paragraphs after spacing restoration |
| `-page-image` | Include the rendered page image in reading-order requests |
| `-embed-pictures` | Embed figure images in the Markdown output |
| `-verbose` | Print completed translations; with `-batch 1`, stream translation text |
| `-models` | Model directory; default: `DPLUS_MODELS`, or `~/models` when unset |
| `-python` | Model worker interpreter; default: `DPLUS_PYTHON`, or `python3` when unset |
| `-device` | PyTorch device for the model worker; default: `cuda:0` |

## Setup

### Runtime

The CLI runs on Linux x64, including WSL. Building it requires Go 1.27.3 or later and a C compiler. The model worker uses an existing Python environment with ROCm PyTorch, `transformers` 5.17.0, `compressed-tensors`, `triton`, `fla-core`, `causal-conv1d`, `llguidance`, `safetensors`, `einops` and `pillow`.

Set `DPLUS_PYTHON` to that environment's Python executable. Replace the example path with your environment's path:

```sh
export DPLUS_PYTHON=/path/to/existing/environment/bin/python
```

### Models

Place the following directories under `~/models`, or specify a different root with `DPLUS_MODELS` or `-models`.

| Directory | Source or preparation | Role |
|---|---|---|
| `d3-ultra-INT4` | `vllm-sr/d3-ultra-INT4` | Shared backbone and decision readout |
| `Qwen3.5-27B-lm_head` | Extract `lm_head.weight` from `Qwen/Qwen3.5-27B` as described below | Vocabulary projection for translation |
| `docling-layout-heron-onnx` | `docling-project/docling-layout-heron-onnx` | Layout detection using `model.onnx` |
| `dplus-spacing-ko` | Export with [tools/export_spacing_ko.py](tools/export_spacing_ko.py) | Korean spacing using `model.onnx` and `vocab.txt` |

The vocabulary head is a BF16 tensor with shape `[248320, 5120]`. Run the following Python code in the existing environment to save it as `~/models/Qwen3.5-27B-lm_head/lm_head.safetensors`. Adjust `destination` if your model root differs. The extraction requires `huggingface_hub` and `safetensors`.

```python
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

revision = "fc05daec18b0a78c049392ed2e771dde82bdf654"
shard = hf_hub_download(
    "Qwen/Qwen3.5-27B", "model.safetensors-00001-of-00011.safetensors",
    revision=revision,
)
with safe_open(shard, "pt") as source:
    head = source.get_tensor("lm_head.weight")
destination = Path.home() / "models" / "Qwen3.5-27B-lm_head"
destination.mkdir(parents=True, exist_ok=True)
save_file(
    {"lm_head.weight": head}, str(destination / "lm_head.safetensors"),
    metadata={"source": "Qwen/Qwen3.5-27B", "revision": revision, "tensor": "lm_head.weight"},
)
```

For Korean spacing, supply a resource directory containing `spacer3.npz` and `spacer-vocab.txt` to the exporter. It takes the resource directory and output model directory as its two arguments.

### Build

```sh
CGO_ENABLED=1 go build -o dist/dplus ./cmd/dplus
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz | tar -xz -C /tmp
cp /tmp/onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so.1.31.1 dist/libonnxruntime.so
```

Keep `libonnxruntime.so` beside the executable.

The AMD device kernel library comes from the `rdna4-w4a16` repository. Place it beside this repository, or set `RDNA4_W4A16` to its directory, then run:

```sh
tools/build.sh
```

The script builds the library with the HIP compiler in the ROCm SDK installed in the Python environment selected by `DPLUS_PYTHON`, and copies `librdna4_w4a16.so` and its two Python binding modules to `dist/`. The worker checks the library ABI version before installing its layer methods. Keep these three files together beside the executable. It also requires a C++20 host compiler. `RDNA4_W4A16_GPU_ARCH` selects the target GPU architecture; the default is `gfx1201`.

## Development tools

### Reading-order controls

| Flag | Meaning |
|---|---|

### Generation benchmark

```sh
dist/dplus bench -batch 1,2,4,8,16 paper.dplus.json
```

`bench` translates the paragraphs in an existing JSON report at each requested batch size. It appends one record per batch size to `paper.dplus.bench.jsonl`, including elapsed time, token throughput, decode steps, batch occupancy, padding, peak device memory and per-paragraph results. Token sequences and first-token logits are compared against the first batch size in the invocation.

### Developer mode

```sh
dist/dplus translate -dev paper.pdf
cargo run --manifest-path ../rdna4-w4a16/record/Cargo.toml -- paper.dplus.dev.jsonl
```

`-dev` records device timings for decode, prefill and decision execution. It requires `librdna4_w4a16.so`. `-dev-every N` samples per-operation INT4 timings once every N spans in each phase; the default is `16`, and `0` disables per-operation timing. Phase timings and operation counts remain enabled.

The record program of `rdna4-w4a16` validates the record file and reports phase totals and operation timings. The format and timing correction are described in [architecture.md](architecture.md#developer-mode).

### Build checks

Run [tools/build.sh](tools/build.sh) under Linux or WSL to check Go formatting, vet, build and tests, then build the device kernel library from `rdna4-w4a16`.

## License

Dplus is licensed under the MIT License; see [LICENSE](LICENSE). [internal/worker/dry_penalty.py](internal/worker/dry_penalty.py) adapts the DRY repetition penalty of vLLM and is licensed under the Apache License 2.0; see [NOTICE](NOTICE).

## Model licenses

d3-ultra and Heron use the Apache License 2.0. The Korean spacing model uses QuickSpacer's MIT license.
