# Dplus

[영어 원문](../../README.md)

Dplus는 PDF의 읽는 순서와 문단 경계를 복원하고, 원문과 번역을 하나의 연속된 Markdown 문서로 쓰는 로컬 PDF 번역기다. 문단은 단과 쪽 경계를 넘어 이어질 수 있다. 그림·표·캡션·각주는 판정 모델이 정한 문단 사이의 위치에 배치한다.

Heron은 문서 영역과 종류를 식별한다. 공유 d3-ultra-INT4 backbone은 문서 구조를 판정하고 번역을 생성하며, 생성에는 Qwen3.5-27B vocabulary head를 사용한다. 처리 단계와 모델 인터페이스는 [architecture.md](architecture.md)를 참조한다.

## 문서 번역

[Runtime과 모델을 준비한 뒤](#설정) 다음 명령을 실행한다.

```sh
dist/dplus translate -to ko -pages 1-3 paper.pdf
```

문서 전체를 처리하려면 `-pages`를 생략한다. 기본 대상 언어는 한국어다.

출력 경로는 입력 파일 이름을 기준으로 정한다.

| 출력 | 내용 |
|---|---|
| `paper.dplus.md` | 읽는 순서로 정렬한 원문 문단과 번역문. 블록 종류와 inserted element를 표시한다 |
| `paper.dplus.json` | 쪽 분류, 블록, 읽는 순서, 문단 출처, 이미지 경로, 번역 결과와 확인 결과, runtime 통계 |
| `paper.dplus.pictures/` | 200 DPI로 렌더링한 쪽에서 그림 영역을 잘라 저장한 PNG 파일 |

Markdown 문서는 각 그림을 배정된 위치에서 연결한다. 이미지를 base64 데이터로 포함하려면 `-embed-pictures`를 사용한다. 표는 추출한 텍스트를 유지한다. 스캔·손상·빈 쪽으로 분류된 쪽은 건너뛰고 출력에 표시한다. 두 번째 판정 후에도 읽는 순서 검증을 통과하지 못한 쪽은 JSON 보고서에 검토 표시를 남긴다. 번역이 끝나면 CLI는 각 번역을 원문과 대조한 결과(빠진 참조 표시, 상대 길이, 종료 사유)를 문단 종류별과 원문 길이별로 집계해 출력하고, JSON 보고서에도 남긴다.

### 번역 옵션

| 옵션 | 의미 |
|---|---|
| `-to` | 대상 언어. 기본값: `ko` |
| `-from` | 원문 언어. 기본값: 문단별로 감지하는 `auto` |
| `-pages` | 처리할 쪽. 예: `1-3,7`. 기본값: 모든 쪽 |
| `-output` | 확장자를 제외한 출력 경로. 기본값: 입력 경로의 `.pdf`를 `.dplus`로 바꾼 경로 |
| `-lexicon` | 한국어 용어와 지정할 띄어쓰기를 담은 JSON 파일. 예: `{"안전보건": "안전 보건"}` |
| `-batch` | 동시에 생성할 문단 수의 상한. 기본값: `16` |
| `-no-translate` | 띄어쓰기를 복원한 뒤 재구성한 문단을 출력한다 |
| `-page-image` | 읽는 순서 요청에 렌더링한 쪽 이미지를 포함한다 |
| `-embed-pictures` | Markdown 출력에 그림 이미지를 포함한다 |
| `-verbose` | 완료된 번역을 출력한다. `-batch 1`이면 번역 텍스트를 streaming한다 |
| `-models` | 모델 디렉터리. 기본값: `DPLUS_MODELS`, 미설정 시 `~/models` |
| `-python` | Model worker의 interpreter. 기본값: `DPLUS_PYTHON`, 미설정 시 `python3` |
| `-device` | Model worker의 PyTorch device. 기본값: `cuda:0` |

## 설정

### Runtime

CLI는 WSL을 포함한 Linux x64에서 실행한다. 빌드에는 Go 1.27.3 이상과 C compiler가 필요하다. Model worker는 ROCm PyTorch, `transformers` 5.17.0, `compressed-tensors`, `triton`, `fla-core`, `causal-conv1d`, `llguidance`, `safetensors`, `einops`, `pillow`가 설치된 기존 Python 환경을 사용한다.

`DPLUS_PYTHON`을 해당 환경의 Python 실행 파일로 설정한다. 아래 예시 경로를 사용할 환경의 경로로 바꾼다.

```sh
export DPLUS_PYTHON=/path/to/existing/environment/bin/python
```

### 모델

다음 디렉터리를 `~/models` 아래에 배치한다. 다른 모델 루트를 사용하려면 `DPLUS_MODELS` 또는 `-models`로 지정한다.

| 디렉터리 | 출처 또는 준비 방법 | 역할 |
|---|---|---|
| `d3-ultra-INT4` | `vllm-sr/d3-ultra-INT4` | 공유 backbone과 판정 readout |
| `Qwen3.5-27B-lm_head` | 아래 방법으로 `Qwen/Qwen3.5-27B`에서 `lm_head.weight`를 추출한다 | 번역의 vocabulary projection |
| `docling-layout-heron-onnx` | `docling-project/docling-layout-heron-onnx` | `model.onnx`를 사용하는 layout detection |
| `dplus-spacing-ko` | [tools/export_spacing_ko.py](../../tools/export_spacing_ko.py)로 export한다 | `model.onnx`와 `vocab.txt`를 사용하는 한국어 띄어쓰기 |

Vocabulary head는 shape이 `[248320, 5120]`인 BF16 tensor다. 기존 환경에서 다음 Python 코드를 실행하면 `~/models/Qwen3.5-27B-lm_head/lm_head.safetensors`로 저장한다. 모델 루트가 다르면 `destination`을 바꾼다. 추출에는 `huggingface_hub`와 `safetensors`가 필요하다.

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

한국어 띄어쓰기 모델의 exporter에는 `spacer3.npz`와 `spacer-vocab.txt`가 들어 있는 리소스 디렉터리를 제공한다. 두 인자는 리소스 디렉터리와 출력 모델 디렉터리 순서다.

### 빌드

```sh
CGO_ENABLED=1 go build -o dist/dplus ./cmd/dplus
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz | tar -xz -C /tmp
cp /tmp/onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so.1.31.1 dist/libonnxruntime.so
```

`libonnxruntime.so`는 실행 파일과 같은 디렉터리에 둔다.

AMD device kernel 라이브러리는 `rdna4-w4a16` 저장소에서 가져온다. 이 저장소 옆에 두거나 `RDNA4_W4A16`에 그 디렉터리를 지정한 뒤 다음 명령을 실행한다.

```sh
tools/build.sh
```

이 스크립트는 `DPLUS_PYTHON`으로 선택한 Python 환경에 설치된 ROCm SDK의 HIP compiler로 라이브러리를 빌드하고, `librdna4_w4a16.so`를 `dist/`에 복사한다. C++20 host compiler도 필요하다. `RDNA4_W4A16_GPU_ARCH`로 대상 GPU architecture를 선택하며 기본값은 `gfx1201`이다.

## 개발 도구

### 읽는 순서 제어

| 옵션 | 의미 |
|---|---|

### 생성 benchmark

```sh
dist/dplus bench -batch 1,2,4,8,16 paper.dplus.json
```

`bench`는 기존 JSON 보고서의 문단을 지정한 batch size마다 번역한다. Batch size별로 실행 시간, token 처리량, decode step, batch 점유율, padding, 최대 장치 메모리와 문단별 결과를 `paper.dplus.bench.jsonl`에 기록 하나씩 추가한다. Token sequence와 첫 token의 logits는 해당 실행에서 첫 번째로 지정한 batch size의 결과와 비교한다.

### 개발자 모드

```sh
dist/dplus translate -dev paper.pdf
cargo run --manifest-path ../rdna4-w4a16/record/Cargo.toml -- paper.dplus.dev.jsonl
```

`-dev`는 decode, prefill과 판정 실행의 장치 시간을 기록한다. `librdna4_w4a16.so`가 필요하다. `-dev-every N`은 각 phase에서 N개 span마다 한 번 INT4 연산별 시간을 측정한다. 기본값은 `16`이며, `0`이면 연산별 시간 측정을 끈다. Phase 시간과 연산 횟수 기록은 유지한다.

`rdna4-w4a16`의 기록 프로그램은 기록 파일을 검증하고 phase별 합계와 연산별 시간을 출력한다. 기록 형식과 시간 보정은 [architecture.md](architecture.md#개발자-모드)에 설명한다.

### 빌드 검사

Linux 또는 WSL에서 [tools/build.sh](../../tools/build.sh)를 실행하면 Go formatting, vet, 빌드와 테스트를 검사한 뒤 `rdna4-w4a16`에서 device kernel 라이브러리를 빌드한다.

## 라이선스

Dplus는 MIT License를 따른다. [LICENSE](../../LICENSE)를 참조한다.

## 모델 라이선스

d3-ultra와 Heron은 Apache License 2.0을 사용한다. 한국어 띄어쓰기 모델은 QuickSpacer의 MIT license를 사용한다.
