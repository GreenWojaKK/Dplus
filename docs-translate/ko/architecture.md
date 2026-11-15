# Dplus 아키텍처

[영어 원문](../../architecture.md)

Dplus는 PDF의 읽는 순서와 문단 경계를 복원하고, 구성한 텍스트를 번역해 하나의 연속된 문서로 쓴다. 그림·표·캡션·각주는 문단 사이에 배치한다. 출력에는 쪽 번호와 원문 블록을 metadata로 연결한다.

## 구성 요소

| 구성 요소 | 구현 | 역할 |
|---|---|---|
| Pipeline | Go, [cmd/dplus](../../cmd/dplus)와 [internal](../../internal) | 쪽 내용을 추출하고 블록을 구성하며, 구조 판정을 요청하고 문단을 조립해 출력 파일을 쓴다 |
| Model worker | Python, [internal/worker/backbone.py](../../internal/worker/backbone.py) | 공유 backbone을 로드하고 판정 요청에 응답하며 번역을 생성한다 |
| Device library | HIP와 C++, `rdna4-w4a16` 저장소 | Packed INT4 행렬곱을 실행하고 장치 시간을 기록한다 |

Pipeline은 `go-pdfium`을 통해 pdfium으로 문자를 추출하고 쪽을 렌더링한다. Heron과 한국어 띄어쓰기 모델은 ONNX Runtime의 Go binding으로 실행한다. CLI는 실행 파일 디렉터리에서 ONNX Runtime 공유 라이브러리를 로드한다.

Worker는 GPU에 상주하는 d3-ultra-INT4 backbone 한 벌을 소유한다. 판정은 `prefix_bidir` attention에서 답 코드 readout을 사용한다. 번역은 같은 backbone을 causal attention으로 실행하고 Qwen3.5-27B vocabulary head를 사용한다. 각 실행 경로는 자신의 계산 상태를 유지한다.

## 문서 표현

| 객체 | 내용 |
|---|---|
| 쪽 | 쪽 번호, 크기, 텍스트 방향, PDF 객체 수와 추출한 문자 |
| 블록 | 텍스트, bounding box, 쪽 번호, Heron의 `kind`와 신뢰도 `score`, 텍스트 출처, 할당된 `role`과 선택적 그림 경로 |
| 문단 | 연결한 원문, 블록 종류와 그 신뢰도 점수, 역할, 원문 쪽과 블록 ID, 띄어쓰기를 복원한 텍스트, 번역, 번역 확인 결과와 선택적 그림 경로 |

블록의 `kind`는 Heron의 분류다. `role`은 해당 블록이 출력에 참여하는 방식을 정한다.

| 역할 | 블록 종류 | 처리 |
|---|---|---|
| 본문 | `title`, `section_header`, `text`, `list_item`, `formula`, `code` | 읽는 순서를 정하고 이어지는 블록을 문단으로 연결한다 |
| Inserted element | `picture`, `table`, `caption`, `footnote`와 그 밖의 종류 | 본문 문단 사이의 위치를 정한다 |
| 제거 대상 | `page_header`, `page_footer` | Serialized document에서 제외한다 |

판정 모델은 `text` 블록을 연속된 본문 또는 inserted element로 추가 분류한다. 이후 구조 판정에는 Heron 분류, 신뢰도 점수와 본문 판정 결과를 전달한다.

## 처리 단계

### 쪽 분류와 블록 추출

Gate는 PDF 객체 수와 추출한 문자를 사용해 각 쪽을 `healthy`, `scanned`, `corrupt`, `blank`로 분류한다. `healthy`인 쪽은 블록 추출로 진행한다. 건너뛴 쪽의 분류도 출력에 기록한다.

Heron은 렌더링한 쪽 이미지에서 영역을 검출하고 bounding box, 블록 종류와 신뢰도 점수를 반환한다. Pipeline은 이미지 좌표를 쪽 좌표로 변환하고 겹치는 검출 결과를 정리한다.

1. 같은 영역의 검출 결과가 경쟁하면 신뢰도가 높은 결과를 유지한다.
2. 표나 그림 내부에 포함된 검출 결과보다 이를 포함하는 표·그림 영역을 유지한다.
3. 그 밖에 신뢰도가 더 높은 영역에 대부분 가려진 검출 결과를 제거한다.

PDF 글자 층의 각 문자는 자신의 중심을 포함하는 유지된 영역 중 신뢰도가 가장 높은 영역에 할당한다. 어느 영역에도 속하지 않는 문자는 쪽 보고서에서 센다. 블록 구성 단계는 문자를 줄로 묶고 표의 행 경계를 유지한다. 겹침 판정식과 임계값은 [internal/layout/blocks.go](../../internal/layout/blocks.go)에 정의한다.

그림은 쪽을 200 DPI로 렌더링한 뒤 검출 영역을 잘라 PNG 파일로 저장한다. 블록과 해당 출력 문단은 이 파일을 참조한다.

### 본문 구분과 읽는 순서

Pipeline은 쪽마다 판정 요청 하나를 제출해 `text` 블록이 연속된 본문인지 분류한다. 이후 다음에 읽을 블록을 반복해서 물어 본문 블록의 순서를 정한다.

읽는 순서 검증은 텍스트를 왼쪽에서 오른쪽으로, 블록을 위에서 아래로 읽는 문서에 두 가지 블록 쌍 규칙을 적용한다.

1. 수평으로 겹치고 수직으로 분리된 두 블록에서는 위쪽 블록이 아래쪽 블록보다 먼저 온다.
2. 한 블록이 다른 블록의 완전히 왼쪽에 있으면 먼저 온다. 단, 두 블록의 수직 중심 사이에 있으면서 양쪽과 수평으로 겹치는 블록이 있으면 이 규칙을 적용하지 않는다.

검증 전에 bounding box를 쪽의 텍스트 방향에 맞춰 회전한다. 그림과 표는 두 번째 규칙의 구분 요소로 참여한다. 좌표 관계의 판정식과 허용 오차는 [internal/order/order.go](../../internal/order/order.go)에 정의한다.

각 선택 단계의 후보는 아직 읽지 않은 본문 블록 중 규칙 1에 따른 선행 블록을 모두 읽은 블록이다. 후보가 하나면 바로 선택한다. 여러 개면 판정 모델이 선택지의 확률을 계산하고, pipeline이 허용된 후보 중 확률이 가장 높은 블록을 선택한다.

완성된 순서는 두 규칙으로 검사한다. 각 위반은 충돌한 블록 쌍을 식별한다. 위반이 있는 쪽은 해당 위반을 입력에 포함해 두 번째 판정을 수행한다. 위반이 남으면 쪽 보고서에 순서를 유지하고 검토 표시를 설정한다.

### 문단 구성

Pipeline은 읽는 순서에서 인접한 본문 블록의 각 경계가 같은 문장이나 문단으로 이어지는지 묻는다. 질문은 쪽별로 묶는다. 쪽을 넘는 경계는 양쪽의 읽는 순서가 준비된 뒤 판정한다.

제목과 절 제목은 문단 경계를 형성한다. 그 밖의 경계 질문에는 인접 블록의 종류와 텍스트를 넣는다. 이어지는 블록은 단·쪽 경계와 그 사이의 inserted element를 넘어 연결한다. 각 결과 문단은 원문 블록 ID와 쪽 번호를 유지한다.

### Inserted element 배치

배치는 세 단계로 수행한다.

1. **캡션 연결.** 모델은 각 캡션의 대상을 같은 쪽의 표·그림 중에서 선택하거나 대상 없음을 선택한다. 연결된 캡션은 대상과 한 묶음을 이루고, 묶음 안에서 대상보다 앞에 온다.
2. **위치 선택.** 모델은 각 요소 또는 묶음에 대해 해당 쪽에 걸친 본문 문단 뒤의 위치를 선택한다. 첫 쪽에는 문서 첫 본문 문단 앞도 허용한다. 본문 문단이 없는 쪽에서는 요소를 앞선 본문 뒤에 배치한다.
3. **같은 위치의 순서.** 여러 요소가 같은 위치에 놓이면 다음 요소를 차례로 묻는 질문으로 순서를 정한다.

위치 선택 요청에는 참조 표시와 각주 번호를 포함한 요소와 후보 문단의 전문을 넣는다. 캡션 연결과 위치 선택은 각각 쪽별 질문을 한 요청으로 묶는다.

### 띄어쓰기와 번역

Pipeline은 `-from` 옵션이나 텍스트의 문자 체계로 문단별 원문 언어를 정한 뒤 띄어쓰기를 복원한다.

| 언어 | 처리 |
|---|---|
| 한국어 | ONNX 띄어쓰기 모델로 단어 경계를 예측한 뒤 띄어쓰기 사전을 적용한다 |
| 중국어와 일본어 | 동아시아 문자 사이의 공백을 제거한다 |
| 그 밖의 언어 | 추출한 공백을 유지한다 |

번역 대상 문단 종류는 제목, 절 제목, 본문, 목록 항목, 캡션, 각주, 표, 문서 색인, key-value 영역과 form이다. 이미 대상 언어로 작성된 문단은 원문을 유지한다.

번역 요청은 원문 언어와 대상 언어를 지정하는 system message, 띄어쓰기를 복원한 원문과 출력 token 예산을 포함한다. System message는 번역문만 출력하고, 숫자·이름·참조 표시·수식을 보존하며, 보고서와 논문의 평이한 문어체로 쓰도록 요청한다. 각 문단을 독립적으로 번역하므로 모든 문단에 같은 문체를 지시한다. 또한 문단의 블록 종류와 신뢰도 점수를 알리고, 그 종류의 번역이 유지할 형식을 지시한다.

| 문단 종류 | 번역 형식 |
|---|---|
| `title`, `section_header` | 문장이 아닌 간결한 제목. 절 번호는 그대로 둔다 |
| 본문의 `text` | 모든 문장을 순서대로 이어지는 산문으로 |
| Inserted element로 분류된 `text` | 모든 문장을 순서대로, 본문과 분리된 글로 |
| `list_item` | 앞의 기호, 번호 또는 문자를 유지한다 |
| `caption` | Figure, Table 같은 표시어는 대상 언어의 일반적인 표기로, 번호는 그대로 둔다 |
| `footnote` | 앞의 표시를 유지한다 |
| `table` | 셀을 읽는 순서대로 두고 줄바꿈을 유지한다. 모든 셀을 번역하고 숫자·단위·코드는 그대로 둔다 |
| `document_index`, `key_value_region`, `form` | 항목과 label을 번역하고 숫자·날짜·코드·쪽 번호는 그대로 둔다 |

문단은 첫 블록의 종류와 점수를 가진다. 수식이나 코드 블록으로 시작한 문단은 그 뒤를 잇는 첫 산문 블록의 종류와 점수를 가진다. 예산은 원문 문자당 4 token이며, 64–4,096 token 범위로 제한한다. 각 문단은 독립적으로 번역한다.

각 요청은 출력 전체가 맞아야 하는 정규식인 pattern도 함께 전달한다. Pattern은 대상 언어의 문자 체계(한국어는 한글, 중국어는 한자, 일본어는 한자와 가나), Latin·Greek 글자, 그리고 숫자·문장 부호·기호처럼 모든 문자 체계가 공유하는 문자를 허용한다. Generation은 pattern을 벗어나는 token을 선택하지 않으므로, 번역에는 다른 문자 체계의 글이 들어가지 않는다.

### 번역 확인

생성이 끝나면 각 번역을 띄어쓰기를 복원한 원문과 대조한다.

| 항목 | 정의 |
|---|---|
| 참조 표시 | 원문에 나오는 서로 다른 숫자열(전각 숫자는 ASCII로 읽고 천 단위 구분 기호는 뺀다)과 각주 기호 `*`, `†`, `‡`, `§`, `¶`. 번역에 없는 표시는 누락으로 기록한다 |
| 상대 길이 | 번역의 원문 문자당 문자 수를 같은 원문 언어에서 번역한 실행 내 문단들의 중앙값으로 나눈 값. 40자 이상인 원문만 판단하며, 0.5 미만이면 짧은 번역으로 표시한다 |
| 종료 사유 | worker가 보고한 종료 token 또는 출력 예산 소진 |

문단은 자신의 확인 결과를 기록한다. 실행 요약은 문단 수, 종료 사유, 참조 표시 수, 누락된 표시 수, 짧은 번역 수를 문단 종류별(inserted element는 본문과 구분)과 원문 길이별(100자 미만, 100–399자, 400–1,599자, 1,600자 이상)로 집계한다.

### 출력

Markdown 출력은 원문 문단과 번역문을 읽는 순서로 담고, 블록 종류와 inserted element를 표시한다. 그림 참조는 `-embed-pictures`에 따라 상대 이미지 경로나 내장한 base64 데이터를 사용한다.

JSON 보고서는 쪽 분류, 블록 좌표와 분류, 읽는 순서 판정과 검토 표시, 문단 출처, 그림 경로, 번역 결과와 확인 결과, runtime 통계를 유지한다.

## Worker 프로토콜

Go client는 처음 사용할 때 Python worker를 시작한다. 요청과 응답은 stdin과 stdout을 통해 한 줄에 JSON 객체 하나씩 전달한다. Worker는 먼저 준비 상태와 장치 정보를 보고한다. 각 요청에는 ID를 넣고 응답에는 해당 ID를 넣는다. Stdin을 닫으면 worker가 종료된다.

| 연산 | 입력 | 응답 |
|---|---|---|
| `decide` | `state`의 문서 정보, 종류가 지정된 질문과 선택적 쪽 이미지. 읽는 순서 요청에는 `moving`과 `whole`도 지정할 수 있다 | 선택한 항목 또는 확률을 포함한 답 |
| `generate` | `key`로 식별하는 문단 요청, batch 설정, 선택적 텍스트 streaming·비교 설정 | 완료된 문단 결과, 선택적 text delta, pool 통계와 최종 완료 메시지 |
| `dev` | 장치 기록의 마감 요청 | 개발자 모드 요약 |

프로토콜의 판정 `state`는 모델에 전달하는 문서 입력이다. 모델 cache는 이 입력을 처리하며 계산한 상태를 담는다. Choice 질문은 답 후보를 지정하고, yes-or-no 질문은 조건에 대한 확률을 요청한다.

CLI는 쪽 구조를 처리하고 문단과 요소 배치를 구성한 뒤 띄어쓰기를 복원하고, 번역 전체를 generation 요청 하나로 제출한다. 완료된 번역은 `key`로 원래 문단에 연결한다.

## 판정 실행

D3 prompt는 질문과 선택지를 나열한 header, 문서 입력, 답 코드가 지정된 질문을 포함한다. 읽는 순서 요청은 쪽마다 고정한 질문과 선택지 집합을 사용하며, 모든 본문 블록을 ID로 식별한다.

입력은 고정 부분과 단계별로 바뀌는 부분으로 나눈다.

| 구간 | 내용 |
|---|---|
| 고정 부분 | 쪽 크기, 블록 설명과 이전 pass의 위반 |
| 단계별 부분(`moving`) | 선택한 블록, 직전 블록 텍스트의 끝부분과 다음에 올 수 있는 블록 |

첫 pass에서 worker는 고정 prompt prefix를 계산해 cache에 저장하고, 각 suffix를 이 cache를 사용해 계산한다. 저장된 prefix는 단계별 suffix에 대한 attention 없이 계산된다. 두 번째 pass는 전체 prompt를 다시 계산한다.

Cache는 고정 입력과 질문 집합에 연결한다. Prefix가 다른 요청은 cache를 교체한다. `moving` 부분이 없는 판정 요청과 generation은 cache를 해제한다. 쪽 이미지를 포함하는 요청은 전체 prompt를 계산한다.

판정 요청이 사용 가능한 GPU 메모리를 넘으면 worker는 질문을 두 묶음으로 나눠 각 묶음을 다시 실행한다. 필요한 만큼 이 분할을 반복한다. 질문 하나도 처리할 수 없으면 오류를 반환한다.

## 생성 실행

Worker는 checkpoint의 chat template으로 각 요청을 구성하고 thinking을 비활성화한다. Causal backbone이 hidden state를 생성하고, BF16 Qwen3.5-27B vocabulary head가 마지막 hidden state를 token logits로 projection한다. Head tensor의 shape은 `[248320, 5120]`이다.

Decoding은 다음 두 가지를 적용한 뒤 logit이 가장 높은 token을 선택한다.

| 적용 항목 | 규칙 |
|---|---|
| DRY 반복 penalty | 행의 출력에서 32 token 이상 이어진 반복을 한 번 더 잇는 token은 logit에서 `0.8 × 1.75^(반복 길이 − 32)`만큼 깎인다. 줄바꿈, `:`, `"`, `*`를 포함한 token을 넘어서는 반복은 이어진 것으로 세지 않는다. 허용 길이보다 짧은 구절의 반복에는 penalty가 없다 |
| Pattern | Pattern이 있는 요청은 llguidance가 출력을 정규식 안에 두는 token을 계산하고, 나머지 token은 제외한다 |

요청은 continuous batching으로 decode step을 공유한다. 각 활성 행은 generation 요청 하나를 나타내며, token sequence, position, attention mask, KV cache, recurrent state와 convolution state를 유지한다. 최대 행 수는 설정한 batch size이며 기본값은 16이다.

행은 `<|im_end|>` 또는 모델의 종료 token을 생성하거나 출력 예산에 도달하면 완료된다. Worker는 텍스트, 종료 사유와 token 수를 반환한 뒤 모든 cache 구성 요소에서 해당 행을 제거한다.

대기 요청은 prompt가 긴 순서로 정렬한다. Prefill은 왼쪽 padding과 행별 position을 사용하며, forward pass 하나의 예산은 padding을 포함한 query token 2,048개다. 더 긴 입력은 causal cache를 유지하며 여러 forward로 나눠 처리한다. 새 행은 활성 pool의 KV 길이에 맞춘 뒤 합류한다. 모든 행에서 mask 처리된 선두 KV 위치는 제거한다.

대기 요청이 있는 동안 scheduler는 decode step마다 비어 있는 batch slot 수를 누적한다. 이 값이 설정한 batch size에 도달하거나 활성 pool이 비면 새 행의 합류를 허용한다.

### 메모리 admission

Prefill 전에 worker는 기존 행과 합류할 행에 필요한 상태 크기를 추정한다. Model configuration으로 token당 KV byte 수와 요청당 recurrent·convolution byte 수를 정한다. Dense batch의 KV 할당은 행들 중 가장 긴 prompt와 출력 예산을 기준으로 계산한다.

기본 상태 예산은 5 GiB다. Admission은 임시 계산을 위해 사용 가능한 GPU 메모리 중 4 GiB를 추가로 확보한다. Admission 예산을 넘는 요청은 활성 행이 완료되기를 기다린다. Pool의 메모리가 부족해지면 미완료 요청을 더 작은 batch로 다시 시작한다. 단독으로도 수용할 수 없는 요청은 필요한 상태 크기와 메모리 부족량을 반환한다.

### Packed INT4 연산

Device library는 입력 shape과 dtype에 따라 행렬곱 구현을 선택한다. 행렬의 한 행은 입력 token 하나를 나타내며, prefill 요청 하나는 여러 행을 구성할 수 있다.

| 입력 행 수 | 구현 |
|---|---|
| 1 | Packed INT4 가중치를 읽는 dot-product kernel |
| 2–512 | Weight tile을 공유 메모리로 dequantize하고 입력 행들이 재사용하는 matrix kernel |
| 그 밖의 shape 또는 지원하지 않는 dtype·bias | Checkpoint의 linear layer |

Dequantization은 BF16으로 `(nibble - zero_point) * scale`을 계산한다. Matrix kernel은 BF16 matrix instruction과 FP32 누산을 사용한다. Dispatch 조건은 `rdna4-w4a16`의 `kernels/int4_linear.hip`에 정의한다. [tools/build.sh](../../tools/build.sh)는 그 저장소에서 라이브러리를 빌드해 `librdna4_w4a16.so`를 실행 파일 옆에 복사하고, worker가 그 파일을 로드한다.

## 개발자 모드

`-dev`를 켜면 device library가 실행 span과 행렬곱 횟수를 기록한다. Worker는 각 span을 해당 phase의 이름으로 연다.

| Phase | Span |
|---|---|
| `decode` | 활성 generation pool의 decode step 하나 |
| `prefill` | 합류할 요청의 prefill과 cache 통합 |
| `decision_state` | 판정의 state pass 하나 |
| `decision_suffix` | 질문 블록 pass 또는 저장된 prefix를 사용하는 suffix 계산 |

HIP event로 span의 실행 시간과 vocabulary head 연산을 측정한다. INT4 연산은 phase, 행렬 차원 M/K/N, dtype, 실행 경로와 fallback 사유별로 센다. 연산별 시간은 각 phase에서 N개 span마다 한 번 측정하며, `-dev-every`로 N을 정한다. ROCTx range는 외부 profiler가 이 span을 식별하도록 표시한다.

Recorder는 시작 시 event overhead를 측정한다. 연산별 시간에서는 event 쌍의 overhead를 빼고, 기록 reader는 연산별 시간을 측정한 span에서 누적 instrumentation overhead를 뺀다. CLI의 전체 실행 시간에는 instrumentation이 포함된다.

출력은 실행 metadata, span 기록과 요약을 담은 JSONL이다. `rdna4-w4a16`의 `kernels/instrument.h`, `kernels/instrument.cpp`, `record/src/spec.rs`가 형식을 정의한다. 그 저장소의 Rust 기록 reader는 기록 구조를 검증하고 span에서 계산한 합계와 요약을 대조한다.
