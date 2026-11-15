# Dplus Architecture

Dplus reconstructs a PDF's reading order and paragraph boundaries, translates the resulting text, and writes a continuous document. Figures, tables, captions and footnotes occupy positions between paragraphs. Page numbers and source blocks remain attached to the output as metadata.

## Components

| Component | Implementation | Responsibility |
|---|---|---|
| Pipeline | Go, [cmd/dplus](cmd/dplus) and [internal](internal) | Extract page content, construct blocks, request structural decisions, assemble paragraphs and write output files |
| Model worker | Python, [internal/worker/backbone.py](internal/worker/backbone.py) | Load the shared backbone, answer decision requests and generate translations |
| Device library | HIP and C++, the `rdna4-w4a16` repository | Execute packed INT4 matrix products and record device timings |

The pipeline uses pdfium through `go-pdfium` to extract characters and render pages. Heron and the Korean spacing model run through the Go bindings for ONNX Runtime. The CLI loads the ONNX Runtime shared library from the executable directory.

The worker owns one GPU-resident d3-ultra-INT4 backbone. Decisions use its answer-code readout under `prefix_bidir` attention. Translation uses the same backbone with causal attention and the Qwen3.5-27B vocabulary head. Each execution path maintains its own computational state.

## Document representation

| Object | Contents |
|---|---|
| Page | Page number, dimensions, text direction, PDF object counts and extracted characters |
| Block | Text, bounding box, page number, Heron `kind` and confidence `score`, text source, assigned `role` and optional figure path |
| Paragraph | Joined source text, block type, its confidence score and role, source pages and block IDs, restored text, translation, translation check and optional figure path |

A block's `kind` is its Heron classification. Its `role` determines how it participates in the output:

| Role | Block types | Processing |
|---|---|---|
| Body | `title`, `section_header`, `text`, `list_item`, `formula`, `code` | Determine reading order and join continuing blocks into paragraphs |
| Inserted element | `picture`, `table`, `caption`, `footnote` and other types | Assign a position between body paragraphs |
| Dropped | `page_header`, `page_footer` | Exclude from the serialized document |

The decision model further classifies `text` blocks as running text or inserted elements. Subsequent structural decisions receive the Heron classification, confidence score and running-text classification.

## Processing stages

### Page classification and block extraction

The gate classifies each page as `healthy`, `scanned`, `corrupt` or `blank` using PDF object counts and extracted characters. Healthy pages proceed to block extraction. The output records the classification of skipped pages.

Heron detects regions in a rendered page image and returns bounding boxes, block types and confidence scores. The pipeline converts image coordinates to page coordinates and resolves overlapping detections:

1. For competing detections of the same region, retain the higher-confidence detection.
2. Retain a table or picture container over detections contained within it.
3. Remove other detections mostly covered by a higher-confidence region.

Each text-layer character is assigned to the highest-confidence retained region containing its center. Characters outside all retained regions are counted in the page report. The block builder groups characters into lines and preserves table row boundaries. The overlap predicates and thresholds are defined in [internal/layout/blocks.go](internal/layout/blocks.go).

For figures, the pipeline renders the page at 200 DPI, crops the detected region and saves it as a PNG file. The block and its output paragraph reference that file.

### Body classification and reading order

The pipeline submits one decision request per page to classify its `text` blocks as running text. It then orders the body blocks by repeatedly asking which block should be read next.

Reading-order validation uses two pairwise rules for a document with left-to-right text and top-to-bottom block progression:

1. For vertically separated blocks that overlap horizontally, the upper block precedes the lower block.
2. A block entirely to the left of another precedes it, unless a block between their vertical centers overlaps both horizontally.

Bounding boxes are rotated to the page's text direction before validation. Pictures and tables participate as separators for the second rule. The geometry predicates and tolerances are defined in [internal/order/order.go](internal/order/order.go).

At each selection step, eligible blocks are the unread body blocks whose predecessors under rule 1 have been read. A single eligible block is selected directly. Otherwise, the decision model ranks the options and the pipeline selects the eligible block with the highest probability.

The completed order is checked against both rules. Each violation identifies the pair of blocks in conflict. A page with violations receives a second decision pass with those violations included in the input. If violations remain, the page report retains the order and sets the review flag.

### Paragraph construction

The pipeline asks whether each boundary between consecutive body blocks continues the same sentence or paragraph. These questions are grouped by page. A boundary across pages is evaluated after both reading orders are available.

Titles and section headers establish paragraph boundaries. Other boundary questions include the types and text of the adjacent blocks. Continuing blocks are joined across columns, pages and intervening inserted elements. Each resulting paragraph retains its source block IDs and page numbers.

### Inserted-element placement

Placement has three stages:

1. **Caption matching.** For each caption, the model selects a table or figure on the same page, or selects no match. Matched captions form a group with the target and precede it within that group.
2. **Position selection.** For each element or group, the model selects a position after a body paragraph represented on the element's page. The first page also permits a position before the document's first body paragraph. On pages with no body paragraph, elements follow the preceding body text.
3. **Local ordering.** When multiple elements share a position, the model selects their sequence through successive next-element questions.

Position-selection requests include the full text of the elements and candidate paragraphs, including reference marks and footnote numbers. Caption matching and position selection each group a page's questions into one request.

### Spacing and translation

The pipeline determines each paragraph's source language from the `-from` option or its character scripts, then restores spacing:

| Language | Processing |
|---|---|
| Korean | Predict word boundaries with the ONNX spacing model, then apply the spacing lexicon |
| Chinese and Japanese | Remove spaces between East Asian characters |
| Other languages | Preserve extracted spacing |

Translatable paragraph types include titles, section headers, body text, list items, captions, footnotes, tables, document indexes, key-value regions and forms. Paragraphs already in the target language retain their source text.

Translation requests contain a system message specifying the source and target languages, the restored source text and an output token budget. The system message requests the translation alone, with numbers, names, reference marks and formulas preserved, in the plain written style of reports and papers. Every paragraph receives the same style instruction because each is translated independently. The message also names the paragraph's block type with its confidence score and states the form a translation of that type keeps:

| Paragraph type | Form of the translation |
|---|---|
| `title`, `section_header` | A concise heading rather than a sentence; section numbers unchanged |
| `text` in the body | Every sentence in order as continuous prose |
| `text` classified as an inserted element | Every sentence in order, as a passage set apart from the running text |
| `list_item` | The bullet, number or letter kept at the start |
| `caption` | The label word, such as Figure or Table, in the target language's usual form; its number unchanged |
| `footnote` | The marker kept at the start |
| `table` | Cells in reading order with line breaks kept; every cell translated; numbers, units and codes unchanged |
| `document_index`, `key_value_region`, `form` | Entries and labels translated; numbers, dates, codes and page numbers unchanged |

A paragraph takes the type and score of its first block, or of the first prose block that continues a formula or code block. The budget is four tokens per source character, bounded to 64–4,096 tokens. Each paragraph is translated independently.

Each request also carries a pattern, a regular expression its whole output must match: characters of the scripts the target language is written in (Hangul for Korean, Han for Chinese, Han and kana for Japanese), Latin and Greek letters, and the characters all scripts share, such as digits, punctuation and symbols. Generation selects no token outside the pattern, so a translation contains no text in another script.

### Translation check

After generation, each translation is compared with its restored source text:

| Measure | Definition |
|---|---|
| Reference marks | The source's distinct digit runs, with full-width digits read as ASCII and thousands separators removed, and footnote symbols `*`, `†`, `‡`, `§` and `¶`. Marks absent from the translation are listed as missing |
| Relative length | The translation's characters per source character, divided by the median of the run's translations from the same source language. Sources of at least 40 characters are judged; a relative length below 0.5 marks the translation as short |
| Stop reason | End token or output budget, as reported by the worker |

The paragraph records its check. The run summary tallies paragraphs, stop reasons, marks, missing marks and short translations by paragraph type, inserted elements apart from body paragraphs, and by source length: under 100, 100–399, 400–1,599 and 1,600 or more characters.

### Output

The Markdown output contains source paragraphs and translations in reading order, with block types and inserted elements marked. Figure references use relative image paths or embedded base64 data, according to `-embed-pictures`.

The JSON report retains page classifications, block geometry and classifications, reading-order decisions and review flags, paragraph provenance, figure paths, translation outcomes and checks, and runtime statistics.

## Worker protocol

The Go client starts the Python worker on first use. Requests and replies are JSON objects, one per line, over stdin and stdout. The worker first reports readiness and its device information. Each request carries an ID; each reply carries the corresponding ID. Closing stdin terminates the worker.

| Operation | Input | Responses |
|---|---|---|
| `decide` | Document facts in `state`, typed questions and optional page images; reading-order requests can also provide `moving` and `whole` | Answers, including selected options or probabilities |
| `generate` | Paragraph requests identified by `key`, batch settings and optional text-streaming or comparison settings | Completed paragraph results, optional text deltas, pool statistics and a final completion message |
| `dev` | Request to finalize the device record | Developer-mode summary |

The protocol's decision `state` is the document input supplied to the model. Model caches contain the computational state produced while processing that input. Choice questions identify candidate answers; yes-or-no questions request a probability for a condition.

The CLI processes page structure, assembles paragraphs and placement, restores spacing, then submits the translations as one generation request. Completed translations are associated with their original paragraphs through `key`.

## Decision execution

A d3 prompt contains a header listing questions and options, the document input, and the question with coded answers. Reading-order requests use a fixed question and option set for a page, with all body blocks identified by ID.

The input is divided into a fixed portion and a step-specific portion:

| Portion | Contents |
|---|---|
| Fixed | Page dimensions, block descriptions and any violations from the preceding pass |
| Step-specific (`moving`) | Selected blocks, the end of the preceding block's text and eligible next blocks |

During the first pass, the worker computes and caches the fixed prompt prefix, then evaluates each suffix against that cache. The cached prefix is computed without attention to the step-specific suffix. The second pass recomputes the full prompt.

The cache is associated with the fixed input and question set. A request with a different prefix replaces it. Decision requests without a `moving` portion release the cache, as does generation. Requests containing page images use full-prompt evaluation.

If a decision request exceeds available GPU memory, the worker splits its questions into two groups and retries each group. This subdivision repeats as needed. A single question that cannot be evaluated returns an error.

## Generation execution

The worker formats each request with the checkpoint's chat template and disables thinking. The causal backbone produces hidden states; the BF16 Qwen3.5-27B vocabulary head projects the final hidden state to token logits. The head tensor has shape `[248320, 5120]`.

Decoding selects the token with the highest logit after two adjustments:

| Adjustment | Rule |
|---|---|
| DRY repetition penalty | A token that would extend a repeat of at least 32 tokens of the row's output loses `0.8 × 1.75^(repeat length − 32)` from its logit. A repeat does not extend across a token containing a line break, `:`, `"` or `*`. Repeats of phrases shorter than the allowed length are not penalized |
| Pattern | For a request with a pattern, llguidance computes the tokens that keep the output within the regular expression, and all other tokens are excluded |

Requests share decode steps through continuous batching. Each active row represents one generation request and maintains its token sequence, positions, attention mask, KV cache, recurrent state and convolution state. The maximum row count is the configured batch size, which defaults to 16.

A row completes when it produces `<|im_end|>` or the model's end token, or exhausts its output budget. The worker returns the text, stop reason and token counts, then removes the row from every cache component.

Waiting requests are ordered by prompt length, longest first. Prefill uses left padding and per-row positions, with a budget of 2,048 padded query tokens per forward pass. Longer inputs are split across forwards using the causal cache. New rows are aligned to the active pool's KV length before joining. Leading KV positions masked in every row are removed.

While requests wait, the scheduler accumulates the number of unused batch slots across decode steps. New rows are admitted when this count reaches the configured batch size, or when the active pool is empty.

### Memory admission

Before prefill, the worker estimates the state required by existing and incoming rows. The model configuration determines KV bytes per token and recurrent and convolution bytes per request. Dense-batch KV allocation uses the longest prompt plus output budget among the rows.

The default state budget is 5 GiB. Admission also reserves 4 GiB of available GPU memory for temporary computation. Requests exceeding the admission budget wait for active rows to finish. If a pool runs out of memory, its unfinished requests restart with a smaller batch. A request that cannot fit alone returns the required state size and memory deficit.

### Packed INT4 operations

The device library selects a matrix-product implementation from the input shape and dtype. One matrix row represents one input token; a prefill request can contribute multiple matrix rows.

| Input rows | Implementation |
|---|---|
| 1 | Dot-product kernel reading packed INT4 weights |
| 2–512 | Matrix kernel that dequantizes weight tiles into shared memory and reuses them across input rows |
| Other shapes or unsupported dtype or bias | Checkpoint linear layer |

Dequantization computes `(nibble - zero_point) * scale` in BF16. The matrix kernel uses BF16 matrix instructions with FP32 accumulation. Dispatch conditions are defined in `kernels/int4_linear.hip` of `rdna4-w4a16`. [tools/build.sh](tools/build.sh) builds the library from that repository and copies `librdna4_w4a16.so` beside the executable, where the worker loads it.

## Developer mode

With `-dev`, the device library records execution spans and matrix-product counts. The worker opens each span under the name of its phase:

| Phase | Span |
|---|---|
| `decode` | One decode step of the active generation pool |
| `prefill` | Prefill and cache integration for incoming requests |
| `decision_state` | A decision state pass |
| `decision_suffix` | A question-block pass or evaluation of a suffix against a cached prefix |

HIP events measure span duration and vocabulary-head operations. INT4 operations are counted by phase, matrix dimensions M/K/N, dtype, execution path and fallback reason. Per-operation timings are sampled once every N spans of each phase, where `-dev-every` sets N. ROCTx ranges identify these spans for external profilers.

The recorder calibrates event overhead at startup. Per-operation timings subtract event-pair overhead; the record reader subtracts the accumulated instrumentation overhead from sampled span timings. CLI elapsed time includes instrumentation.

The output is JSONL containing run metadata, span records and a summary. `rdna4-w4a16` defines the format in `kernels/instrument.h`, `kernels/instrument.cpp` and `record/src/spec.rs`. Its Rust record reader validates record structure and checks the summary against totals calculated from the spans.
