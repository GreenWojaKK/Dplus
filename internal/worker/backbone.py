"""Dplus inference worker: one copy of d3-ultra-INT4 in device memory serves both decisions and generation.

Decisions go through the checkpoint's own `system_one`. Generation runs the same language model on its causal path and
multiplies the hidden states by the vocabulary head of Qwen3.5-27B.

Protocol: one JSON object per line on stdin, replies on stdout. The first reply reports readiness. The worker ends when
stdin closes.

A `decide` request gets one reply. With a `moving` part beside its state, the state is the two joined, and the prompt
up to the end of the first part is computed once for the requests that share it. A `generate` request carries a list
of requests and gets several replies: one for each request as it finishes, its text as it forms when deltas are asked
for, one with what the pool of generations took, and a last one that ends the list.

A generation takes the most probable token at each step, after the DRY repetition penalty and, when its request
carries a `pattern`, among the tokens that keep its text within that regular expression.

A `dev` request ends the developer mode and gets its summary.

Arguments: the d3-ultra-INT4 folder, the safetensors file holding `lm_head.weight`, the device, the library of device
kernels, which may be missing, and for the developer mode the record file to write and how many spans of a phase
share one whose int4 products are timed.
"""

import contextlib
import json
import math
import os
import sys
import time

# Anything a library prints must stay off the reply stream.
replies = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)

# ROCm under WSL rejects the allocations expandable segments make. This has to be set before torch starts.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import numpy  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

# The DRY repetition penalty: a token that would extend a repeat of at least DRY_ALLOWED tokens of the output loses
# DRY_MULTIPLIER * DRY_BASE ** (repeat length - DRY_ALLOWED) from its logit. A repeat does not run across a token
# holding one of DRY_BREAKERS. Multiplier, base and breakers are those of vLLM's DRY example and llama.cpp. A text may
# repeat a phrase as often as it needs, and phrases are bounded in length where a loop is not: the allowed length
# stands above the phrases texts repeat, so that only a loop is penalized.
DRY_MULTIPLIER = 0.8
DRY_BASE = 1.75
DRY_ALLOWED = 32
DRY_BREAKERS = ("\n", ":", '"', "*")
# The exponent's bound, as in llama.cpp: the largest whose power a float32 holds.
DRY_MAX_EXPONENT = int(numpy.float32(88.7228391) / numpy.log(numpy.float32(DRY_BASE)))

# Device memory one pass of a decision's questions may take to copy the shared state's keys and values into its rows.
QUESTION_PASS_BYTES = 1 << 30

# Device memory the states of the generations in the pool may take, and what stays free beside them for the forward pass.
STATE_BUDGET = 5 << 30
WORKSPACE = 4 << 30
# Padded prompt tokens one prefill forward takes. More are prefilled in several forwards along the sequence.
PREFILL_TOKENS = 2048
# How many of the most probable first tokens a probe reports.
PROBE_TOP = 5


def send(message):
    replies.write(json.dumps(message, ensure_ascii=False) + "\n")


def memory(device):
    if device.type != "cuda":
        return {}
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20),
        "peak_mib": round(torch.cuda.max_memory_allocated(device) / 2**20),
    }


def dry_penalties(window, breakers):
    """The DRY penalty of each token that would extend a repeat within the window, by token.

    Adapted from `_dry_penalties` of vLLM's DRY example (examples/features/logits_processor/dry.py, Apache-2.0), a
    port of the scan in llama.cpp's `llama_sampler_dry_apply`: a Z-algorithm run from the end finds, for each
    position, how long the tokens ending there match the end of the window; the token after such a match is charged
    for the longest repeat it would extend.
    """
    size = len(window)
    if size <= DRY_ALLOWED:
        return {}

    def back(i):  # back(0) is the last token
        return window[size - 1 - i]

    # The nearest breaker from the end bounds the repeat.
    limit = size
    for i in range(size):
        if back(i) in breakers:
            limit = i
            break
    if limit < DRY_ALLOWED:
        return {}
    matched = [0] * size
    last = size - 1
    left = right = 0
    for k in range(1, size):
        if k > right:
            z = 0
            while z + k < size and back(z) == back(z + k):
                z += 1
            matched[last - k] = min(z, limit)
            if z > 0:
                left, right = k, k + z - 1
        else:
            inside = right - k + 1
            if matched[last - (k - left)] < inside:
                matched[last - k] = min(matched[last - (k - left)], limit)
            else:
                j = right + 1
                while j < size and back(j) == back(j - k):
                    j += 1
                matched[last - k] = min(j - k, limit)
                left, right = k, j - 1
    longest = {}
    for i in range(size - 1):
        if matched[i] >= DRY_ALLOWED and longest.get(window[i + 1], -1) < matched[i]:
            longest[window[i + 1]] = matched[i]
    return {
        token: DRY_MULTIPLIER * DRY_BASE ** min(length - DRY_ALLOWED, DRY_MAX_EXPONENT)
        for token, length in longest.items() if token not in breakers
    }


def release_page_cache(*paths):
    """Drops the checkpoint's pages from the page cache: under WSL they count against the Windows commit limit."""
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY)
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(descriptor)


class Recorder:
    """The developer mode of the kernel library, when it is on: it records where the device's time goes.

    A span is one piece of work, named after its phase: `decode` for a decode step, `prefill` for a prefill,
    `decision_state` for a decision's state pass, and `decision_suffix` for the rest of a decision, which is the end
    of its state and its question blocks. The library times each span on the device and counts the int4 products
    inside it with their rows, inputs, outputs, dtype and path.
    """

    def __init__(self):
        self.library = None

    def start(self, library, record, every):
        """Starts recording into the file. One span in every `every` of a phase has its int4 products timed."""
        library.record_begin(record, every)
        self.library = library

    @contextlib.contextmanager
    def span(self, phase, rows, tokens):
        if self.library is None:
            yield
            return
        self.library.span_begin(phase, rows, tokens)
        try:
            yield
        finally:
            self.library.span_end()

    def head(self, hidden, head):
        """The logits of the hidden states [rows, size] under the vocabulary head: one product of a span."""
        if self.library is None:
            return hidden @ head.T
        from rdna4_w4a16 import Dtype, Op, Path, Reason
        dtype = {torch.bfloat16: Dtype.BF16, torch.float16: Dtype.F16, torch.float32: Dtype.F32}.get(hidden.dtype, Dtype.OTHER)
        self.library.op_begin(Op.HEAD, Path.DENSE, Reason.NONE, hidden.shape[0], head.shape[1], head.shape[0], dtype)
        try:
            return hidden @ head.T
        finally:
            self.library.op_end()

    def finish(self):
        """Ends the mode and gives its summary, or None when it is off."""
        if self.library is None:
            return None
        summary = self.library.record_end()
        self.library = None
        return summary


recorder = Recorder()


def install_packed_linear(path):
    """Makes the int4 layers multiply up to 512 rows of input straight from their packed weights: the rows of a decode
    step, and the tokens of a short prompt or of the end of a decision prompt. Returns the kernel library.

    The checkpoint's layer unpacks its whole weight to BF16 on every call and multiplies. Up to a few hundred rows
    most of that time goes to writing and reading the unpacked weight. The kernel reads the packed words once and
    uses the same BF16 weights, so its outputs differ from the layer's only where sums taken in another order round
    the last bit the other way. With more rows the layer's own multiplication is the faster one.

    The library chooses the path of each product from its rows, sizes, dtype and bias, and names the reason when the
    product is the checkpoint's layer's to run. In the developer mode that product is recorded with its reason.
    """
    # The build distributes the binding beside the library; the worker itself is embedded in Go.
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    from rdna4_w4a16 import Library, Dtype, Op, Path, Reason, Status
    library = Library(path)
    dtypes = {torch.bfloat16: Dtype.BF16, torch.float16: Dtype.F16, torch.float32: Dtype.F32}
    layers = sys.modules["d3q.kernels"].Int4Linear
    unpack_and_multiply = layers.forward

    def forward(layer, x):
        rows = x.numel() // layer.in_features
        kind = dtypes.get(x.dtype, Dtype.OTHER)
        route = library.route(layer.out_features, layer.in_features, layer.group_size, rows, kind, layer.bias is not None)
        if route > 0:
            flat = x.reshape(rows, layer.in_features).contiguous()
            y = torch.empty(rows, layer.out_features, dtype=torch.bfloat16, device=x.device)
            zero = layer.weight_zero_point
            route = library.raw.w4a16_linear(
                layer.weight_packed.data_ptr(), layer.weight_scale.data_ptr(), None if zero is None else zero.data_ptr(),
                flat.data_ptr(), y.data_ptr(), layer.out_features, layer.in_features, layer.group_size, rows,
            )
            if route == Status.OK:
                return y.view(*x.shape[:-1], layer.out_features)
        if route == Status.ARGUMENT:
            library.check(route, "linear")
        if recorder.library is None:
            return unpack_and_multiply(layer, x)
        library.op_begin(Op.INT4, Path.UNPACK, Reason(-route), rows, layer.in_features, layer.out_features, kind)
        try:
            return unpack_and_multiply(layer, x)
        finally:
            library.op_end()

    layers.forward = forward
    return library


def tail_attention(runtime, attention, hidden, rope, entry, state_tokens):
    """Attention of the tail of a prompt on the keys and values of its start. The tail is the end of the state and
    then the question block. As in the model's own layout, the state does not read the question block; the question
    block reads everything. This is the runtime's `StateOnce._attention` with that mask."""
    shape = hidden.shape[:-1]
    heads = (*shape, -1, attention.head_dim)
    query, gate = torch.chunk(attention.q_proj(hidden).view(*shape, -1, attention.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*shape, -1)
    query = attention.q_norm(query.view(heads)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden).view(heads)).transpose(1, 2)
    value = attention.v_proj(hidden).view(heads).transpose(1, 2)
    query, key = runtime.qwen.apply_rotary_pos_emb(query, key, *rope)
    keys, values = torch.cat([entry[0], key], dim=2), torch.cat([entry[1], value], dim=2)
    start, length = entry[0].shape[2], shape[1]
    mask = torch.ones(1, 1, length, start + length, dtype=torch.bool, device=hidden.device)
    mask[:, :, :state_tokens, start + state_tokens :] = False
    groups = query.shape[1] // keys.shape[1]
    if groups > 1:
        keys, values = keys.repeat_interleave(groups, dim=1), values.repeat_interleave(groups, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(query, keys, values, attn_mask=mask, scale=attention.scaling)
    return attention.o_proj(out.transpose(1, 2).reshape(*shape, -1) * torch.sigmoid(gate))


class Row:
    """One generation request and what it has produced. `matcher` gives a new matcher of the request's pattern, which
    follows the output and tells the tokens that may come next; None when the request has no pattern."""

    def __init__(self, request, ids, matcher=None):
        self.key = request["key"]
        self.ids = ids
        self.budget = max(int(request["max_tokens"]), 1)
        self.new_matcher = matcher
        self.restart()

    def restart(self):
        self.tokens, self.stop, self.sent, self.first_top = [], None, "", None
        self.joined, self.seconds, self.reported = 0.0, 0.0, False
        self.matcher = self.new_matcher() if self.new_matcher else None

    def take(self, token, stops):
        """Adds the next token and sets the stop reason when the generation ends with it."""
        if token in stops:
            self.stop = "eos"
            return
        self.tokens.append(token)
        if self.matcher is not None and not self.matcher.consume_token(token):
            raise RuntimeError(f"a token outside the pattern was chosen: {self.matcher.get_error()}")
        if len(self.tokens) >= self.budget:
            self.stop = "length"


def recurrent(layer):
    """Whether a cache layer holds the states of a Gated DeltaNet layer. The others hold keys and values."""
    return hasattr(layer, "recurrent_states")


def keep_rows(cache, rows):
    """Keeps the given rows of every state in the cache."""
    for layer in cache.layers:
        if recurrent(layer):
            for states in (layer.conv_states, layer.recurrent_states):
                for index, state in states.items():
                    if state is not None:
                        states[index] = state[rows]
        else:
            layer.keys, layer.values = layer.keys[rows], layer.values[rows]


def drop_keys(cache, count):
    """Removes the first keys and values of every row."""
    for layer in cache.layers:
        if not recurrent(layer):
            layer.keys, layer.values = layer.keys[:, :, count:], layer.values[:, :, count:]


def left_pad(tensor, width):
    """Puts zeros in front of keys or values, [rows, heads, length, size], up to the width."""
    return torch.nn.functional.pad(tensor, (0, 0, width - tensor.shape[2], 0))


def add_rows(cache, other, width):
    """Adds the rows of another cache, with the keys and values of both padded in front to the width."""
    for layer, new in zip(cache.layers, other.layers):
        if recurrent(layer):
            for states, more in ((layer.conv_states, new.conv_states), (layer.recurrent_states, new.recurrent_states)):
                for index, state in states.items():
                    if state is not None:
                        states[index] = torch.cat([state, more[index]])
        else:
            layer.keys = torch.cat([left_pad(layer.keys, width), left_pad(new.keys, width)])
            layer.values = torch.cat([left_pad(layer.values, width), left_pad(new.values, width)])


class Pool:
    """Generations that share decode steps. A row leaves when it stops, and waiting requests are prefilled and take the
    free places.

    Every row holds its keys and values at the end of the cache with padding in front. Padding is masked and each row's
    positions count its own tokens, so a row computes what it would alone, up to the order of the arithmetic.
    """

    def __init__(self, backbone):
        self.backbone, self.rows = backbone, []
        self.past = self.mask = self.inputs = None
        self.most = self.steps = self.row_steps = self.joins = self.prompt_tokens = self.padded_tokens = self.output_tokens = 0
        self.prefill_seconds = self.decode_seconds = 0.0

    def held(self):
        """Device memory the states of the rows take now."""
        if not self.rows:
            return 0
        return len(self.rows) * (self.backbone.state_per_request + self.mask.shape[1] * self.backbone.state_per_token)

    def choose(self, rows, logits):
        """The token each row takes next, given its logits [rows, vocabulary]: the most probable one after the DRY
        penalty for extending a repeat of the row's output, among the tokens the row's pattern allows."""
        backbone = self.backbone
        logits = logits.float()
        at, tokens, penalties = [], [], []
        for index, row in enumerate(rows):
            for token, penalty in dry_penalties(row.tokens, backbone.breakers).items():
                at.append(index)
                tokens.append(token)
                penalties.append(min(penalty, 3.0e38))
        if at:
            logits[at, tokens] -= torch.tensor(penalties, dtype=logits.dtype, device=logits.device)
        backbone.constrain(rows, logits)
        return logits.argmax(-1)

    def take(self, rows, chosen):
        """Gives each row the token chosen for it. Returns the rows that stopped and the indices of the others."""
        stopped, running = [], []
        now = time.perf_counter()
        for index, (row, token) in enumerate(zip(rows, chosen.tolist())):
            row.take(token, self.backbone.stops)
            if row.stop is None:
                running.append(index)
            else:
                row.seconds = now - row.joined
                self.output_tokens += len(row.tokens)
                stopped.append(row)
        return stopped, running

    def trim(self):
        """Drops the leading keys that are padding in every row."""
        padding = int(self.mask.shape[1] - self.mask.sum(-1).max())
        if padding:
            drop_keys(self.past, padding)
            self.mask = self.mask[:, padding:]

    def join(self, rows, probe=False):
        """Prefills waiting rows together, gives each its first token and adds them to the pool. Returns the rows that
        stopped at that token."""
        with recorder.span("prefill", len(rows), len(rows) * max(len(row.ids) for row in rows)):
            return self.prefill(rows, probe)

    def prefill(self, rows, probe):
        began = time.perf_counter()
        backbone = self.backbone
        width = max(len(row.ids) for row in rows)
        ids = torch.full((len(rows), width), backbone.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            row.joined = began
            ids[index, width - len(row.ids) :] = torch.tensor(row.ids)
            mask[index, width - len(row.ids) :] = 1
        ids, mask = ids.to(backbone.device), mask.to(backbone.device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        columns = max(PREFILL_TOKENS // len(rows), 1)
        past = None
        for start in range(0, width, columns):
            end = min(start + columns, width)
            output = backbone.text(
                input_ids=ids[:, start:end], attention_mask=mask[:, :end], position_ids=positions[:, start:end],
                past_key_values=past, use_cache=True,
            )
            past = output.past_key_values
        logits = recorder.head(output.last_hidden_state[:, -1], backbone.head)
        if probe:
            top = logits.float().topk(PROBE_TOP)
            for row, tokens, values in zip(rows, top.indices.tolist(), top.values.tolist()):
                row.first_top = [{"token": token, "logit": round(value, 4)} for token, value in zip(tokens, values)]
        chosen = self.choose(rows, logits)
        stopped, running = self.take(rows, chosen)
        if running:
            if stopped:
                index = torch.tensor(running, device=backbone.device)
                keep_rows(past, index)
                mask, chosen = mask[index], chosen[index]
            if self.rows:
                keys = max(self.mask.shape[1], mask.shape[1])
                add_rows(self.past, past, keys)
                pad = torch.nn.functional.pad
                self.mask = torch.cat([pad(self.mask, (keys - self.mask.shape[1], 0)), pad(mask, (keys - mask.shape[1], 0))])
                self.inputs = torch.cat([self.inputs, chosen])
            else:
                self.past, self.mask, self.inputs = past, mask, chosen
            self.rows = self.rows + [rows[index] for index in running]
            self.trim()
        self.joins += 1
        self.most = max(self.most, len(self.rows))
        self.prompt_tokens += sum(len(row.ids) for row in rows)
        self.padded_tokens += len(rows) * width
        self.prefill_seconds += time.perf_counter() - began
        return stopped

    def step(self):
        """Runs one decode step, gives each row its next token and lets go of the rows that stop. Returns those rows."""
        with recorder.span("decode", len(self.rows), len(self.rows)):
            return self.decode()

    def decode(self):
        began = time.perf_counter()
        backbone = self.backbone
        self.mask = torch.cat([self.mask, self.mask.new_ones(len(self.rows), 1)], dim=1)
        positions = torch.tensor([len(row.ids) + len(row.tokens) - 1 for row in self.rows], device=backbone.device)
        output = backbone.text(
            input_ids=self.inputs[:, None], attention_mask=self.mask, position_ids=positions[:, None],
            past_key_values=self.past, use_cache=True,
        )
        self.past = output.past_key_values
        chosen = self.choose(self.rows, recorder.head(output.last_hidden_state[:, -1], backbone.head))
        self.steps += 1
        self.row_steps += len(self.rows)
        stopped, running = self.take(self.rows, chosen)
        if not running:
            self.rows, self.past, self.mask, self.inputs = [], None, None, None
        elif stopped:
            index = torch.tensor(running, device=backbone.device)
            keep_rows(self.past, index)
            self.rows, self.mask, self.inputs = [self.rows[i] for i in running], self.mask[index], chosen[index]
            self.trim()
        else:
            self.inputs = chosen
        self.decode_seconds += time.perf_counter() - began
        return stopped

    def report(self):
        return {
            "rows": self.most, "steps": self.steps, "row_steps": self.row_steps, "joins": self.joins,
            "prompt_tokens": self.prompt_tokens, "padded_tokens": self.padded_tokens, "output_tokens": self.output_tokens,
            "prefill_seconds": round(self.prefill_seconds, 3), "decode_seconds": round(self.decode_seconds, 3),
            **memory(self.backbone.device),
        }


class Backbone:
    def __init__(self, model_path, head_path, device, kernels="", record="", every=0):
        self.device = torch.device(device)
        self.root = model_path
        sys.path.insert(0, model_path)
        import transformers.modeling_utils as loader
        from d3q.load import load_d3

        # The loader reserves the unquantized size up front, which ROCm rejects. The packed weights fit.
        loader.caching_allocator_warmup = lambda *args, **kwargs: None
        self.model = load_d3(model_path, device=device)
        self.tokenizer = self.model.tokenizer
        self.text = self.model.backbone.language_model
        # A pass of questions copies the state's keys and values, BF16 for every attention head, into each of its rows.
        # The runtime bounds those copies for devices far larger than this one; a smaller pass gives the same answers.
        config = self.text.config
        per_key = 2 * config.num_attention_heads * config.head_dim * 2
        sys.modules["d3_runtime"].STATE_ONCE_KEYS = QUESTION_PASS_BYTES // per_key
        # What a generation holds while it runs: keys and values per token in the full-attention layers, BF16, and in
        # each Gated DeltaNet layer the recurrent state, FP32, and the last inputs of the convolution, BF16.
        full = sum(kind == "full_attention" for kind in config.layer_types)
        self.state_per_token = full * 2 * config.num_key_value_heads * config.head_dim * 2
        recurrent = config.linear_num_value_heads * config.linear_key_head_dim * config.linear_value_head_dim * 4
        channels = 2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim
        convolution = channels * config.linear_conv_kernel_dim * 2
        self.state_per_request = (len(config.layer_types) - full) * (recurrent + convolution)
        with safe_open(head_path, "pt") as source:
            self.head = source.get_tensor("lm_head.weight").to(self.device, torch.bfloat16)
        release_page_cache(os.path.join(model_path, "model.safetensors"), os.path.join(model_path, "readout.safetensors"), head_path)
        # The start of the prompts that share a fixed part of their state: its key, tokens and the model's state after them.
        self.start = (None, None, None)
        self.packed_linear = bool(kernels) and os.path.exists(kernels) and self.device.type == "cuda"
        self.dev = False
        if self.packed_linear:
            library = install_packed_linear(kernels)
            if record:
                self.record(library, record, every)
        self.stops = {self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        if self.tokenizer.eos_token_id is not None:
            self.stops.add(self.tokenizer.eos_token_id)
        # The tokens a repeat does not run across, and what patterns need of the tokenizer; both made on first use.
        self.breaker_tokens = None
        self.guided, self.grammars = None, {}

    @property
    def breakers(self):
        """The tokens whose text holds one of DRY_BREAKERS."""
        if self.breaker_tokens is None:
            texts = self.tokenizer.batch_decode([[token] for token in range(len(self.tokenizer))])
            self.breaker_tokens = frozenset(
                token for token, text in enumerate(texts) if any(breaker in text for breaker in DRY_BREAKERS)
            )
        return self.breaker_tokens

    def matcher(self, pattern):
        """What makes the matchers of a request's pattern, a regular expression its whole output must match; None
        without a pattern."""
        if not pattern:
            return None
        import llguidance
        import llguidance.hf

        if self.guided is None:
            self.guided = llguidance.hf.from_tokenizer(self.tokenizer, eos_token=sorted(self.stops))
        if pattern not in self.grammars:
            grammar = llguidance.LLMatcher.grammar_from_regex(pattern)
            problem = llguidance.LLMatcher.validate_grammar(grammar, self.guided)
            if problem:
                raise ValueError(f"the pattern {pattern!r} cannot be used: {problem}")
            self.grammars[pattern] = grammar
        return lambda: llguidance.LLMatcher(self.guided, self.grammars[pattern])

    def constrain(self, rows, logits):
        """Leaves each row that has a pattern only the tokens that keep its output within the pattern."""
        constrained = [index for index, row in enumerate(rows) if row.matcher is not None]
        if not constrained:
            return
        vocabulary = self.guided.vocab_size
        bits = numpy.full((len(constrained), (vocabulary + 31) // 32), -1, dtype=numpy.int32)
        for place, index in enumerate(constrained):
            row = bits[place]
            rows[index].matcher.unsafe_compute_mask_ptr(row.ctypes.data, row.nbytes)
        bits = torch.from_numpy(bits).to(logits.device)
        allowed = ((bits[:, :, None] >> torch.arange(32, device=logits.device, dtype=torch.int32)) & 1).bool()
        allowed = allowed.reshape(len(constrained), -1)[:, :vocabulary]
        index = torch.tensor(constrained, device=logits.device)
        picked = logits[index]
        picked[:, :vocabulary].masked_fill_(~allowed, -math.inf)
        picked[:, vocabulary:] = -math.inf  # The head has more rows than the tokenizer has tokens.
        logits[index] = picked

    def record(self, library, record, every):
        """Starts the developer mode. A decision's state pass and its question passes become spans."""
        recorder.start(library, record, every)
        self.dev = True
        once = self.model.state_once
        state, questions = once.state, once.questions

        def timed_state(embeds, rope):
            with recorder.span("decision_state", 1, embeds.shape[1]):
                return state(embeds, rope)

        def timed_questions(cache, embeds, lengths, rope):
            with recorder.span("decision_suffix", embeds.shape[0], embeds.shape[0] * embeds.shape[1]):
                return questions(cache, embeds, lengths, rope)

        once.state, once.questions = timed_state, timed_questions

    def ready(self):
        with open(os.path.join(self.root, "MODEL_MANIFEST.json"), encoding="utf-8") as manifest:
            parameters = json.load(manifest)["parameters"]["loaded"]
        name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"
        return {
            "ready": True, "device": name, "parameters": parameters, "head_parameters": self.head.numel(),
            "packed_linear": self.packed_linear, "dev": self.dev, **memory(self.device),
        }

    @torch.inference_mode()
    def decide(self, request):
        state, result = request["state"], None
        if request.get("moving") is not None:
            state = {**state, **request["moving"]}
            if not request.get("images") and len(request["questions"]) == 1:
                result = self.decide_on_start(request["state"], state, request["questions"])
        if result is None:
            self.start = (None, None, None)  # A whole prompt needs the memory the kept start holds.
            result = self.answer(state, request["questions"], request.get("images"))
        return {"answers": result["answers"], "input_tokens": result["usage"]["input_tokens"], **memory(self.device)}

    def answer(self, state, questions, images):
        """Answers the questions about a state. A request that does not fit the device is asked again as two, each
        with half of the questions: the state is then read with fewer questions listed before it."""
        try:
            return self.model.system_one(state=state, questions=questions, images=images)
        except torch.OutOfMemoryError:
            if len(questions) < 2:
                raise
        torch.cuda.empty_cache()  # What the failed request held is free once its error is gone.
        names = list(questions)
        halves = [self.answer(state, {name: questions[name] for name in part}, images) for part in (names[: len(names) // 2], names[len(names) // 2 :])]
        return {
            "answers": {**halves[0]["answers"], **halves[1]["answers"]},
            "usage": {"input_tokens": sum(half["usage"]["input_tokens"] for half in halves)},
        }

    def decide_on_start(self, fixed, state, questions):
        """Answers one question about a state whose first part, `fixed`, other requests share.

        The prompt up to the end of the fixed part is computed once and kept: the runtime's own state pass over those
        tokens. Each request then runs only the rest of its prompt on it, the end of its state and its question block.
        Alone, the start cannot read the end of the state, which the whole prompt's state block would; this is the one
        difference from `system_one`. Returns None when this path does not apply.
        """
        model = self.model
        runtime = sys.modules["d3_runtime"]
        once = model.state_once
        if type(once) is not runtime.StateOnce:
            return None
        prepared = model.prepare(state, questions)
        if not prepared.runnable:
            return None
        name = prepared.runnable[0]
        sequence, state_tokens = prepared.sequences[name], prepared.states[name]
        key = json.dumps([fixed, questions], ensure_ascii=False)
        if self.start[0] != key:
            # Where the prompt of the fixed part alone parts from this one is where the fixed part ends.
            alone = model.prepare(fixed, questions).sequences[name]
            shared = next((i for i, (a, b) in enumerate(zip(sequence, alone)) if a != b), min(len(sequence), len(alone)))
            self.start = (key, sequence[: min(shared, state_tokens)], None)
        _, start, cache = self.start
        if not start or sequence[: len(start)] != start:
            return None
        embed = model.backbone.get_input_embeddings()
        rotary = model.backbone.language_model.rotary_emb
        with runtime.sdpa_backends(self.device), model.request_temperature(len(prepared.questions)):
            if cache is None:
                embeds = embed(torch.as_tensor([start], device=self.device))
                positions = torch.arange(len(start), device=self.device).view(1, 1, -1).expand(3, 1, -1)
                cache = once.state(embeds, rotary(embeds, positions))
                self.start = (key, start, cache)
            rest = sequence[len(start) :]
            with recorder.span("decision_suffix", 1, len(rest)):
                hidden = embed(torch.as_tensor([rest], device=self.device))
                positions = (len(start) + torch.arange(len(rest), device=self.device)).view(1, 1, -1).expand(3, 1, -1)
                rope = rotary(hidden, positions)
                present = torch.ones(1, len(rest), dtype=torch.bool, device=self.device)
                for (layer, softmax), entry in zip(once.stack, cache):
                    normed = layer.input_layernorm(hidden)
                    if softmax:
                        mixed = tail_attention(once, layer.self_attn, normed, rope, entry, state_tokens - len(start))
                    else:
                        mixed, _ = once._delta(layer.linear_attn, normed, entry, present)
                    hidden = hidden + mixed
                    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                last = once.language_model.norm(hidden[:, -1])
            if model.readout_dtype == "float32":
                logits = last.float() @ model.readout.T
            else:
                logits = torch.nn.functional.linear(last, model.readout).float()
            count = len(prepared.questions[name].keys)
            logits = logits.masked_fill(torch.arange(runtime.MAX_OPTIONS, device=self.device)[None] >= count, float("-inf"))
            probabilities = (logits / model.temperature).softmax(-1).cpu().tolist()[0][:count]
        return model.respond(prepared, {name: probabilities}, len(sequence))

    def encode(self, request):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": request["system"]}, {"role": "user", "content": request["user"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    def admit(self, pool, waiting, bound):
        """How many of the leading waiting rows fit beside the rows of the pool, the bytes the first needs alone, and
        the budget. Every row holds keys and values for the longest prompt and output budget among them."""
        budget = bound
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            cached = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            budget = min(bound, free + cached + pool.held() - WORKSPACE)
        longest = max((len(row.ids) + row.budget for row in pool.rows), default=0)
        count = 0
        for row in waiting:
            longest = max(longest, len(row.ids) + row.budget)
            if (len(pool.rows) + count + 1) * (self.state_per_request + longest * self.state_per_token) > budget:
                break
            count += 1
        alone = self.state_per_request + (len(waiting[0].ids) + waiting[0].budget) * self.state_per_token
        return count, alone, budget

    def text_of(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def emit(self, identifier, running, stopped, deltas, probe):
        """Sends the text the rows have formed since the last time, when asked for, and the outcome of the rows that stopped."""
        if deltas:
            for row in [*running, *stopped]:
                text = self.text_of(row.tokens)
                if not text.endswith("\ufffd") and len(text) > len(row.sent):
                    send({"id": identifier, "key": row.key, "delta": text[len(row.sent) :]})
                    row.sent = text
        for row in stopped:
            outcome = {
                "id": identifier, "key": row.key, "done": True, "text": self.text_of(row.tokens), "stop": row.stop,
                "prompt_tokens": len(row.ids), "tokens": len(row.tokens), "seconds": round(row.seconds, 3),
            }
            if probe:
                outcome["token_ids"], outcome["first_top"] = row.tokens, row.first_top
            send(outcome)
            row.reported = True

    @torch.inference_mode()
    def generate(self, identifier, request):
        """Runs the requests in a pool of up to `batch_size` rows, the longest prompts first.

        A join takes about as long whatever the number of rows it adds, and a free place costs one row of a step.
        Waiting rows therefore join together, once the free places have added up to a step's worth of rows.
        """
        self.start = (None, None, None)  # Its memory is the generations' now.
        waiting = [Row(item, self.encode(item), self.matcher(item.get("pattern"))) for item in request["requests"]]
        waiting.sort(key=lambda row: -len(row.ids))
        size = max(int(request.get("batch_size", 1)), 1)
        bound = int(request["state_mib"]) << 20 if request.get("state_mib") else STATE_BUDGET
        deltas, probe = request.get("deltas", False), request.get("probe", False)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        pool, idle, failed = Pool(self), 0, False
        while waiting or pool.rows:
            if failed and self.device.type == "cuda":  # What the failed pool held is free only now that its error is gone.
                torch.cuda.empty_cache()
            joining, failed = [], False
            try:
                free = size - len(pool.rows)
                if waiting and free > 0 and (not pool.rows or idle >= size):
                    count, alone, budget = self.admit(pool, waiting[:free], bound)
                    if count:
                        joining, waiting = waiting[:count], waiting[count:]
                        stopped = pool.join(joining, probe)
                        self.emit(identifier, [row for row in joining if row.stop is None], stopped, deltas, probe)
                    elif not pool.rows:
                        row, waiting = waiting[0], waiting[1:]
                        message = f"the request needs {alone >> 20} MiB of state, {(alone - budget) >> 20} MiB more than is free"
                        send({"id": identifier, "key": row.key, "done": True, "error": message})
                        continue
                    idle = 0
                if pool.rows:
                    if waiting:
                        idle += size - len(pool.rows)
                    stopped = pool.step()
                    self.emit(identifier, pool.rows, stopped, deltas, probe)
            except Exception as error:  # A failure must not end the requests that did not take part in it.
                caught = [row for row in {id(row): row for row in [*pool.rows, *joining]}.values() if not row.reported]
                send({"id": identifier, "pool": pool.report()})
                pool, idle, failed = Pool(self), 0, True
                if isinstance(error, torch.OutOfMemoryError) and len(caught) > 1:
                    # The rows did not fit after all: they start again in a smaller pool.
                    for row in caught:
                        row.restart()
                    waiting = sorted(caught + waiting, key=lambda row: -len(row.ids))
                    size = max(len(caught) // 2, 1)
                else:
                    for row in caught:
                        send({"id": identifier, "key": row.key, "done": True, "error": f"{type(error).__name__}: {error}"})
        send({"id": identifier, "pool": pool.report()})
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {"finished": True}


def main():
    model_path, head_path, device = sys.argv[1], sys.argv[2], sys.argv[3]
    kernels, record, every = [*sys.argv[4:7], "", "", ""][:3]
    backbone = Backbone(model_path, head_path, device, kernels, record, int(every or 0))
    send(backbone.ready())
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            identifier = request.get("id")
            try:
                if request["op"] == "decide":
                    reply = backbone.decide(request)
                elif request["op"] == "generate":
                    reply = backbone.generate(identifier, request)
                elif request["op"] == "dev":
                    reply = {"summary": recorder.finish()}
                else:
                    raise ValueError(f"unknown operation {request['op']!r}")
            except Exception as error:  # A failed request must not end the worker.
                reply = {"error": f"{type(error).__name__}: {error}"}
            send({"id": identifier, **reply})
    finally:
        recorder.finish()  # A record file that was not asked for still ends with its summary.


if __name__ == "__main__":
    main()
