# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DRY (Don't Repeat Yourself) repetition penalty of one generation.

Adapted from vLLM's DRY example, examples/features/logits_processor/dry.py, licensed under the Apache License 2.0:
`max_exponent` and `_dry_penalties`, which port the scan in llama.cpp's `llama_sampler_dry_apply`. Changes: the
penalties are computed for one row of a generation pool from the tokens it has produced, with no vLLM types;
`max_exponent` rounds through float32 with the standard library instead of NumPy; and the scan returns at once for a
window no longer than the allowed length, which can hold no repeat of that length.
"""

import math
import struct

# llama.cpp's FLOAT_MAX_LOG: ln(float32 max).
_FLOAT_MAX_LOG = 88.7228391


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def max_exponent(base):
    """The exponent's bound, as in llama.cpp: FLOAT_MAX_LOG / ln(base) in float32, truncated. 0 for a base too close
    to 1 to bound."""
    if base <= 1.000001:
        return 0
    return int(_float32(_float32(_FLOAT_MAX_LOG) / _float32(math.log(_float32(base)))))


def dry_penalties(window, breakers, multiplier, base, allowed_length, max_exp):
    """The penalty to subtract from the logit of each token that would extend a repeat of at least `allowed_length`
    tokens within the window, by token: `multiplier * base ** (repeat length - allowed_length)`, the exponent bounded
    by `max_exp` when it is not 0.

    A Z-algorithm run from the end finds, for each position, how long the tokens ending there match the end of the
    window; the token after such a match is charged for the longest repeat it would extend. A repeat does not run
    across a token in `breakers`, and those tokens are never charged.
    """
    size = len(window)
    if size <= allowed_length:
        return {}

    def back(i):  # back(0) is the last token
        return window[size - 1 - i]

    # The nearest breaker from the end bounds the repeat.
    limit = size
    for i in range(size):
        if back(i) in breakers:
            limit = i
            break
    if limit < allowed_length:
        return {}

    matched = [0] * size
    last = size - 1
    left = right = 0
    for k in range(1, size):
        if k > right:
            # Outside the current Z-box: extend naively.
            z = 0
            while z + k < size and back(z) == back(z + k):
                z += 1
            matched[last - k] = min(z, limit)
            if z > 0:
                left, right = k, k + z - 1
        else:
            inside = right - k + 1
            if matched[last - (k - left)] < inside:
                # Fully inside the Z-box: copy.
                matched[last - k] = min(matched[last - (k - left)], limit)
            else:
                # Touches the right edge: extend past it.
                j = right + 1
                while j < size and back(j) == back(j - k):
                    j += 1
                matched[last - k] = min(j - k, limit)
                left, right = k, j - 1

    longest = {}
    for i in range(size - 1):
        if matched[i] >= allowed_length and longest.get(window[i + 1], -1) < matched[i]:
            longest[window[i + 1]] = matched[i]

    penalties = {}
    for token, length in longest.items():
        if token in breakers:
            continue
        exponent = length - allowed_length
        if max_exp and exponent > max_exp:
            exponent = max_exp
        penalties[token] = multiplier * base**exponent
    return penalties
