# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""Qwen2 with packed (variable-length) attention: split into config, cache, utils, attention, decoder layers, modeling."""

from .attention import PackedAttention, PackedAttentionMoT
from .cache import BaseNavitOutputWithPast, NaiveCache
from .config import Qwen2Config
from .decoder_layers import (
    Decoder_layer_dict,
    Qwen2DecoderLayer,
    Qwen2MoEDecoderLayer,
    Qwen2MoTDecoderLayer,
)
from .modeling import Qwen2ForCausalLM, Qwen2Model
from .utils import pad_sequence

__all__ = [
    "BaseNavitOutputWithPast",
    "Decoder_layer_dict",
    "NaiveCache",
    "PackedAttention",
    "PackedAttentionMoT",
    "Qwen2Config",
    "Qwen2DecoderLayer",
    "Qwen2ForCausalLM",
    "Qwen2MoEDecoderLayer",
    "Qwen2MoTDecoderLayer",
    "Qwen2Model",
    "pad_sequence",
]
