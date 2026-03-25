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

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.utils import ModelOutput


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None
