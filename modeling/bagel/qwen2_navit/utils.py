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

import torch


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)
