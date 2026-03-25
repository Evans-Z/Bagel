# Copyright (c) 2022 Facebook, Inc. and its affiliates.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: CC BY-NC 4.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under CC BY-NC 4.0, with the full license text
# available at https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

import math

import numpy as np
import torch
from torch import nn
from transformers.activations import ACT2FN

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# TimestepEmbedder
# Reference:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    For BAGEL-7B-MoT (hidden_size=3584):
        timestep_embedding: scalar t → sinusoidal [256]
        mlp: Linear(256, 3584) → SiLU → Linear(3584, 3584)
        Output: [N, 3584] — added to each latent token's embedding
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),  # (256, 3584)
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),              # (3584, 3584)
        )
        self.frequency_embedding_size = frequency_embedding_size  # 256

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLPconnector(nn.Module):
    """
    Projects ViT features into LLM hidden space.

    For BAGEL-7B-MoT (vit_hidden=1152, llm_hidden=3584, act=gelu_pytorch_tanh):
        fc1: Linear(1152, 3584) → gelu → fc2: Linear(3584, 3584)
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)    # (1152, 3584)
        self.fc2 = nn.Linear(out_dim, out_dim)   # (3584, 3584)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class PositionEmbedding(nn.Module):
    """
    Frozen 2D sincos position embeddings for image patches.

    For BAGEL-7B-MoT:
        latent_pos_embed: PositionEmbedding(32, 3584) → 32²=1024 positions (or 64²=4096 at inference)
        vit_pos_embed:    PositionEmbedding(70, 3584) → 70²=4900 positions
    pos_embed: [max_patches², hidden_size], frozen (requires_grad=False)
    """
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side ** 2, hidden_size), 
            requires_grad=False
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids):
        return self.pos_embed[position_ids]