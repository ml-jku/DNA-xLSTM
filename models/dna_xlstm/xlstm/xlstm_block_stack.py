# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck

# Modified by Pieter-Jan Hoedt, Niklas Schmidinger, Lisa Schneckenreiter and Sohvi Luukkonen 

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
from .components.ln import LayerNorm
from .modeling_xlstm_rcps import RCPSWrapper

@dataclass
class xLSTMBlockStackConfig:
    mlstm_block: Optional[mLSTMBlockConfig] = None
    slstm_block: Optional[sLSTMBlockConfig] = None

    context_length: int = -1
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.0

    # bidirectional sequence processing
    bidirectional: bool = False
    bidirectional_alternating: bool = False
    m_backend_bidirectional: bool = False 

    # position embeddings
    m_position_embeddings: bool = False 
    s_position_embeddings: bool = True

    # The block indices at which sLSTM blocks are placed.
    # Indexing starts from 0.
    slstm_at: Union[list[int], Literal["all"]] = field(default_factory=list)

    # _block_map is a string that specifies which block is used at which position
    # 0: use the mLSTM block
    # 1: use the sLSTM block
    _block_map: str = None
    
    # rcps mode
    complement_map: dict = None
    rcps: bool = False

    @property
    def block_map(self) -> list[int]:
        return list(map(int, self._block_map.split(",")))

    def _create_block_map(self) -> str:
        """Creates the block map, that specifies which block is used at which position."""
        block_map = [0] * self.num_blocks

        for slstm_position_idx in self.slstm_at:
            assert (
                slstm_position_idx < self.num_blocks
            ), f"Invalid slstm position {slstm_position_idx}"
            block_map[slstm_position_idx] = 1

        block_map_str = ",".join(map(str, block_map))

        return block_map_str

    def __post_init__(self):
        if self.mlstm_block is None:
            self.slstm_at = "all"
        if self.slstm_at == "all":
            self.slstm_at = list(range(self.num_blocks))

        if self.mlstm_block is not None:
            self.mlstm_block.mlstm.embedding_dim = self.embedding_dim
            self.mlstm_block.mlstm.bias = self.bias
            self.mlstm_block.mlstm.dropout = self.dropout
            self.mlstm_block.mlstm.context_length = self.context_length
            self.mlstm_block.mlstm._num_blocks = self.num_blocks
            # call post init, for setting inner_embedding_dim
            self.mlstm_block.__post_init__()

        if self.slstm_block is not None:
            self.slstm_block.slstm.dropout = self.dropout
            self.slstm_block.slstm.embedding_dim = self.embedding_dim
            self.slstm_block._num_blocks = self.num_blocks
            self.slstm_block.__post_init__()

        self._block_map = self._create_block_map()


# Absolute Position Embeddings for sLSTM
def exists(val):
    return val is not None

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = torch.einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale       


# Rotary position embeddings for mLSTM
def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def get_mlstm_inner_dim(m_config) -> int:
    proj_up_dim = m_config.proj_factor * m_config.embedding_dim
    multiple_of_multiplier = proj_up_dim / m_config.round_proj_up_to_multiple_of
    if m_config.round_proj_up_dim_up:
        multiple_of_multiplier = math.ceil(multiple_of_multiplier)
    else:
        multiple_of_multiplier = math.floor(multiple_of_multiplier)
    return int(multiple_of_multiplier * m_config.round_proj_up_to_multiple_of)


class xLSTMBlockStack(nn.Module):
    config_class = xLSTMBlockStackConfig

    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config

        self.blocks = self._create_blocks(config=config)
        if config.add_post_blocks_norm:
            self.post_blocks_norm = LayerNorm(ndim=config.embedding_dim, bias=False)
            if config.rcps:
                self.post_blocks_norm = RCPSWrapper(self.post_blocks_norm)
        else:
            self.post_blocks_norm = nn.Identity()

        m_config = config.mlstm_block.mlstm
        m_inner_dim = get_mlstm_inner_dim(m_config)
        freqs_cos, freqs_sin = precompute_freqs_cis(m_inner_dim // m_config.num_heads, config.context_length)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.absolute_pos_embeds = ScaledSinusoidalEmbedding(dim=config.embedding_dim, theta=100_000) if self.config.s_position_embeddings else None

    def _create_blocks(self, config: xLSTMBlockStackConfig):

        blocks = []
        for block_idx, block_type_int in enumerate(config.block_map):
            if block_type_int == 0:
                config = deepcopy(self.config.mlstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(mLSTMBlock(config=config))
            elif block_type_int == 1:
                config = deepcopy(self.config.slstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(sLSTMBlock(config=config))
            else:
                raise ValueError(f"Invalid block type {block_type_int}")

        return nn.ModuleList(blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.reset_parameters()
        if not isinstance(self.post_blocks_norm, nn.Identity):
            if self.config.rcps:
                self.post_blocks_norm.submodule.reset_parameters()
            else:
                self.post_blocks_norm.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        if self.config.s_position_embeddings:
            x = x + self.absolute_pos_embeds(x)

        for i, block in enumerate(self.blocks):

            # positional embeddings for mLSTM
            if isinstance(block, mLSTMBlock) and self.config.m_position_embeddings:
                kwargs["use_position_embeddings"] = self.config.m_position_embeddings
                kwargs["freqs_cos"] = self.freqs_cos
                kwargs["freqs_sin"] = self.freqs_sin

            if self.config.bidirectional:
                
                if self.config.bidirectional_alternating:
                    # blockwise-alternating bidirectional 
                    #if (i // 2) % 2 != 0: # every seconf block (for hybrid models)
                    if (i % 2) != 0:
                        #x = block(x, **kwargs)
                        x = checkpoint(block, x, **kwargs, use_reentrant=False)
                    else:
                        x = torch.flip(x, dims=[1])
                        #x = block(x, **kwargs)
                        x = checkpoint(block, x, **kwargs, use_reentrant=False)
                        x = torch.flip(x, dims=[1])
                else:
                    if isinstance(block, mLSTMBlock) and self.config.m_backend_bidirectional:
                        # enable mLSTM native bidirectionality
                        kwargs["bidirectional"] = True
                        #x = block(x, **kwargs)
                        x = checkpoint(block, x, **kwargs, use_reentrant=False)
                    else:
                        # blockwise bidirectional
                        #x_left = block(x, **kwargs)
                        x_left = checkpoint(block, x, **kwargs, use_reentrant=False)
                        x_flipped = torch.flip(x, dims=[1])
                        #x_right = block(x_flipped, **kwargs)
                        x_right = checkpoint(block, x_flipped, **kwargs, use_reentrant=False)
                        x = x_left + torch.flip(x_right, dims=[1])
            else:
                # unidirectional case
                #x = block(x, **kwargs)
                x = checkpoint(block, x, **kwargs, use_reentrant=False)

        x = self.post_blocks_norm(x)

        return x
