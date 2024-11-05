import torch
from torch import nn
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig

from .xlstm import xLSTMLMModel, xLSTMLMModelConfig

from typing import List

from transformers import PretrainedConfig
from collections import namedtuple


class xLSTMConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality and RC equivariance."""

    model_type = "xlstm"

    def __init__(
        self,
        d_model: int = 256,
        n_layer: int = 4,
        tie_weights: bool = False,
        max_length: int = 1024,
        vocab_size: int = 16,
        pad_vocab_size_multiple: int = 8,
        m_conv1d_kernel_size: int = 4,
        m_conv1d_causal: bool = True,
        m_qkv_proj_blocksize: int = 4,
        m_num_heads: int = 4,
        m_proj_factor: float = 2.0,
        m_backend: str = "parallel",
        m_chunk_size: int = 64,
        m_position_embeddings: bool = False,
        m_bias: bool = False,
        s_num_heads: int = 4,
        s_conv1d_kernel_size: int = 4,
        s_conv1d_causal: bool = True,
        s_lstm_at: List = [],
        s_proj_factor: float = 1.3,
        s_round_proj_up_dim_up: bool = True,
        s_round_proj_up_to_multiple_of: int = 64,
        s_position_embeddings: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        bidirectional_alternating: bool = False,
        m_backend_bidirectional: bool = False,
        rcps: bool = False,
        complement_map: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.tie_weights = tie_weights

        self.max_length = max_length
        self.vocab_size = vocab_size

        self.pad_vocab_size_multiple = pad_vocab_size_multiple

        self.rcps = rcps

        # parse arguments into nested dictionary
        if len(s_lstm_at) > 0:
            slstm_config = {
                "slstm": {
                    "backend": "cuda",
                    "num_heads": s_num_heads,
                    "conv1d_kernel_size": s_conv1d_kernel_size,
                    "conv1d_causal": s_conv1d_causal,
                    "bias_init": "powerlaw_blockdependent",
                },
                "feedforward": {
                    "proj_factor": s_proj_factor,
                    "round_proj_up_dim_up": s_round_proj_up_dim_up,
                    "round_proj_up_to_multiple_of": s_round_proj_up_to_multiple_of,
                },
                "rcps": rcps
            }
        else:
            slstm_config = {}

        if len(s_lstm_at) == n_layer:
            mlstm_cfg = {}
        else:
            mlstm_cfg = {
                "mlstm": {
                    "conv1d_kernel_size": m_conv1d_kernel_size,
                    "conv1d_causal": m_conv1d_causal,
                    "qkv_proj_blocksize": m_qkv_proj_blocksize,
                    "num_heads": m_num_heads,
                    "proj_factor": m_proj_factor,
                    "round_proj_up_to_multiple_of": 64,
                    "round_proj_up_dim_up": False,
                    "backend": m_backend,
                    "chunk_size": m_chunk_size,
                },
                "rcps": rcps,
            }

        xlstm_cfg = {
            "vocab_size": vocab_size,
            "mlstm_block": mlstm_cfg,
            "slstm_block": slstm_config,
            "bidirectional": bidirectional,
            "bidirectional_alternating": bidirectional_alternating,
            "m_backend_bidirectional": m_backend_bidirectional,
            "m_position_embeddings": m_position_embeddings,
            "s_position_embeddings": s_position_embeddings,
            "context_length": self.max_length,
            "num_blocks": self.n_layer,
            "dropout": dropout,
            "embedding_dim": self.d_model,
            "complement_map": complement_map if complement_map is not None else {},
            "rcps": rcps,
            "bias": m_bias,
            "slstm_at": s_lstm_at
        }

        xlstm_cfg = OmegaConf.create(xlstm_cfg)
        xlstm_cfg = from_dict(
            data_class=xLSTMLMModelConfig,
            data=OmegaConf.to_container(xlstm_cfg),
            config=DaciteConfig(strict=True),
        )
        self.xlstm_cfg = xlstm_cfg


class xLSTMLMHeadModel(nn.Module):

    def __init__(
        self,
        config: xLSTMConfig,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.config = config
        vocab_size = config.vocab_size
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )

        xlstm_cfg = config.xlstm_cfg

        self.backbone = xLSTMLMModel(xlstm_cfg).to(device)

        self.backbone.reset_parameters()

    def forward(
        self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """

        lm_logits = self.backbone(input_ids)

        LMOutput = namedtuple("LMOutput", ["logits"])
        return LMOutput(logits=lm_logits)
    
    def headless_forward(self, position_ids=None, inference_params=None, num_last_tokens=0):
        hiddens = self.backbone.headless_forward(position_ids)
        return hiddens




