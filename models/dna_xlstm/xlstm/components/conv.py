# Copyright JKU Linz 2024
# Maximilian Beck
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    causal: bool = True
    conv1d_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


class CausalConv1d(nn.Module):
    config_class = CausalConv1dConfig
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, config: CausalConv1dConfig):
        super().__init__()
        self.config = config
        self.groups = self.config.feature_dim
        if self.config.channel_mixing:
            self.groups = 1
        if self.config.kernel_size == 0:
            self.conv = None  # Noop
        else:
            # padding of this size assures temporal causality.
            self.pad = self.config.kernel_size - 1 if config.causal else "same"
            self.conv = nn.Conv1d(
                in_channels=self.config.feature_dim,
                out_channels=self.config.feature_dim,
                kernel_size=self.config.kernel_size,
                padding=self.pad,
                groups=self.groups,
                bias=self.config.causal_conv_bias,
                **self.config.conv1d_kwargs,
            )
        # B, C, L
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        self.conv.reset_parameters()

    def _create_weight_decay_optim_groups(self) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        if self.config.kernel_size == 0:
            return (), ()
        else:
            weight_decay = (self.conv.weight,)
            no_weight_decay = ()
            if self.config.causal_conv_bias:
                no_weight_decay += (self.conv.bias,)
            return weight_decay, no_weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return x
        
        y = x.transpose(2, 1)  # Transpose to (B, F, T)
        y = self.conv(y)  # Apply convolution
        
        if self.config.causal:
            y = y[:, :, :-self.pad]  # Remove padding for causality
        y = y.transpose(2, 1)  # Transpose back to (B, T, F)
        
        return y
