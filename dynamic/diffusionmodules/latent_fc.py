import math
from typing import Iterable

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from typing import NamedTuple
from enum import Enum
from torch import nn
from torch.nn import init


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(-math.log(max_period) *
                   th.arange(start=0, end=half, dtype=th.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat(
            [embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding




class LatentFC(nn.Module):
    """
    concat x to hidden layers
    default MLP for the latent DPM in the paper!
    """
    def __init__(self,
                 num_layers=10,
                 num_time_layers=2,
                 num_channels=2048,
                 num_time_emb_channels=64,
                 model_channels=512,
                 condition_bias=1,
                 time_last_act=False,
                 dropout=0,
                 use_norm=True,
                 activation=Activation.silu,
                 last_act=Activation.none):
        super().__init__()

        self.skip_layers = list(range(1, num_layers))
        ##############  time_embed
        self.num_time_emb_channels = num_time_emb_channels
        layers = []
        for i in range(num_time_layers):
            if i == 0:
                a = num_time_emb_channels
                b = num_channels
            else:
                a = num_channels
                b = num_channels
            layers.append(nn.Linear(a, b))
            if i < num_time_layers - 1 or time_last_act:
                layers.append(activation.get_act())
        self.time_embed = nn.Sequential(*layers)
        ##############  time_embed

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = activation
                norm = use_norm
                cond = True
                a, b = num_channels, model_channels
                dropout = dropout
            elif i == num_layers - 1:
                act = Activation.none
                norm = False
                cond = False
                a, b = model_channels, num_channels
                dropout = 0
            else:
                act = activation
                norm = use_norm
                cond = True
                a, b = model_channels, model_channels
                dropout = dropout

            if i in self.skip_layers:
                a += num_channels

            logger.warning(f'cond = {cond}, in_channels = {a}, cond_channels = {num_channels}, out_channels = {b}')
            self.layers.append(
                MLPLNAct(
                    in_channels=a,
                    out_channels=b,
                    norm=norm,
                    activation=act,
                    cond_channels=num_channels,
                    use_cond=cond,
                    condition_bias=condition_bias,
                    dropout=dropout,
                ))
        self.last_act = last_act.get_act()

    def forward(self, x, t):
        t = timestep_embedding(t, self.num_time_emb_channels)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return h


class MLPLNAct(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm: bool,
            use_cond: bool,
            activation: Activation,
            cond_channels: int,
            condition_bias: float = 0,
            dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()


    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x



if __name__=='__main__':
    a = LatentFC(num_channels=2048)
    i = torch.randn(4, 2048)
    t = torch.randn(4)
    o = a(i,t)
    print(o.shape)