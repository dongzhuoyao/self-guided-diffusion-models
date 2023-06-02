from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from einops_exts import rearrange_many, repeat_many, check_shape
from torch import nn, einsum


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d



class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


def l2norm(t):
    return F.normalize(t, dim = -1)


class Attention_LR(nn.Module):
    def __init__(
        self,
        query_dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        cosine_sim_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(query_dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(query_dim, dim_head * 2, bias = False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias = False),
            LayerNorm(query_dim)
        )

    def forward(self, x, context = None, mask = None, attn_bias = None):
        device =  x.device
        b, _c, _w, _h = x.shape
        x = rearrange(x, 'b c w h-> b (w h) c')

        x_normed = self.norm(x)

        q, k, v = (self.to_q(x_normed), *self.to_kv(x_normed).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 d', b = b)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # cosine sim attention

        if self.cosine_sim_attn:
            raise NotImplementedError #make sure not use it, to slow
            q, k = map(l2norm, (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.cosine_sim_scale

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return rearrange((x + out),'b (w h) c-> b c w h', w=_w, h=_h)

class CrossAttention_LR(nn.Module):
    def __init__(
        self,
        query_dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False,
        dropout=0.0#always 0. following lucidrains
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, query_dim)

        self.norm = LayerNorm(query_dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias = False),
            LayerNorm(query_dim)
        )

    def forward(self, x, context, mask = None):
        device = x.device
        b, _c, _w, _h = x.shape
        x = rearrange(x, 'b c w h-> b (w h) c')
        x_normed = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x_normed), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 d', h = self.heads,  b = b)

        k = torch.cat((nk, k, q), dim = -2)#Imagen D.3.1
        v = torch.cat((nv, v, q), dim = -2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        return rearrange((x + out),'b (w h) c-> b c w h', w=_w, h=_h)
