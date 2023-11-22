# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
import torch
from torch import nn
from torch.nn import functional as F
from tome.clip.clip_vision import ResidualAttentionBlock, CLIP_VIT, VisionTransformer
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r
from typing import Tuple
from torch.jit import Final
from timm.layers import use_fused_attn


class ToMeBlock(ResidualAttentionBlock):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def attention(self, x: torch.Tensor, attn_size: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        attn_mask = None

        if attn_size is not None:
            attn_mask = attn_size.log()[:, None, None, :, 0]

        if self.attn_mask is not None:
            attn_mask = attn_size + self.attn_mask

        return self.attn(x, attn_mask=attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None

        x_attn, metric = self.attention(self.ln_1(x), attn_size)
        x = x + x_attn

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self.mlp(self.ln_2(x))
        return x


class ToMeAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(1)


class ToMeVisionTransformer(VisionTransformer):
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

def convert_attention_block(
    src: nn.MultiheadAttention, dst: ToMeAttention
) -> Tuple[ToMeAttention, torch.device]:
    src_state_dict = src.state_dict()
    dst_state_dict = dst.state_dict()
    src_to_dst_keys = [
        ("in_proj_weight", "qkv.weight"),
        ("in_proj_bias", "qkv.bias"),
        ("out_proj.weight", "proj.weight"),
        ("out_proj.bias", "proj.bias"),
    ]

    for src_key, dst_key in src_to_dst_keys:
        dst_state_dict[dst_key] = src_state_dict[src_key]
    dst.load_state_dict(dst_state_dict)
    src_device = src_state_dict["in_proj_weight"].device
    return dst.to(src_device), src_device

def make_tome_class(transformer_class):
    class ToMeClipVIT(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        def encode_image(self, *args, **kwdargs) -> torch.Tensor:
            if isinstance(self.r, int):
                self._tome_info["r"] = parse_r(len(self.visual.transformer.resblocks), self.r)
            else:
                assert len(self.visual.transformer.resblocks) == len(self.r)
                self._tome_info["r"] = self.r
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().encode_image(*args, **kwdargs)

    return ToMeClipVIT


def apply_patch(
    model: CLIP_VIT, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeClipVIT = make_tome_class(model.__class__)

    model.__class__ = ToMeClipVIT
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info

            attn = ToMeAttention(module.attn.embed_dim, module.attn.num_heads, qkv_bias=True)
            _, device = convert_attention_block(module.attn, attn)
            module.attn = attn.to(device)

        elif isinstance(module, VisionTransformer):
            module.__class__ = ToMeVisionTransformer

    pass
