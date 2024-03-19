import torch
import torch.nn as nn
from ldm.modules.attention import FeedForward, CrossAttention
from ldm.modules.diffusionmodules.util import checkpoint

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class PointTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class PointTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, opt=None):
        super().__init__()
        self.opt = opt
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.proj_in = nn.Linear(in_channels,
                                 inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [PointTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Linear(inner_dim,
                                              in_channels))
        self.free_distance = self.opt.free_distance

    def forward(self, x, context=None):
        x_in = x
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = self.proj_out(x)
        x = (2 * self.free_distance) * torch.sigmoid(x) - self.free_distance
        return x + x_in
