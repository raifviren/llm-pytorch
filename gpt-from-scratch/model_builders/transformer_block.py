"""
Created at 21/12/25
@author: raif.viren@gmail.com
"""
from torch import nn

from .feed_forward_layer import FeedForward
from .layer_norm import LayerNorm
from .multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.mha = MultiHeadAttention(d_in=cfg["emb_dim"],
                                      d_out=cfg["emb_dim"],
                                      context_length=cfg["context_length"],
                                      num_heads=cfg["n_heads"],
                                      dropout=cfg["drop_rate"],
                                      qkv_bias=cfg["qkv_bias"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        x = self.mha(x)
        x = self.drop(x)

        x = x + shortcut
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)

        x = x + shortcut

        return x
