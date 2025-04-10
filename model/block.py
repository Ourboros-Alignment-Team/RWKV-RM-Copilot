import os
import torch.nn as nn
from .ffn import RWKV_CMix_x070
from .att import RWKV_Tmix_x070


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first, attention_mask=None):
        if self.layer_id == 0:
            x = self.ln0(x)

        
        x_attn, v_first = self.att(
            self.ln1(x),
            v_first,
            attention_mask=attention_mask,
        )
        x = x + x_attn

        ffn_out = self.ffn(
            self.ln2(x), attention_mask=attention_mask
        )

        x = x + ffn_out
        return x, v_first
