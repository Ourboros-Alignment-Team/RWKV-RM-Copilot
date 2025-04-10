import os
import torch
import torch.nn as nn
from .state import BlockState, BlockStateList
from .rwkvLinear import make_linear_ffn



class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = make_linear_ffn(args.n_embd, args.n_embd * 4, bias=False)
        self.value = make_linear_ffn(args.n_embd * 4, args.n_embd, bias=False)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2] :, None])
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
