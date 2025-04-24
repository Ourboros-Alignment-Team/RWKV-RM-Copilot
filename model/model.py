########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from .block import Block
import types
import gc
from typing import Union, Optional, List


class RWKVRewardModel(nn.Module):
    def __init__(self, args_in):
        super().__init__()
        args = types.SimpleNamespace(**vars(args_in.model))
        args.train = args_in.train
        args.model = args_in.model
        args.grad_cp = args_in.train.grad_cp

        # args.voice_on = voice_on
        # if voice_on:
        #     args.vocoder = args_in.vocoder
        self.args = args
        if self.args.model.dtype == "fp32":
            self.args.model.dtype = torch.float
        elif self.args.model.dtype == "fp16":
            self.args.model.dtype = torch.half
        elif self.args.model.dtype == "bf16":
            self.args.model.dtype = torch.bfloat16
        if args.load_model is not None:
            model_weights = torch.load(args.load_model, map_location="cpu")
        else:
            model_weights = self.state_dict()
        model_keys = list(model_weights.keys())
        if args.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if "blocks." in x:
                    block_id = int(x.split(".")[1])
                    max_block_id = max(max_block_id, block_id)
            args.n_layer = max_block_id + 1

        if args.n_embd < 0:
            if "head.weight" in model_keys:
                args.n_embd = model_weights["head.weight"].shape[1]
            elif "emb.weight" in model_keys:
                args.n_embd = model_weights["emb.weight"].shape[1]
            else:
                raise ValueError(
                    "无法确定模型的嵌入大小，请确定读取的模型是否为rwkv 或rwkv reward model。"
                )
        print("embd size:", args.n_embd)
        if args.vocab_size < 0:
            if "head.weight" in model_keys:
                args.vocab_size = model_weights["head.weight"].shape[0]
            elif "emb.weight" in model_keys:
                args.vocab_size = model_weights["emb.weight"].shape[0]
            else:
                raise ValueError(
                    "无法确定模型的嵌入大小，请确定读取的模型是否为rwkv 或rwkv reward model。"
                )

        args.dim_att = args.n_embd
        args.n_head = args.dim_att // args.head_size
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        # self.out_weights_layer = nn.Linear(args.n_embd, args.n_embd)
        self.out_score_layer = nn.Linear(args.n_embd, 1)

        # # 初始化out_weights_layer的权重
        # self.out_weights_layer.weight.data.uniform_(-0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5))
        # self.out_weights_layer.bias.data.zero_()
        # 初始化out_score_layer的权重
        self.out_score_layer.weight.data.uniform_(-0.01, 0.01)
        self.out_score_layer.bias.data.uniform_(-0.01, 0.01)
        
        self.load_state_dict(model_weights, strict=False)
        del model_weights

        for p in self.parameters():
            p.data = p.data.to(dtype=self.args.model.dtype)

        gc.collect()
        torch.cuda.empty_cache()

    def get_optim_groups(self):
        args = self.args
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        lr_10x = set()
        all_params = set()

        for n, p in self.named_parameters():
            if (("_w1" in n) or ("_w2" in n)) and (args.train.layerwise_lr > 0):
                if n not in all_params:
                    lr_1x.add(n)
                    all_params.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (
                args.train.layerwise_lr > 0
            ):
                if args.train.my_pile_stage == 2:
                    if n not in all_params:
                        lr_2x.add(n)
                        all_params.add(n)
                else:
                    if n not in all_params:
                        lr_1x.add(n)
                        all_params.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (
                args.train.layerwise_lr > 0
            ):
                if args.train.my_pile_stage == 2:
                    if n not in all_params:
                        lr_3x.add(n)
                        all_params.add(n)
                else:
                    if n not in all_params:
                        lr_2x.add(n)
                        all_params.add(n)
            elif ("time_faaaa" in n) and (args.train.layerwise_lr > 0):
                if args.train.my_pile_stage == 2:
                    if n not in all_params:
                        lr_2x.add(n)
                        all_params.add(n)
                else:
                    if n not in all_params:
                        lr_1x.add(n)
                        all_params.add(n)
            elif ("time_first" in n) and (args.train.layerwise_lr > 0):
                if n not in all_params:
                    lr_3x.add(n)
                    all_params.add(n)
            elif "track_mixing" in n:
                lr_1x.add(n)
            elif "adapter_e" in n or "vocoder_d" in n:
                if ("vocos_backbone" in n) or ("head" in n):
                    lr_10x.add(n)
                else:
                    lr_1x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.train.weight_decay > 0):
                if n not in all_params:
                    lr_decay.add(n)
                    all_params.add(n)
            else:
                if n not in all_params:
                    lr_1x.add(n)
                    all_params.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        lr_10x = sorted(list(lr_10x))
        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[n] for n in lr_1x],
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,
            },
            {
                "params": [param_dict[n] for n in lr_2x],
                "weight_decay": 0.0,
                "my_lr_scale": 2.0,
            },
            {
                "params": [param_dict[n] for n in lr_3x],
                "weight_decay": 0.0,
                "my_lr_scale": 3.0,
            },
            {
                "params": [param_dict[n] for n in lr_10x],
                "weight_decay": 0.0,
                "my_lr_scale": 10.0,
            },
        ]
        print(
            f"param 1x {len(lr_1x)} 2x {len(lr_2x)} 3x {len(lr_3x)} 10x {len(lr_10x)} decay {len(lr_decay)}"
        )

        if args.train.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": args.train.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
        return optim_groups

    def forward_from_embeddings(
        self,
        embeddings,
        attention_mask=None,
    ):
        """
        embeddings : (b, N, n_embd)
        output :  (b, N, n_embd)
        """
        args = self.args
        B, T, n_embd = embeddings.size()
        C = args.n_embd
        H = args.dim_att // args.head_size
        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert n_embd == C

        x = embeddings
        v_first = torch.empty_like(x)
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            if int(args.grad_cp) == 1 and self.training:
                import deepspeed
                x, v_first = deepspeed.checkpointing.checkpoint(
                    block, x, v_first, attention_mask
                )
            else:
                x, v_first = block(x, v_first, attention_mask)

        x = self.ln_out(x)
        # weights = self.out_weights_layer(x[:,-1,:])
        out_score = self.out_score_layer(x[:,-1,:])
        normalized_score = F.sigmoid(out_score) # -1 1

        return normalized_score

    def forward(
        self,
        idx: Union[torch.Tensor, list],
        attention_mask=None,
    ):
        args = self.args
        idx = torch.tensor(idx, device=next(self.parameters()).device, dtype=torch.long)
        B, T = idx.size()
        # assert T <= args.chunk_ctx, "Cannot forward, model ctx_len is exhausted."
        C = args.n_embd
        H = args.dim_att // args.head_size_a
        assert C == H * args.head_size_a

        x = self.emb(idx)
        v_first = torch.empty_like(x)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            if int(args.grad_cp) == 1 and self.training:
                import deepspeed
                x, v_first = deepspeed.checkpointing.checkpoint(
                    block, x, v_first, attention_mask
                )
            else:
                x, v_first = block(x, v_first, attention_mask)

        x = self.ln_out(x)
        # weights = self.out_weights_layer(x[:,-1,:])
        out_score = self.out_score_layer(x[:,-1,:])
        normalized_score = F.sigmoid(out_score)

        return normalized_score

    def embedding(self, idx):
        args = self.args
        idx = idx.to(next(self.parameters()).device, dtype=torch.long)

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size
        assert C == H * args.head_size
        x = self.emb(idx)
        return x
