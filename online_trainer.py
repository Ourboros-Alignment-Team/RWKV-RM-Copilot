import os
import json
import torch
import torch.nn.functional as F
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from config import load_config, rm_prefix, rm_postfix

config = load_config(
    "/home/ubuntu/MachineLr/RWKV-RM-Copilot/configs/server_config.json",
)
os.environ["RWKV_REWARD_MODEL_HEAD_SIZE_A"] = str(config.model.head_size)
os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
CHUNK_LEN = config.model.CHUNK_LEN
os.environ["CHUNK_LEN"] = str(CHUNK_LEN)
from model.model import RWKVRewardModel
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
from typing import List
import gc


class RMOnlineTrainer:
    def __init__(self):
        self.config = config
        self.train_tokenizer = TRIE_TOKENIZER(config.train_tokenizer)
        self.infer_tokenizer = TRIE_TOKENIZER(config.infer_tokenizer)
        self.model = RWKVRewardModel(config)
        self.model.train()
        self.model_engine, self.optimizer = self.build_engine(self.model, config)
        self.model_engine.train()

    def build_engine(self, model, args):

        ds_config = {
            "bfloat16": {"enabled": "auto"},
            "gradient_accumulation_steps": args.deepspeed.gradient_accumulation_steps,
            "gradient_clipping": args.train.grad_cp,
            "train_micro_batch_size_per_gpu": 1,
        }
        if args.deepspeed.zero:
            ds_config["zero_optimization"] = {
                "stage": args.deepspeed.ds_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e6,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e6,
                "contiguous_gradients": True,
            }

            if args.deepspeed.offload_optimizer:
                ds_config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
            if args.deepspeed.offload_param_stage3 and args.deepspeed.ds_stage == 3:
                ds_config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

            optimizer = (
                DeepSpeedCPUAdam(
                    model.get_optim_groups(),
                    lr=args.train.lr_init,
                    betas=(args.train.beta1, args.train.beta2),
                    eps=args.train.adam_eps,
                    adamw_mode=args.train.adamw_mode,
                    weight_decay=args.train.weight_decay,
                    amsgrad=False,
                    bias_correction=True,
                )
                if args.deepspeed.zero and args.deepspeed.offload_optimizer
                else FusedAdam(
                    model.get_optim_groups(),
                    lr=args.train.lr_init,
                    betas=(args.train.beta1, args.train.beta2),
                    eps=args.train.adam_eps,
                    bias_correction=True,
                    adam_w_mode=args.train.adamw_mode,
                    weight_decay=args.train.weight_decay,
                    amsgrad=False,
                )
            )

            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=args.train.lr_init,
                warmup_max_lr=args.train.lr_final,
                warmup_num_steps=args.train.warmup_steps,
                warmup_type="linear",
            )

            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=ds_config,
            )
            print("cuda available", torch.cuda.device_count())
        return model_engine, optimizer

    def save_history(history_dict_list, to_jsonl_dir):
        with open(to_jsonl_dir, "a", encoding="utf-8") as f:
            for history_dict in history_dict_list:
                f.write(json.dumps(history_dict, ensure_ascii=False) + "\n")

    def pad_to_chunk_length(self, token_list, chunk_len):
        if not token_list:
            return []

        max_length = max(len(tokens) for tokens in token_list)
        padded_list = []
        for tokens in token_list:
            padding_length = max_length - len(tokens)
            padded_tokens = [0] * padding_length + tokens
            padded_list.append(padded_tokens)

        new_max_length = max(len(tokens) for tokens in padded_list)

        remainder = new_max_length % chunk_len
        if remainder != 0:
            additional_padding = chunk_len - remainder
            padded_list = [[0] * additional_padding + tokens for tokens in padded_list]

        return padded_list

    def inference(self, history, response) -> float:
        self.model_engine.eval()
        with torch.no_grad():
            input_tokens = [
                self.train_tokenizer.encode(history)
                + rm_prefix
                + self.train_tokenizer.encode(response)
                + rm_postfix
            ]
            input_tokens = self.pad_to_chunk_length(input_tokens, CHUNK_LEN)
            score = self.model_engine(input_tokens).squeeze().item() * 2 - 1
        self.model_engine.train()
        return score

    def train(
        self,
        data_list: List[str],
        batch_size: int = 1,
        save_ckpt: bool = False,
        save_ckpt_dir: str = None,
    ):
        self.model_engine.train()
        input_tokens = [
            self.train_tokenizer.encode(history)
            + rm_prefix
            + self.train_tokenizer.encode(response)
            + rm_postfix
            for history, response, _ in data_list
        ]
        score_list = [score for _, _, score in data_list]

        for i in range(0, len(input_tokens), batch_size):
            batch_input_tokens = input_tokens[i : i + batch_size]
            batch_score_list = score_list[i : i + batch_size]
            batch_input_tokens = self.pad_to_chunk_length(batch_input_tokens, CHUNK_LEN)
            batch_score_list = (
                torch.tensor(batch_score_list)
                .to(
                    self.model_engine.device,
                    dtype=self.model_engine.dtype,
                )
                .unsqueeze(-1)
            )
            predict_scores = self.model_engine(batch_input_tokens) * 2 - 1
            assert (
                predict_scores.shape == batch_score_list.shape
            ), f"{predict_scores.shape}!= {batch_score_list.shape}"
            loss = F.l1_loss(predict_scores, batch_score_list)
            self.model_engine.backward(loss)
            self.model_engine.step()
            gc.collect()
            torch.cuda.empty_cache()
            print("loss", loss.item())

        if save_ckpt:
            self.save_checkpoint(save_ckpt_dir)
            print(f"save checkpoint at {save_ckpt_dir}")

    def save_checkpoint(self, save_ckpt_dir):
        self.model.load_state_dict(self.model_engine.module.state_dict())
        torch.save(self.model.state_dict(), save_ckpt_dir)
        gc.collect()
        torch.cuda.empty_cache()
