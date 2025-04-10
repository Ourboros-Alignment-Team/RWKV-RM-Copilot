import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from types import SimpleNamespace
import json
import gc
deepspeed.init_distributed()


def dict_to_namespace(d) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d


def load_config(path) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    config_data = json.loads(text)
    args = dict_to_namespace(config_data)
    return args


import os

config = load_config(
    "/home/ubuntu/MachineLr/RWKV-RM/configs/train_config.json",
)
os.environ["RWKV_REWARD_MODEL_HEAD_SIZE_A"] = str(
    config.deep_reward_model.model.head_size
)
os.environ["RWKV_CTXLEN"] = str(config.deep_reward_model.model.ctx_len)
os.environ["CHUNK_LEN"] = str(config.deep_reward_model.model.CHUNK_LEN)
from model.model import RWKVRewardModel
from dataset.dataset import RewardModelPairDataset


def build_engine(model, args):

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


# def pad_zeros_at_beginning(token_list):
#     if not token_list:
#         return []

#     max_length = max(len(tokens) for tokens in token_list)
#     padded_list = []

#     for tokens in token_list:
#         padding_length = max_length - len(tokens)
#         padded_tokens = [0] * padding_length + tokens
#         padded_list.append(padded_tokens)


#     return padded_list
def pad_to_chunk_length(token_list, chunk_len):
    if not token_list:
        return []

    # 先进行开头补零对齐
    max_length = max(len(tokens) for tokens in token_list)
    padded_list = []
    for tokens in token_list:
        padding_length = max_length - len(tokens)
        padded_tokens = [0] * padding_length + tokens
        padded_list.append(padded_tokens)

    # 检查是否需要进一步补零以满足 chunk_len 的倍数
    new_max_length = max(len(tokens) for tokens in padded_list)
    # print("max_length", new_max_length)

    remainder = new_max_length % chunk_len
    if remainder != 0:
        additional_padding = chunk_len - remainder
        padded_list = [[0] * additional_padding + tokens for tokens in padded_list]

    return padded_list


rm_prefix = [
    11,
    65520,
]
rm_postfix = [
    11,
    65521,
]
CHUNK_LEN = 16
batch_size = 6
if __name__ == "__main__":
    model = RWKVRewardModel(config.deep_reward_model)
    model_engine, optimizer = build_engine(
        model,
        config.deep_reward_model,
    )
    folders = [
        "/home/ubuntu/MachineLr/datadisk/rm_dataset/chinese_dpo_pair",
        "/home/ubuntu/MachineLr/datadisk/rm_dataset/rm_datasets_0",
        "/home/ubuntu/MachineLr/datadisk/rm_dataset/rm_datasets_en_0",
        "/home/ubuntu/MachineLr/datadisk/rm_dataset/orca_dpo_pairs",
    ]
    weights_file = [3, 1, 1, 3]
    weights_line = [20, 3, 3, 20]
    dataset = RewardModelPairDataset(
        folders,
        weights_file,
        weights_line,
        rwkv_vocab_dir="/home/ubuntu/MachineLr/RWKV-RM/vocabs/rwkv_vocab_ourborous_rm.txt",
    )
    wandb.init(
        project="rwkv_reward_model",
    )
    n = 0
    batch_pos = []
    batch_neg = []
    for e in range(10000):
        for line_dict in dataset.get_epoch():
            question, chosen_list, rejected_list = (
                line_dict["question"],
                line_dict["chosen"],
                line_dict["rejected"],
            )
            t_question = dataset.tokenizer.encode(question)
            t_chosen_list = [dataset.tokenizer.encode(chosen) for chosen in chosen_list]
            t_rejected_list = [
                dataset.tokenizer.encode(rejected) for rejected in rejected_list
            ]

            # datas = [
            #     t_question + rm_prefix + t_chosen + rm_postfix
            #     for t_chosen in t_chosen_list
            # ] + [
            #     t_question + rm_prefix + t_rejected + rm_postfix
            #     for t_rejected in t_rejected_list
            # ]
            # # 找到datas中最长字符串的长度
            # max_length = max(len(tokens) for tokens in datas)
            # if max_length> 3072:
            #     continue
            pos = [
                t_question + rm_prefix + t_chosen + rm_postfix
                for t_chosen in t_chosen_list
            ]
            neg = [
                t_question + rm_prefix + t_rejected + rm_postfix
                for t_rejected in t_rejected_list
            ]
            max_length = max(len(tokens) for tokens in pos + neg)
            if max_length > 3072:
                continue
            batch_pos += pos
            batch_neg += neg
            n += 1
            if n >= batch_size:
                n = 0
                batch = batch_pos + batch_neg
                border_idx = len(batch_pos)
                print(f"N_POS:{len(batch_pos)}|N_NEG:{len(batch_neg)}")
                batch_pos = []
                batch_neg = []
                batch = pad_to_chunk_length(batch, CHUNK_LEN)
                batch_score = model_engine(batch)

                # Loss=batch_score.mean()
                # print(f"Epoch {e}, Loss: {Loss}")

                positive_score = batch_score[:border_idx]
                negative_score = batch_score[border_idx:]
                diff = negative_score.mean() - positive_score.mean()
                Loss = 1 + diff

                print(
                    f"Epoch {e}, Loss: {Loss}, Positive Score: {positive_score.mean()}, Negative Score: {negative_score.mean()},N_BATCH:{len(batch)}"
                )
                wandb.log(
                    {
                        "Loss": Loss,
                        "Positive Score": positive_score.mean(),
                        "Negative Score": negative_score.mean(),
                    }
                )
                model_engine.backward(Loss)
                model_engine.step()
                
                gc.collect()
                torch.cuda.empty_cache()
        if e % 200 == 0:
            torch.save(
                model_engine.module.state_dict(),
                f"/home/ubuntu/MachineLr/datadisk/rwkv-checkpoint/rw_model_{e}.pth",
            )
