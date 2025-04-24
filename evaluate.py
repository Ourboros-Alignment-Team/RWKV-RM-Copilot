
from types import SimpleNamespace
import json
import os
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

config = load_config(
    "/home/ubuntu/MachineLr/RWKV-RM-Copilot/configs/eval_config.json",
)
os.environ["RWKV_REWARD_MODEL_HEAD_SIZE_A"] = str(
    config.deep_reward_model.model.head_size
)
os.environ["RWKV_CTXLEN"] = str(config.deep_reward_model.model.ctx_len)
CHUNK_LEN = config.deep_reward_model.model.CHUNK_LEN
os.environ["CHUNK_LEN"] = str(CHUNK_LEN)
from model.model import RWKVRewardModel
model = RWKVRewardModel(config.deep_reward_model)
model.to(config.deep_reward_model.model.device)
model.eval()
import gradio as gr
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer=TRIE_TOKENIZER("/home/ubuntu/MachineLr/RWKV-RM-Copilot/vocabs/rwkv_vocab_ourborous_rm.txt")

rm_prefix = [
    11,
    65520,
]
rm_postfix = [
    11,
    65521,
]
def evaluate_reward(history, response, score):
    input_tokens=[tokenizer.encode(history)+rm_prefix+tokenizer.encode(response)+rm_postfix]
    
    remainder = len(input_tokens[0]) % CHUNK_LEN
    if remainder != 0:
        additional_padding = CHUNK_LEN - remainder
        input_tokens = [[0] * additional_padding + tokens for tokens in input_tokens]
    score=model(input_tokens).squeeze().item()*2-1
    
    return score

with gr.Blocks(title="语言强化学习奖励模型测试") as demo:
    gr.Markdown("# 语言强化学习奖励模型测试")
    gr.Markdown("输入对话历史和模型回复，评估回复质量")
    
    history = gr.Textbox(label="历史信息", lines=5, placeholder="请输入对话历史...")
    response = gr.Textbox(label="模型回复", lines=2, placeholder="请输入模型回复...")
    submit_btn = gr.Button("评估")
    score = gr.Number(label="评分", value=0.0, step=0.1, minimum=0, maximum=10)
        
    submit_btn.click(
        fn=evaluate_reward,
        inputs=[history, response],
        outputs=[score]
    )
    

demo.launch(server_name="10.0.5.101")