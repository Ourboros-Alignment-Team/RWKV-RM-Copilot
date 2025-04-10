import torch
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import os
import random
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class RewardModelPairDataset:
    def __init__(self, folders, weights_file, weights_line, rwkv_vocab_dir):
        self.folders = folders
        self.weights_file = weights_file
        self.weights_line = weights_line
        self.rwkv_vocab_dir = rwkv_vocab_dir
        self.question_key = "question"
        self.chosen_key = "chosen"
        self.rejected_key = "rejected"
        self.tokenizer = TRIE_TOKENIZER(self.rwkv_vocab_dir)

    def process_folder(self, args):
        folder, weight_file, weight_line = args
        file_list = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jsonl")
        ]

        sample_files = random.choices(file_list, k=weight_file)
        local_lines = []

        for file in sample_files:
            with open(file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f]
            local_lines.extend(
                random.choices(lines, k=weight_line)
            )

        return local_lines

    @staticmethod
    def parse_json(line):
        try:
            return json.loads(line)
        except:
            return None  # 处理可能的格式错误

    def get_epoch(self):
        lines = []

        # 并行处理文件夹遍历
        with ThreadPoolExecutor() as executor:
            args = zip(self.folders, self.weights_file, self.weights_line)
            results = executor.map(self.process_folder, args)
            for result in results:
                lines.extend(result)

        random.shuffle(lines)

        # 并行处理JSON解析
        with ThreadPoolExecutor() as executor:
            lines = list(filter(None, executor.map(self.parse_json, lines)))

        return lines


class RewardModelScoreDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
