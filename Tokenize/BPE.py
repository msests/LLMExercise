import math
import os
import random
import sys
from collections import defaultdict
import time
from torch.utils.data import Dataset


class TokenizeBPE:
    def __init__(self, data_sets: list[Dataset]):
        self.data_sets = data_sets

        self.token_list: list[str | tuple[int, int]] = []
        self.token_index_map: dict[str, int] = {}

        # 只在训练时使用
        self.alphabet_map: dict[str, int] = {}
        self.indices: list[list[int]] = []
        self.occurrences: defaultdict[tuple[int, int], int] = defaultdict(int)
        self.most_freq_pair: tuple[int, int] = (0, 0)
        self.most_freq: int = 0
        self.vocab_size: int = 0
        self.vocab_size_limit: int = 4000
        self.sample_count: int = 32000
        self.special_tokens: list[str] = ['<s>', '<pad>', '</s>', '<unk>']

        for token in self.special_tokens:
            self.alphabet_map[token] = self.vocab_size
            self.token_list.append(token)
            self.vocab_size += 1

    def LengthOfToken(self, token: str | tuple[int, int]) -> int:
        if isinstance(token, str):
            return 1
        else:
            return self.LengthOfToken(self.token_list[token[0]]) + self.LengthOfToken(self.token_list[token[1]])

    def GetToken(self, token: str | tuple[int, int]) -> str:
        if isinstance(token, str):
            return token
        else:
            return self.GetToken(self.token_list[token[0]]) + self.GetToken(self.token_list[token[1]])

    def PrintTokenList(self):
        for token in self.token_list:
            print(self.GetToken(token))

    # 将token_list保存到文件
    def SaveToFile(self, file_name: str):
        with open(file_name, 'w', encoding='utf-8') as f:
            for token in self.token_list:
                f.write(self.GetToken(token) + '\n')

    # 从文件加载token_list
    def LoadFromFile(self, file_name: str):
        self.token_list.clear()
        self.token_index_map.clear()

        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                token = line
                self.token_list.append(token)
                self.token_index_map[token] = len(self.token_list) - 1

    def Tokenize(self, text: str) -> list[int]:
        indices = []
        token = ''
        for char in text:
            temp_token = token + char
            if temp_token in self.token_index_map:
                token = temp_token
                continue
            else:
                indices.append(self.token_index_map[token])
                token = char
        indices.append(self.token_index_map[token])
        return indices

    def Merge(self):
        length = self.LengthOfToken(self.most_freq_pair)
        print(f"Merge {self.most_freq_pair} with length {length}")

        self.token_list.append(self.most_freq_pair)
        self.vocab_size += 1
        self.MergeIndices()

    def MergeIndices(self):
        for i in range(len(self.indices)):
            new_indices = []
            j = 0
            while j < len(self.indices[i]):
                if (j < len(self.indices[i]) - 1 and
                    self.indices[i][j] == self.most_freq_pair[0] and
                        self.indices[i][j + 1] == self.most_freq_pair[1]):
                    # 合并这对token
                    new_indices.append(self.vocab_size - 1)  # 使用新的合并token ID
                    j += 2  # 跳过下一个token，因为已经合并了
                else:
                    new_indices.append(self.indices[i][j])
                    j += 1
            self.indices[i] = new_indices

    def CollectBasicTokens(self):
        data_set = self.data_sets[0]
        sample_total = data_set.num_rows - data_set.num_rows % self.sample_count
        sample_range = sample_total // self.sample_count

        for i in range(self.sample_count):
            lower = i * sample_range

            sample_index = -1
            while sample_index == -1 or data_set[sample_index]['text'] == '':
                sample_index = lower + \
                    math.floor(random.random() * sample_range)

            text = data_set[sample_index]['text']
            indices = []
            for char in text:
                if char not in self.alphabet_map:
                    self.alphabet_map[char] = self.vocab_size
                    self.token_list.append(char)
                    self.vocab_size += 1
                indices.append(self.alphabet_map[char])
            self.indices.append(indices)

    def FindMaxOccurrence(self):
        self.occurrences.clear()
        self.most_freq = 0
        self.most_freq_pair = (0, 0)

        # 优化：减少字典查找次数
        for i in range(len(self.indices)):
            length = len(self.indices[i])

            for j in range(length - 1):
                # 创建token对，只计算一次
                token_pair = (self.indices[i][j], self.indices[i][j + 1])

                # 使用defaultdict，自动处理不存在的键
                self.occurrences[token_pair] += 1

                # 只在当前计数大于最大频率时才更新
                current_count = self.occurrences[token_pair]

                if current_count > self.most_freq:
                    self.most_freq = current_count
                    self.most_freq_pair = token_pair

    def Process(self):
        self.CollectBasicTokens()

        while self.vocab_size < self.vocab_size_limit:
            start_time = time.time()
            self.FindMaxOccurrence()
            self.Merge()
            print(
                f"Merge {self.vocab_size} tokens in {time.time() - start_time} seconds")
