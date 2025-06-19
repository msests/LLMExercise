import math
import os
import random
import sys
import re
from collections import defaultdict
import time
from torch.utils.data import Dataset
import multiprocessing as mp
from multiprocessing import Pool, Manager
import functools
from typing import List, Dict, Tuple, Any


class TokenizeBPE:
    UNKNOWN_INDEX = 3
    START_INDEX = 0
    PAD_INDEX = 1
    END_INDEX = 2

    option_wrap_sentence: bool = False
    option_preserve_tokens: bool = False

    sentence_tokens: list[str] = ['.', '!', '?', '\n', '。', '！', '？']

    def __init__(self,
                 data_sets: list[Dataset],
                 option_wrap_sentence: bool = False,
                 option_preserve_tokens: bool = False,
                 preserve_tokens: list[str] = None):
        self.data_sets = data_sets

        self.option_wrap_sentence = option_wrap_sentence
        self.option_preserve_tokens = option_preserve_tokens

        self.token_list: list[str | tuple[int, int]] = []
        self.alphabet_map: dict[str, int] = {}

        # 只在分词时使用
        self.token_index_map: dict[str, int] = {}

        # 只在训练时使用
        self.indices: list[list[int]] = []
        self.occurrences: defaultdict[tuple[int, int], int] = defaultdict(int)
        self.most_freq_pair: tuple[int, int] = (0, 0)
        self.most_freq: int = 0
        self.vocab_size: int = 0
        self.vocab_size_limit: int = 1400
        self.sample_count: int = 32000

        # 内置特殊符号
        self.special_tokens: list[str] = [
            '<s>', '<pad>', '</s>', '<unk>', '<eos>']

        # 添加特殊符号
        for token in self.special_tokens:
            self.alphabet_map[token] = self.vocab_size
            self.token_list.append(token)
            self.vocab_size += 1

        # 添加保留符号
        self.preserve_token_indices: list[int] = []
        if preserve_tokens is not None:
            for token in preserve_tokens:
                self.alphabet_map[token] = self.vocab_size
                self.token_list.append(token)
                self.preserve_token_indices.append(self.vocab_size)
                self.vocab_size += 1

    def IsSpecialToken(self, token: int) -> bool:
        return token < len(self.special_tokens)

    def IsPreserveToken(self, index: int) -> bool:
        return index in self.preserve_token_indices

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
            # 首先写入 alphabet 的数量
            alphabet_count = len(self.alphabet_map)
            f.write(f"{alphabet_count}\n")

            # 写入token
            for token in self.token_list:
                f.write(self.GetToken(token) + '\n')

    # 从文件加载token_list
    def LoadFromFile(self, file_name: str):
        self.token_list.clear()
        self.token_index_map.clear()
        self.alphabet_map.clear()

        with open(file_name, 'r', encoding='utf-8') as f:
            # 读取第一行获取 alphabet 数量
            alphabet_count = int(f.readline().rstrip('\n'))

            # 读取 alphabet_count 行作为 alphabet
            for _ in range(alphabet_count):
                token = f.readline().rstrip('\n')
                self.alphabet_map[token] = len(self.alphabet_map)
                self.token_list.append(token)
                self.token_index_map[token] = len(self.token_list) - 1

            # 读取剩余的 token
            for line in f:
                token = line.rstrip('\n')
                self.token_list.append(token)
                self.token_index_map[token] = len(self.token_list) - 1

    def Tokenize(self, text: str) -> list[int]:
        indices = []
        token = ''

        text = self.Preprocess(text)

        if self.option_wrap_sentence:
            indices.append(self.START_INDEX)

        for char in text:
            if char not in self.alphabet_map:
                if len(token) > 0:
                    indices.append(self.token_index_map[token])
                    token = ''
                indices.append(self.UNKNOWN_INDEX)
                continue

            if self.option_wrap_sentence and char in self.sentence_tokens:
                if len(token) > 0:
                    indices.append(self.token_index_map[token])
                    token = ''
                indices.append(self.END_INDEX)
                indices.append(self.alphabet_map[char])
                indices.append(self.START_INDEX)

            temp_token = token + char
            if temp_token in self.token_index_map:
                token = temp_token
                continue
            else:
                indices.append(self.token_index_map[token])
                token = char

        if len(token) > 0:
            indices.append(self.token_index_map[token])

        if indices[-1] == self.START_INDEX:
            indices.pop()

        return indices

    def Stringify(self, indices: list[int]) -> str:
        # ANSI背景色代码
        colors = [
            '\033[41m',  # 红色背景
            '\033[42m',  # 绿色背景
            '\033[43m',  # 黄色背景
            '\033[44m',  # 蓝色背景
            '\033[45m',  # 紫色背景
            '\033[46m',  # 青色背景
        ]
        reset = '\033[0m'  # 重置颜色

        text = ''
        for i, index in enumerate(indices):
            color = colors[i % len(colors)]
            text += color + self.token_list[index] + reset
        return text

    def Merge(self, use_multiprocess: bool = False, num_processes: int = None):
        length = self.LengthOfToken(self.most_freq_pair)
        print(f"Merge {self.most_freq_pair} with length {length}")

        self.token_list.append(self.most_freq_pair)
        self.vocab_size += 1

        if use_multiprocess:
            self.MergeIndicesMultiProcess(num_processes)
        else:
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

    def Preprocess(self, text: str) -> str:
        # 将多个空格替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 将标点符号前后的空格去除
        text = re.sub(r'(\s?[.,!?;:\'"])\s?', r'\1', text)
        # 去除首尾空格
        text = text.strip()
        return text

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
            text = self.Preprocess(text)
            indices = []

            if self.option_wrap_sentence:
                indices.append(self.START_INDEX)

            for char in text:
                if char not in self.alphabet_map:
                    self.alphabet_map[char] = self.vocab_size
                    self.token_list.append(char)
                    self.vocab_size += 1

                if self.option_wrap_sentence and char in self.sentence_tokens:
                    indices.append(self.END_INDEX)
                    indices.append(self.alphabet_map[char])
                    indices.append(self.START_INDEX)
                else:
                    indices.append(self.alphabet_map[char])

            if indices[-1] == self.START_INDEX:
                indices.pop()

            self.indices.append(indices)

    def FindMaxOccurrence(self):
        self.occurrences.clear()
        self.most_freq = 0
        self.most_freq_pair = (0, 0)

        # 优化：减少字典查找次数
        for i in range(len(self.indices)):
            length = len(self.indices[i])

            for j in range(length - 1):
                # 特殊字符将作为单独的token，不参与合并
                if self.option_preserve_tokens:
                    if self.IsPreserveToken(self.indices[i][j]) or self.IsPreserveToken(self.indices[i][j + 1]):
                        continue

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

    # 静态方法，用于多进程
    @staticmethod
    def _process_sample_chunk_static(chunk_data: Tuple[int, int, List[str], Dict[str, Any]]) -> Tuple[List[List[int]], Dict[str, int]]:
        """处理样本块的静态方法，用于多进程"""
        start_idx, end_idx, sample_texts, shared_config = chunk_data

        local_alphabet_map = shared_config['alphabet_map'].copy()
        local_token_list = shared_config['token_list'].copy()
        local_vocab_size = shared_config['vocab_size']

        local_indices = []

        for i in range(start_idx, end_idx):
            if i >= len(sample_texts):
                break

            text = sample_texts[i]
            # 预处理文本
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'(\s?[.,!?;:\'"])\s?', r'\1', text)
            text = text.strip()

            indices = []

            # 添加句子开始标记（如果需要）
            # 这里简化处理，实际使用时需要根据配置决定
            # indices.append(0)  # START_INDEX

            for char in text:
                if char not in local_alphabet_map:
                    local_alphabet_map[char] = local_vocab_size
                    local_token_list.append(char)
                    local_vocab_size += 1

                # 简化句子标记处理
                # if char in ['.', '!', '?', '\n', '。', '！', '？']:
                #     indices.append(2)  # END_INDEX
                #     indices.append(local_alphabet_map[char])
                #     indices.append(0)  # START_INDEX
                # else:
                indices.append(local_alphabet_map[char])

            # 移除末尾的START_INDEX
            if indices and indices[-1] == 0:  # START_INDEX
                indices.pop()

            local_indices.append(indices)

        return local_indices, local_alphabet_map

    @staticmethod
    def _find_occurrences_chunk_static(chunk_data: Tuple[List[List[int]], bool, List[int]]) -> Dict[Tuple[int, int], int]:
        """计算频率统计的静态方法，用于多进程"""
        indices_chunk, option_preserve_tokens, preserve_token_indices = chunk_data
        local_occurrences = defaultdict(int)

        for indices in indices_chunk:
            length = len(indices)
            for j in range(length - 1):
                # 特殊字符将作为单独的token，不参与合并
                if option_preserve_tokens:
                    if (indices[j] in preserve_token_indices or
                            indices[j + 1] in preserve_token_indices):
                        continue

                # 创建token对
                token_pair = (indices[j], indices[j + 1])
                local_occurrences[token_pair] += 1

        return dict(local_occurrences)

    @staticmethod
    def _merge_indices_chunk_static(chunk_data: Tuple[List[List[int]], Tuple[int, int], int]) -> List[List[int]]:
        """合并索引的静态方法，用于多进程"""
        indices_chunk, most_freq_pair, new_token_id = chunk_data
        merged_indices = []

        for indices in indices_chunk:
            new_indices = []
            j = 0
            while j < len(indices):
                if (j < len(indices) - 1 and
                    indices[j] == most_freq_pair[0] and
                        indices[j + 1] == most_freq_pair[1]):
                    # 合并这对token
                    new_indices.append(new_token_id)
                    j += 2  # 跳过下一个token，因为已经合并了
                else:
                    new_indices.append(indices[j])
                    j += 1
            merged_indices.append(new_indices)

        return merged_indices

    # 多进程相关方法
    def CollectBasicTokensMultiProcess(self, num_processes: int = None):
        """多进程版本的CollectBasicTokens"""
        if num_processes is None:
            num_processes = mp.cpu_count()

        data_set = self.data_sets[0]
        sample_total = data_set.num_rows - data_set.num_rows % self.sample_count
        sample_range = sample_total // self.sample_count

        # 准备样本文本
        sample_texts = []
        for i in range(self.sample_count):
            lower = i * sample_range
            sample_index = -1
            while sample_index == -1 or data_set[sample_index]['text'] == '':
                sample_index = lower + \
                    math.floor(random.random() * sample_range)
            sample_texts.append(data_set[sample_index]['text'])

        # 准备共享配置
        shared_config = {
            'alphabet_map': self.alphabet_map.copy(),
            'token_list': self.token_list.copy(),
            'vocab_size': self.vocab_size
        }

        # 分块处理
        chunk_size = max(1, self.sample_count // num_processes)
        chunks = []

        for i in range(0, self.sample_count, chunk_size):
            end_idx = min(i + chunk_size, self.sample_count)
            chunks.append((i, end_idx, sample_texts, shared_config))

        # 多进程处理 - 使用静态方法
        with Pool(processes=num_processes) as pool:
            results = pool.map(
                TokenizeBPE._process_sample_chunk_static, chunks)

        # 合并结果
        all_indices = []
        merged_alphabet_map = {}
        merged_token_list = []
        max_vocab_size = self.vocab_size

        for local_indices, local_alphabet_map in results:
            all_indices.extend(local_indices)

            # 合并字母表
            for char, idx in local_alphabet_map.items():
                if char not in merged_alphabet_map:
                    merged_alphabet_map[char] = len(merged_alphabet_map)
                    merged_token_list.append(char)
                    max_vocab_size = max(
                        max_vocab_size, merged_alphabet_map[char] + 1)

        # 更新实例变量
        self.indices = all_indices
        self.alphabet_map = merged_alphabet_map
        self.token_list = merged_token_list
        self.vocab_size = max_vocab_size

    def FindMaxOccurrenceMultiProcess(self, num_processes: int = None):
        """多进程版本的FindMaxOccurrence"""
        if num_processes is None:
            num_processes = mp.cpu_count()

        # 分块处理indices
        chunk_size = max(1, len(self.indices) // num_processes)
        chunks = []

        for i in range(0, len(self.indices), chunk_size):
            end_idx = min(i + chunk_size, len(self.indices))
            indices_chunk = self.indices[i:end_idx]
            chunks.append(
                (indices_chunk, self.option_preserve_tokens, self.preserve_token_indices))

        # 多进程处理 - 使用静态方法
        with Pool(processes=num_processes) as pool:
            results = pool.map(
                TokenizeBPE._find_occurrences_chunk_static, chunks)

        # 合并结果
        self.occurrences.clear()
        self.most_freq = 0
        self.most_freq_pair = (0, 0)

        for local_occurrences in results:
            for token_pair, count in local_occurrences.items():
                self.occurrences[token_pair] += count
                if self.occurrences[token_pair] > self.most_freq:
                    self.most_freq = self.occurrences[token_pair]
                    self.most_freq_pair = token_pair

    def MergeIndicesMultiProcess(self, num_processes: int = None):
        """多进程版本的MergeIndices"""
        if num_processes is None:
            num_processes = mp.cpu_count()

        # 分块处理indices
        chunk_size = max(1, len(self.indices) // num_processes)
        chunks = []

        for i in range(0, len(self.indices), chunk_size):
            end_idx = min(i + chunk_size, len(self.indices))
            indices_chunk = self.indices[i:end_idx]
            chunks.append(
                (indices_chunk, self.most_freq_pair, self.vocab_size - 1))

        # 多进程处理 - 使用静态方法
        with Pool(processes=num_processes) as pool:
            results = pool.map(TokenizeBPE._merge_indices_chunk_static, chunks)

        # 合并结果
        merged_indices = []
        for chunk_result in results:
            merged_indices.extend(chunk_result)

        self.indices = merged_indices

    def ProcessMultiProcess(self, num_processes: int = None):
        """多进程版本的Process方法"""
        if num_processes is None:
            num_processes = mp.cpu_count()

        print(f"使用 {num_processes} 个进程进行BPE训练")

        self.CollectBasicTokensMultiProcess(num_processes)

        while self.vocab_size < self.vocab_size_limit:
            start_time = time.time()
            self.FindMaxOccurrenceMultiProcess(num_processes)
            self.Merge(use_multiprocess=True, num_processes=num_processes)
            print(
                f"Merge {self.vocab_size} tokens in {time.time() - start_time} seconds")
