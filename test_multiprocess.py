#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多进程BPE功能的脚本
"""

import time
import multiprocessing as mp
from Dataset.WikiText.Loader import WikiTextDataset
from Tokenize.BPE import TokenizeBPE


def test_single_process():
    """测试单进程版本"""
    print("=== 测试单进程版本 ===")
    train_set, val_set, test_set = WikiTextDataset()

    bpe = TokenizeBPE([train_set],
                      option_wrap_sentence=True,
                      option_preserve_tokens=True,
                      preserve_tokens=[".", "?", "!", ",", ":", ";"])

    start_time = time.time()
    bpe.Process()
    single_time = time.time() - start_time

    print(f"单进程版本耗时: {single_time:.2f} 秒")
    return single_time


def test_multi_process(num_processes):
    """测试多进程版本"""
    print(f"=== 测试多进程版本 ({num_processes} 进程) ===")
    train_set, val_set, test_set = WikiTextDataset()

    bpe = TokenizeBPE([train_set],
                      option_wrap_sentence=True,
                      option_preserve_tokens=True,
                      preserve_tokens=[".", "?", "!", ",", ":", ";"])

    start_time = time.time()
    bpe.ProcessMultiProcess(num_processes=num_processes)
    multi_time = time.time() - start_time

    print(f"多进程版本耗时: {multi_time:.2f} 秒")
    return multi_time


def main():
    """主函数"""
    print("开始测试BPE多进程功能...")
    print(f"CPU核心数: {mp.cpu_count()}")

    # 测试单进程版本
    single_time = test_single_process()

    # 测试多进程版本
    multi_time = test_multi_process(4)

    # 计算加速比
    speedup = single_time / multi_time
    print(f"\n=== 性能对比 ===")
    print(f"单进程时间: {single_time:.2f} 秒")
    print(f"多进程时间: {multi_time:.2f} 秒")
    print(f"加速比: {speedup:.2f}x")
    print(f"效率: {(speedup / 4) * 100:.1f}%")


if __name__ == "__main__":
    main()
